#!/usr/bin/env python3
"""Refresh curated_models.yaml by auto-discovering GGUF models from Hugging Face."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT_DIR / "model_discovery_policy.yaml"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "curated_models.yaml"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "offlinellm-catalog-discovery/1.0"
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
RETRIABLE_HTTP_CODES = {429, 500, 502, 503, 504}


@dataclass
class DiscoveryPolicy:
    endpoint: str
    search: str
    model_filter: str
    sort: str
    direction: str
    limit: int
    full: bool
    target_count: int
    minimum_count: int
    minimum_recommended_count: int
    fallback_context_length: int
    likes_weight: int
    bucket_targets: dict[str, int]
    trusted_authors: set[str]
    min_downloads: int
    min_likes: int
    recency_from: dt.date
    blocklist_repo_substrings: list[str]
    base_model_owner_allowlist: set[str]
    quantization_priority: list[str]
    size_min_gb: float
    size_max_gb: float
    reject_path_contains: list[str]
    reject_path_regex: list[re.Pattern[str]]
    segmentation: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DiscoveryPolicy":
        source = raw.get("source") or {}
        selection = raw.get("selection") or {}
        filters = raw.get("filters") or {}
        files = raw.get("files") or {}
        segmentation = raw.get("segmentation") or {}

        recency_from_raw = str(filters.get("recencyFrom", "2024-01-01"))
        recency_from = dt.date.fromisoformat(recency_from_raw)

        size = filters.get("sizeGb") or {}
        reject_regex = [re.compile(str(item), re.IGNORECASE) for item in files.get("rejectPathRegex", [])]

        return cls(
            endpoint=str(source.get("endpoint")),
            search=str(source.get("search", "gguf")),
            model_filter=str(source.get("filter", "text-generation")),
            sort=str(source.get("sort", "downloads")),
            direction=str(source.get("direction", "-1")),
            limit=int(source.get("limit", 300)),
            full=bool(source.get("full", True)),
            target_count=int(selection.get("targetCount", 24)),
            minimum_count=int(selection.get("minimumCount", 20)),
            minimum_recommended_count=int(selection.get("minimumRecommendedCount", 8)),
            fallback_context_length=int(selection.get("fallbackContextLength", 8192)),
            likes_weight=int((selection.get("popularityScore") or {}).get("likesWeight", 250)),
            bucket_targets={
                str(key): int(value)
                for key, value in (selection.get("bucketTargets") or {}).items()
            },
            trusted_authors={str(author).lower() for author in filters.get("trustedAuthors", [])},
            min_downloads=int(filters.get("minDownloads", 0)),
            min_likes=int(filters.get("minLikes", 0)),
            recency_from=recency_from,
            blocklist_repo_substrings=[str(value).lower() for value in filters.get("blocklistRepoSubstrings", [])],
            base_model_owner_allowlist={
                str(owner).lower() for owner in filters.get("baseModelOwnerAllowlist", [])
            },
            quantization_priority=[str(item).upper() for item in filters.get("quantizationPriority", [])],
            size_min_gb=float(size.get("min", 0.0)),
            size_max_gb=float(size.get("max", 100.0)),
            reject_path_contains=[str(item).lower() for item in files.get("rejectPathContains", [])],
            reject_path_regex=reject_regex,
            segmentation={str(key): value for key, value in segmentation.items()},
        )


@dataclass
class CandidateModel:
    repo_id: str
    author: str
    downloads: int
    likes: int
    last_modified: str
    base_model: str | None
    family: str
    context_length: int
    file_name: str
    quantization: str
    size_bytes: int
    sha256: str | None
    candidate_key: str
    score: int
    bucket: str
    retained: bool = False

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1_073_741_824


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh curated_models.yaml from Hugging Face")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_headers(token: str | None, *, accept: str = "application/json") -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def sleep_for_retry(attempt: int) -> None:
    index = min(attempt, len(RETRY_DELAYS_SECONDS) - 1)
    delay = RETRY_DELAYS_SECONDS[index] + random.uniform(0.0, 0.35)
    time.sleep(delay)


def fetch_json(url: str, token: str | None) -> Any:
    last_error: Exception | None = None
    attempts = len(RETRY_DELAYS_SECONDS) + 1

    for attempt in range(attempts):
        request = urllib.request.Request(url=url, headers=build_headers(token), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            retriable = exc.code in RETRIABLE_HTTP_CODES
            last_error = exc
            if retriable and attempt < attempts - 1:
                sleep_for_retry(attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code} while requesting {url}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < attempts - 1:
                sleep_for_retry(attempt)
                continue
            raise RuntimeError(f"Network error while requesting {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON payload from {url}") from exc

    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return data


def parse_recency(value: str) -> dt.date | None:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError:
        return None


def normalize_base_model(value: str | None, fallback_repo: str) -> str:
    if value and value.strip():
        cleaned = value.strip()
        cleaned = re.sub(r"^quantized:", "", cleaned, flags=re.IGNORECASE)
        return cleaned
    return fallback_repo


def to_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return slug.strip("-")


def generate_model_id(base_or_repo: str, quantization: str) -> str:
    return f"{to_slug(base_or_repo)}-{to_slug(quantization)}"


def infer_family(base_model: str, repo_id: str) -> str:
    key = f"{base_model} {repo_id}".lower()
    if "qwen" in key:
        return "qwen"
    if "gemma" in key:
        return "gemma"
    if "mistral" in key or "mixtral" in key:
        return "mistral"
    if "phi" in key:
        return "phi3"
    if "llama" in key:
        return "llama3"
    if "deepseek" in key:
        return "deepseek"
    if "gpt-oss" in key or "gptoss" in key:
        return "gpt-oss"
    return "general"


def humanize_name(base_model: str | None, repo_id: str) -> str:
    source = base_model or repo_id
    tail = source.split("/")[-1]
    text = re.sub(r"[-_]+", " ", tail).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return repo_id
    return text


def parse_base_model(card_data: dict[str, Any]) -> str | None:
    base = card_data.get("base_model")
    if isinstance(base, list) and base:
        first = base[0]
        if isinstance(first, str):
            return first
    if isinstance(base, str):
        return base
    return None


def infer_bucket(size_gb: float, policy: DiscoveryPolicy) -> str:
    small_max = float((policy.segmentation.get("small") or {}).get("maxSizeGb", 3.5))
    medium_max = float((policy.segmentation.get("medium") or {}).get("maxSizeGb", 8.0))
    if size_gb <= small_max:
        return "small"
    if size_gb <= medium_max:
        return "medium"
    return "large"


def map_bucket_to_constraints(bucket: str, size_gb: float, policy: DiscoveryPolicy) -> tuple[str, list[str], int]:
    if bucket == "small":
        seg = policy.segmentation.get("small") or {}
        group = str(seg.get("group", "recommended"))
        devices = [str(item) for item in seg.get("devices", ["iphone", "ipad", "mac"])]
        min_ram = int(seg.get("minRamGb", 4))
        return group, devices, min_ram

    if bucket == "medium":
        seg = policy.segmentation.get("medium") or {}
        threshold = float(seg.get("lowRamThresholdGb", 4.5))
        min_ram = int(seg.get("minRamGbLow", 6)) if size_gb <= threshold else int(
            seg.get("minRamGbHigh", 8)
        )
        group = str(seg.get("group", "advanced"))
        devices = [str(item) for item in seg.get("devices", ["ipad", "mac"])]
        return group, devices, min_ram

    seg = policy.segmentation.get("large") or {}
    threshold = float(seg.get("highRamThresholdGb", 16.0))
    min_ram = int(seg.get("minRamGbMid", 12)) if size_gb <= threshold else int(
        seg.get("minRamGbHigh", 16)
    )
    group = str(seg.get("group", "advanced"))
    devices = [str(item) for item in seg.get("devices", ["mac"])]
    return group, devices, min_ram


def make_tags(candidate: CandidateModel) -> list[str]:
    tags: list[str] = [candidate.family]

    if candidate.bucket == "small":
        tags.append("mobile")
    elif candidate.bucket == "medium":
        tags.append("balanced")
    else:
        tags.append("powerful")

    repo_lower = candidate.repo_id.lower()
    if "coder" in repo_lower or "coding" in repo_lower:
        tags.append("coding")
    if "instruct" in repo_lower or "chat" in repo_lower:
        tags.append("instruct")
    if candidate.context_length >= 65_536:
        tags.append("long-context")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def make_description(candidate: CandidateModel) -> str:
    return (
        "Auto-curated from Hugging Face. "
        f"{candidate.quantization} GGUF (~{candidate.size_gb:.1f} GB) with "
        f"{candidate.context_length} context length."
    )


def matches_quant(path: str, quantization: str) -> bool:
    return re.search(rf"(?<![A-Z0-9]){re.escape(quantization)}(?![A-Z0-9])", path.upper()) is not None


def select_quantized_gguf(
    tree_nodes: list[dict[str, Any]],
    quant_priority: list[str],
    reject_contains: list[str],
    reject_regex: list[re.Pattern[str]],
) -> tuple[dict[str, Any], str] | None:
    candidates: list[dict[str, Any]] = []
    for node in tree_nodes:
        if not isinstance(node, dict):
            continue
        path = node.get("path")
        if not isinstance(path, str) or not path.lower().endswith(".gguf"):
            continue
        path_lower = path.lower()
        if any(token in path_lower for token in reject_contains):
            continue
        if any(pattern.search(path) for pattern in reject_regex):
            continue
        if node.get("type") not in {None, "file"}:
            continue
        candidates.append(node)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda item: str(item.get("path", "")))
    for quant in quant_priority:
        for node in candidates:
            path = str(node.get("path", ""))
            if matches_quant(path, quant):
                return node, quant

    return None


def owner_from_model_ref(model_ref: str | None, fallback_repo: str) -> str:
    if model_ref and "/" in model_ref:
        return model_ref.split("/", 1)[0].lower()
    if "/" in fallback_repo:
        return fallback_repo.split("/", 1)[0].lower()
    return fallback_repo.lower()


def normalize_candidate_key(base_model: str | None, fallback_repo: str) -> str:
    normalized = normalize_base_model(base_model, fallback_repo)
    normalized = normalized.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def is_blocklisted(repo_id: str, blocklist: list[str]) -> bool:
    lower = repo_id.lower()
    return any(token in lower for token in blocklist)


def load_existing_repos(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = load_yaml(path)
    models = data.get("models")
    if not isinstance(models, list):
        return set()
    repos: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        repo = model.get("hfRepo")
        if isinstance(repo, str) and repo.strip():
            repos.add(repo.strip().lower())
    return repos


def build_list_url(policy: DiscoveryPolicy) -> str:
    params = {
        "search": policy.search,
        "filter": policy.model_filter,
        "sort": policy.sort,
        "direction": policy.direction,
        "limit": str(policy.limit),
        "full": "true" if policy.full else "false",
    }
    return f"{policy.endpoint}?{urllib.parse.urlencode(params)}"


def resolve_candidate(
    summary: dict[str, Any],
    policy: DiscoveryPolicy,
    token: str | None,
) -> CandidateModel | None:
    repo_id = summary.get("id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        return None
    repo_id = repo_id.strip()
    author = str(summary.get("author") or repo_id.split("/", 1)[0]).strip()
    author_lower = author.lower()

    if author_lower not in policy.trusted_authors:
        return None
    if is_blocklisted(repo_id, policy.blocklist_repo_substrings):
        return None
    if bool(summary.get("private")) or bool(summary.get("gated")) or bool(summary.get("disabled")):
        return None

    downloads = int(summary.get("downloads") or 0)
    likes = int(summary.get("likes") or 0)
    if downloads < policy.min_downloads or likes < policy.min_likes:
        return None

    last_modified = str(summary.get("lastModified") or "")
    recency = parse_recency(last_modified)
    if recency and recency < policy.recency_from:
        return None

    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    detail_url = f"{policy.endpoint}/{encoded_repo}"
    detail = fetch_json(detail_url, token)
    if not isinstance(detail, dict):
        return None

    card_data = detail.get("cardData")
    if not isinstance(card_data, dict):
        card_data = {}

    base_model = parse_base_model(card_data)
    base_owner = owner_from_model_ref(base_model, repo_id)
    if policy.base_model_owner_allowlist and base_owner not in policy.base_model_owner_allowlist:
        return None

    tree_url = f"{policy.endpoint}/{encoded_repo}/tree/main?recursive=1"
    tree = fetch_json(tree_url, token)
    if not isinstance(tree, list):
        return None

    selected = select_quantized_gguf(
        tree_nodes=[item for item in tree if isinstance(item, dict)],
        quant_priority=policy.quantization_priority,
        reject_contains=policy.reject_path_contains,
        reject_regex=policy.reject_path_regex,
    )
    if selected is None:
        return None

    selected_node, quantization = selected
    path = selected_node.get("path")
    size = selected_node.get("size")
    if not isinstance(path, str) or not path:
        return None
    if not isinstance(size, int) or size <= 0:
        return None

    size_gb = size / 1_073_741_824
    if size_gb < policy.size_min_gb or size_gb > policy.size_max_gb:
        return None

    lfs = selected_node.get("lfs")
    sha256: str | None = None
    if isinstance(lfs, dict):
        oid = lfs.get("oid")
        if isinstance(oid, str) and re.fullmatch(r"[0-9a-fA-F]{64}", oid):
            sha256 = oid.lower()

    gguf = detail.get("gguf")
    if not isinstance(gguf, dict):
        gguf = {}

    context_length = gguf.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        context_length = policy.fallback_context_length

    family = infer_family(base_model or repo_id, repo_id)
    candidate_key = normalize_candidate_key(base_model, repo_id)
    score = downloads + likes * policy.likes_weight
    bucket = infer_bucket(size_gb, policy)

    return CandidateModel(
        repo_id=repo_id,
        author=author,
        downloads=downloads,
        likes=likes,
        last_modified=last_modified,
        base_model=base_model,
        family=family,
        context_length=context_length,
        file_name=path,
        quantization=quantization,
        size_bytes=size,
        sha256=sha256,
        candidate_key=candidate_key,
        score=score,
        bucket=bucket,
    )


def dedupe_candidates(candidates: list[CandidateModel], existing_repos: set[str]) -> list[CandidateModel]:
    grouped: dict[str, list[CandidateModel]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.candidate_key, []).append(candidate)

    winners: list[CandidateModel] = []
    for candidate_key in sorted(grouped.keys()):
        group = grouped[candidate_key]
        group_sorted = sorted(
            group,
            key=lambda item: (-item.score, -item.downloads, -item.likes, item.repo_id),
        )
        retained_choices = [item for item in group_sorted if item.repo_id.lower() in existing_repos]
        chosen = retained_choices[0] if retained_choices else group_sorted[0]
        chosen.retained = chosen.repo_id.lower() in existing_repos
        winners.append(chosen)

    return winners


def select_with_bucket_targets(candidates: list[CandidateModel], policy: DiscoveryPolicy) -> list[CandidateModel]:
    target_count = policy.target_count
    bucket_targets = {
        "small": int(policy.bucket_targets.get("small", 0)),
        "medium": int(policy.bucket_targets.get("medium", 0)),
        "large": int(policy.bucket_targets.get("large", 0)),
    }

    by_bucket: dict[str, list[CandidateModel]] = {"small": [], "medium": [], "large": []}
    for candidate in candidates:
        by_bucket.setdefault(candidate.bucket, []).append(candidate)

    for bucket in by_bucket:
        by_bucket[bucket] = sorted(
            by_bucket[bucket],
            key=lambda item: (-item.score, -item.downloads, -item.likes, item.repo_id),
        )

    selected: list[CandidateModel] = []
    selected_ids: set[str] = set()

    # 1) Retained first inside each bucket.
    for bucket in ("small", "medium", "large"):
        target = bucket_targets.get(bucket, 0)
        retained = [item for item in by_bucket.get(bucket, []) if item.retained]
        for candidate in retained:
            if len([item for item in selected if item.bucket == bucket]) >= target:
                break
            if candidate.candidate_key in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.candidate_key)

    # 2) Fill bucket quotas.
    for bucket in ("small", "medium", "large"):
        target = bucket_targets.get(bucket, 0)
        for candidate in by_bucket.get(bucket, []):
            if len([item for item in selected if item.bucket == bucket]) >= target:
                break
            if candidate.candidate_key in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.candidate_key)

    # 3) Fill remaining slots by score regardless of bucket.
    ranked = sorted(
        candidates,
        key=lambda item: (-item.score, -item.downloads, -item.likes, item.repo_id),
    )
    for candidate in ranked:
        if len(selected) >= target_count:
            break
        if candidate.candidate_key in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.candidate_key)

    # 4) Trim if over target_count, keeping retained first.
    if len(selected) > target_count:
        selected = sorted(
            selected,
            key=lambda item: (item.retained is False, -item.score, item.repo_id),
        )[:target_count]

    bucket_order = {"small": 0, "medium": 1, "large": 2}
    return sorted(selected, key=lambda item: (bucket_order.get(item.bucket, 99), -item.score, item.repo_id))


def build_curated_models(selected: list[CandidateModel], policy: DiscoveryPolicy) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for candidate in selected:
        base_or_repo = normalize_base_model(candidate.base_model, candidate.repo_id)
        model_id = generate_model_id(base_or_repo, candidate.quantization)
        if model_id in used_ids:
            suffix = 2
            while f"{model_id}-{suffix}" in used_ids:
                suffix += 1
            model_id = f"{model_id}-{suffix}"
        used_ids.add(model_id)

        group, devices, min_ram = map_bucket_to_constraints(candidate.bucket, candidate.size_gb, policy)

        models.append(
            {
                "id": model_id,
                "name": humanize_name(candidate.base_model, candidate.repo_id),
                "family": candidate.family,
                "group": group,
                "tags": make_tags(candidate),
                "minRamGb": min_ram,
                "devices": devices,
                "description": make_description(candidate),
                "contextLength": candidate.context_length,
                "hfRepo": candidate.repo_id,
                "fileName": candidate.file_name,
                "quantization": candidate.quantization,
                "sha256": candidate.sha256,
            }
        )

    return models


def write_curated(path: Path, models: list[dict[str, Any]]) -> None:
    payload = {"models": models}
    serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(serialized, encoding="utf-8")


def emit_summary(selected: list[CandidateModel], models_written: list[dict[str, Any]]) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    model_by_repo = {item["hfRepo"]: item for item in models_written}
    lines: list[str] = []
    lines.append("## Auto-curated Models")
    lines.append("")
    lines.append(f"- Selected models: **{len(selected)}**")
    lines.append("")
    lines.append("| Bucket | ID | Repo | Size (GB) | Downloads | Likes | Score | Retained |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for candidate in selected:
        generated = model_by_repo.get(candidate.repo_id)
        model_id = generated["id"] if generated else "-"
        lines.append(
            f"| {candidate.bucket} | `{model_id}` | `{candidate.repo_id}` | "
            f"{candidate.size_gb:.2f} | {candidate.downloads} | {candidate.likes} | "
            f"{candidate.score} | {'yes' if candidate.retained else 'no'} |"
        )

    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def validate_selection(selected: list[CandidateModel], policy: DiscoveryPolicy) -> None:
    if len(selected) < policy.minimum_count:
        raise RuntimeError(
            f"Selection produced only {len(selected)} models, expected at least {policy.minimum_count}"
        )

    recommended_count = sum(1 for item in selected if item.bucket == "small")
    if recommended_count < policy.minimum_recommended_count:
        raise RuntimeError(
            f"Selection produced {recommended_count} recommended models, expected at least "
            f"{policy.minimum_recommended_count}"
        )


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    policy_raw = load_yaml(args.policy)
    policy = DiscoveryPolicy.from_dict(policy_raw)
    existing_repos = load_existing_repos(args.output)

    list_url = build_list_url(policy)
    list_payload = fetch_json(list_url, token)
    if not isinstance(list_payload, list):
        raise RuntimeError("HF list endpoint returned an unexpected payload")

    candidates: list[CandidateModel] = []
    for index, summary in enumerate(list_payload, start=1):
        if not isinstance(summary, dict):
            continue
        try:
            candidate = resolve_candidate(summary, policy, token)
        except Exception as exc:  # noqa: BLE001
            repo_id = summary.get("id", "unknown")
            print(f"[discover] skipped {repo_id}: {exc}", file=sys.stderr)
            continue
        if candidate is None:
            continue
        candidates.append(candidate)
        print(
            f"[discover] accepted {candidate.repo_id} ({candidate.quantization}, {candidate.size_gb:.2f} GB)",
            file=sys.stderr,
        )
        if index % 25 == 0:
            print(f"[discover] scanned {index} list entries...", file=sys.stderr)

    deduped = dedupe_candidates(candidates, existing_repos=existing_repos)
    selected = select_with_bucket_targets(deduped, policy)
    validate_selection(selected, policy)

    models_written = build_curated_models(selected, policy)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected": len(selected),
                    "recommended": sum(1 for item in selected if item.bucket == "small"),
                    "repos": [item.repo_id for item in selected],
                },
                indent=2,
            )
        )
        return 0

    write_curated(args.output, models_written)
    emit_summary(selected, models_written)
    print(
        f"[discover] wrote {args.output} with {len(models_written)} models "
        f"(recommended={sum(1 for item in selected if item.bucket == 'small')})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[discover] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
