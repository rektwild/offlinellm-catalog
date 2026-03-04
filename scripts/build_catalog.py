#!/usr/bin/env python3
"""Build catalog.json from curated seed models and Hugging Face metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CURATED_PATH = ROOT_DIR / "curated_models.yaml"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "catalog.json"
HF_API_BASE = "https://huggingface.co/api/models"
HF_WEB_BASE = "https://huggingface.co"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "offlinellm-catalog-builder/1.0"

REQUIRED_CURATED_FIELDS = {
    "id",
    "name",
    "family",
    "group",
    "tags",
    "minRamGb",
    "devices",
    "description",
    "contextLength",
    "hfRepo",
}


@dataclass(frozen=True)
class CuratedModel:
    id: str
    name: str
    family: str
    group: str
    tags: list[str]
    min_ram_gb: int
    devices: list[str]
    description: str
    context_length: int
    hf_repo: str
    file_name: str | None
    file_regex: str | None
    quantization: str | None
    sha256: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CuratedModel":
        missing = REQUIRED_CURATED_FIELDS - set(raw.keys())
        if missing:
            raise ValueError(f"Missing curated fields for model: {sorted(missing)}")

        file_name = raw.get("fileName")
        file_regex = raw.get("fileRegex")
        if not file_name and not file_regex:
            raise ValueError(f"Model '{raw.get('id')}' must define fileName or fileRegex")

        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            family=str(raw["family"]),
            group=str(raw["group"]),
            tags=[str(tag) for tag in raw["tags"]],
            min_ram_gb=int(raw["minRamGb"]),
            devices=[str(device) for device in raw["devices"]],
            description=str(raw["description"]),
            context_length=int(raw["contextLength"]),
            hf_repo=str(raw["hfRepo"]),
            file_name=str(file_name) if file_name else None,
            file_regex=str(file_regex) if file_regex else None,
            quantization=str(raw["quantization"]) if raw.get("quantization") else None,
            sha256=str(raw["sha256"]) if raw.get("sha256") else None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build catalog.json from curated_models.yaml")
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_curated_models(path: Path) -> list[CuratedModel]:
    if not path.exists():
        raise FileNotFoundError(f"Curated file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("curated_models.yaml must contain a top-level 'models' list")

    models: list[CuratedModel] = []
    for entry in data["models"]:
        if not isinstance(entry, dict):
            raise ValueError("Each curated model entry must be an object")
        models.append(CuratedModel.from_dict(entry))

    return models


def build_headers(token: str | None, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def fetch_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url=url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while requesting {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error while requesting {url}: {exc.reason}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload from {url}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON shape from {url}")

    return data


def fetch_hf_siblings(model: CuratedModel, token: str | None) -> list[dict[str, Any]]:
    encoded_repo = urllib.parse.quote(model.hf_repo, safe="/")
    url = f"{HF_API_BASE}/{encoded_repo}?expand[]=siblings"
    data = fetch_json(url, headers=build_headers(token))

    siblings = data.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError(f"No siblings list found in HF API response for {model.hf_repo}")

    normalized: list[dict[str, Any]] = []
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        filename = sibling.get("rfilename")
        if isinstance(filename, str) and filename:
            normalized.append(sibling)

    if not normalized:
        raise RuntimeError(f"No downloadable files found for {model.hf_repo}")

    return normalized


def select_file(model: CuratedModel, siblings: list[dict[str, Any]]) -> dict[str, Any]:
    if model.file_name:
        for sibling in siblings:
            if sibling.get("rfilename") == model.file_name:
                return sibling
        raise RuntimeError(
            f"Configured fileName '{model.file_name}' not found in repo {model.hf_repo}"
        )

    assert model.file_regex is not None
    pattern = re.compile(model.file_regex)
    matches = [sibling for sibling in siblings if pattern.search(str(sibling.get("rfilename", "")))]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise RuntimeError(
            f"No files matched fileRegex '{model.file_regex}' in repo {model.hf_repo}"
        )

    match_names = ", ".join(str(match.get("rfilename")) for match in matches)
    raise RuntimeError(
        f"fileRegex '{model.file_regex}' matched multiple files in {model.hf_repo}: {match_names}"
    )


def build_download_url(hf_repo: str, filename: str) -> str:
    encoded_repo = urllib.parse.quote(hf_repo, safe="/")
    encoded_file = urllib.parse.quote(filename, safe="/")
    return f"{HF_WEB_BASE}/{encoded_repo}/resolve/main/{encoded_file}"


def parse_size_from_sibling(sibling: dict[str, Any]) -> int | None:
    size = sibling.get("size")
    if isinstance(size, int) and size > 0:
        return size

    lfs = sibling.get("lfs")
    if isinstance(lfs, dict):
        lfs_size = lfs.get("size")
        if isinstance(lfs_size, int) and lfs_size > 0:
            return lfs_size

    return None


def parse_content_range_total(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"/(\d+)$", value)
    if not match:
        return None
    return int(match.group(1))


def fetch_size_via_http(download_url: str, token: str | None) -> int:
    base_headers = build_headers(token, extra={"Accept": "*/*"})

    head_request = urllib.request.Request(download_url, headers=base_headers, method="HEAD")
    try:
        with urllib.request.urlopen(head_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit():
                return int(length)
    except urllib.error.HTTPError as exc:
        if exc.code not in {405, 501}:
            raise RuntimeError(f"HEAD request failed for {download_url} with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HEAD request failed for {download_url}: {exc.reason}") from exc

    range_headers = build_headers(token, extra={"Accept": "*/*", "Range": "bytes=0-0"})
    range_request = urllib.request.Request(download_url, headers=range_headers, method="GET")
    try:
        with urllib.request.urlopen(range_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            total = parse_content_range_total(response.headers.get("Content-Range"))
            if total and total > 0:
                return total

            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > 0:
                return int(length)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Range request failed for {download_url} with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Range request failed for {download_url}: {exc.reason}") from exc

    raise RuntimeError(f"Could not determine sizeBytes for {download_url}")


def infer_quantization(filename: str) -> str:
    match = re.search(r"(Q\d(?:_[A-Za-z0-9]+)*)", filename, re.IGNORECASE)
    if not match:
        raise RuntimeError(
            "Quantization is missing and could not be inferred from filename "
            f"'{filename}'. Add quantization to curated_models.yaml."
        )
    return match.group(1).upper()


def build_catalog_model(model: CuratedModel, token: str | None) -> dict[str, Any]:
    siblings = fetch_hf_siblings(model, token)
    selected = select_file(model, siblings)

    selected_file = str(selected.get("rfilename"))
    download_url = build_download_url(model.hf_repo, selected_file)

    size_bytes = parse_size_from_sibling(selected)
    if size_bytes is None:
        size_bytes = fetch_size_via_http(download_url, token)

    quantization = model.quantization or infer_quantization(selected_file)

    return {
        "id": model.id,
        "name": model.name,
        "family": model.family,
        "quantization": quantization,
        "sizeBytes": size_bytes,
        "contextLength": model.context_length,
        "downloadURL": download_url,
        "sha256": model.sha256,
        "group": model.group,
        "tags": model.tags,
        "minRamGb": model.min_ram_gb,
        "devices": model.devices,
        "description": model.description,
    }


def load_existing_catalog(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(f"Existing catalog at {path} must be an object")

    return data


def normalized_models(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    models: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            models.append(item)

    return sorted(models, key=lambda model: str(model.get("id", "")))


def compute_version(existing_catalog: dict[str, Any] | None, new_models: list[dict[str, Any]]) -> int:
    if existing_catalog is None:
        return 1

    existing_version = existing_catalog.get("version")
    if not isinstance(existing_version, int) or existing_version < 1:
        raise RuntimeError("Existing catalog version must be a positive integer")

    previous_models = normalized_models(existing_catalog.get("models"))
    if previous_models == new_models:
        return existing_version

    return existing_version + 1


def utc_today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def write_catalog(path: Path, catalog: dict[str, Any]) -> None:
    serialized = json.dumps(catalog, indent=2, ensure_ascii=False)
    path.write_text(serialized + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    curated_models = load_curated_models(args.curated)
    generated_models: list[dict[str, Any]] = []

    for model in curated_models:
        print(f"[build] resolving {model.id} from {model.hf_repo}", file=sys.stderr)
        generated_models.append(build_catalog_model(model, token))

    generated_models = sorted(generated_models, key=lambda item: item["id"])

    existing_catalog = load_existing_catalog(args.output)
    version = compute_version(existing_catalog, generated_models)

    catalog = {
        "version": version,
        "updatedAt": utc_today(),
        "models": generated_models,
    }

    write_catalog(args.output, catalog)
    print(
        f"[build] wrote {args.output} with version={version}, models={len(generated_models)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[build] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
