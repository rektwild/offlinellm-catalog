#!/usr/bin/env python3
"""Validate embedding_catalog.json against schema and runtime expectations."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT_DIR / "embedding_catalog.json"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "schema" / "embedding_catalog.schema.json"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "offlinellm-embedding-catalog-validator/1.0"

REQUIRED_MODEL_FIELDS = [
    "id",
    "name",
    "family",
    "quantization",
    "sizeBytes",
    "contextLength",
    "downloadURL",
    "group",
    "tags",
    "minRamGb",
    "devices",
    "description",
]

ALLOWED_QUANTIZATIONS = {"F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0"}
MIN_SIZE_GB = 0.05
MAX_SIZE_GB = 10.0
MIN_MODEL_COUNT = 12
MIN_RECOMMENDED_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate embedding_catalog.json")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_schema(catalog: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(catalog), key=lambda err: list(err.path))

    messages: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        messages.append(f"Schema violation at {path}: {error.message}")

    return messages


def validate_unique_ids(models: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for model in models:
        model_id = model.get("id")
        if isinstance(model_id, str):
            if model_id in seen:
                duplicates.add(model_id)
            seen.add(model_id)

    if not duplicates:
        return []

    return [f"Duplicate model id detected: {model_id}" for model_id in sorted(duplicates)]


def validate_required_fields(models: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    for index, model in enumerate(models):
        model_id = model.get("id", f"index-{index}")
        for field in REQUIRED_MODEL_FIELDS:
            if field not in model:
                errors.append(f"Model '{model_id}' missing required field '{field}'")
                continue

            value = model[field]
            if isinstance(value, str) and not value.strip():
                errors.append(f"Model '{model_id}' has empty field '{field}'")
            if isinstance(value, list) and len(value) == 0:
                errors.append(f"Model '{model_id}' has empty list field '{field}'")

    return errors


def build_headers(token: str | None, *, range_request: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if range_request:
        headers["Range"] = "bytes=0-0"
    return headers


def check_url_reachable(url: str, token: str | None) -> None:
    head_request = urllib.request.Request(url=url, headers=build_headers(token), method="HEAD")
    try:
        with urllib.request.urlopen(head_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", None) or response.getcode()
            if 200 <= status < 400:
                return
            raise RuntimeError(f"Unexpected HTTP status {status}")
    except urllib.error.HTTPError as exc:
        if exc.code not in {405, 501}:
            raise RuntimeError(f"HEAD failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HEAD failed: {exc.reason}") from exc

    get_request = urllib.request.Request(
        url=url,
        headers=build_headers(token, range_request=True),
        method="GET",
    )
    try:
        with urllib.request.urlopen(get_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", None) or response.getcode()
            if 200 <= status < 400:
                return
            raise RuntimeError(f"Unexpected HTTP status {status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Range GET failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Range GET failed: {exc.reason}") from exc


def validate_urls(models: list[dict[str, Any]], token: str | None) -> list[str]:
    errors: list[str] = []

    for model in models:
        model_id = model.get("id", "unknown")
        download_url = model.get("downloadURL")
        if not isinstance(download_url, str) or not download_url.strip():
            errors.append(f"Model '{model_id}' has invalid downloadURL")
            continue

        try:
            check_url_reachable(download_url, token)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Model '{model_id}' URL check failed: {download_url} ({exc})")

    return errors


def bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / 1_073_741_824


def expected_constraints(size_gb: float) -> dict[str, Any] | None:
    if MIN_SIZE_GB <= size_gb <= 0.8:
        return {
            "group": "recommended",
            "devices": {"iphone", "ipad", "mac"},
            "min_ram": 4,
        }
    if 0.8 < size_gb <= 2.5:
        return {
            "group": "advanced",
            "devices": {"ipad", "mac"},
            "min_ram": 6,
        }
    if 2.5 < size_gb <= MAX_SIZE_GB:
        return {
            "group": "advanced",
            "devices": {"mac"},
            "min_ram": 8,
        }
    return None


def validate_semantics(models: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    recommended_count = 0

    for model in models:
        model_id = str(model.get("id", "unknown"))

        size_bytes = model.get("sizeBytes")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            errors.append(f"Model '{model_id}' has invalid sizeBytes for semantic checks")
            continue

        size_gb = bytes_to_gb(size_bytes)
        expected = expected_constraints(size_gb)
        if expected is None:
            errors.append(
                f"Model '{model_id}' size {size_gb:.2f} GB is outside the allowed "
                f"{MIN_SIZE_GB:.2f}-{MAX_SIZE_GB:.1f} GB range"
            )
            continue

        quantization = model.get("quantization")
        if not isinstance(quantization, str) or quantization not in ALLOWED_QUANTIZATIONS:
            errors.append(
                f"Model '{model_id}' quantization '{quantization}' is not in allowed set "
                f"{sorted(ALLOWED_QUANTIZATIONS)}"
            )

        group = model.get("group")
        if group != expected["group"]:
            errors.append(
                f"Model '{model_id}' group '{group}' does not match expected "
                f"'{expected['group']}' for size {size_gb:.2f} GB"
            )
        if group == "recommended":
            recommended_count += 1

        min_ram = model.get("minRamGb")
        if not isinstance(min_ram, int) or min_ram != expected["min_ram"]:
            errors.append(
                f"Model '{model_id}' minRamGb '{min_ram}' does not match expected "
                f"'{expected['min_ram']}' for size {size_gb:.2f} GB"
            )

        devices = model.get("devices")
        if not isinstance(devices, list) or not all(isinstance(item, str) for item in devices):
            errors.append(f"Model '{model_id}' has invalid devices list")
            continue

        device_set = set(devices)
        if device_set != expected["devices"]:
            errors.append(
                f"Model '{model_id}' devices {sorted(device_set)} do not match expected "
                f"{sorted(expected['devices'])} for size {size_gb:.2f} GB"
            )

        tags = model.get("tags")
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            errors.append(f"Model '{model_id}' has invalid tags")
        elif "embedding" not in tags:
            errors.append(f"Model '{model_id}' tags must contain 'embedding'")

        sha256 = model.get("sha256")
        if sha256 is not None and (not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha256)):
            errors.append(f"Model '{model_id}' has invalid sha256 format")

    if len(models) < MIN_MODEL_COUNT:
        errors.append(
            f"Catalog has only {len(models)} models; minimum required is {MIN_MODEL_COUNT}"
        )
    if recommended_count < MIN_RECOMMENDED_COUNT:
        errors.append(
            f"Catalog has only {recommended_count} recommended models; minimum required is "
            f"{MIN_RECOMMENDED_COUNT}"
        )

    return errors


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    catalog_raw = load_json(args.catalog)
    schema_raw = load_json(args.schema)

    if not isinstance(catalog_raw, dict):
        raise RuntimeError("embedding_catalog.json root must be an object")
    if not isinstance(schema_raw, dict):
        raise RuntimeError("schema root must be an object")

    models = catalog_raw.get("models")
    if not isinstance(models, list):
        raise RuntimeError("embedding_catalog.json must include a models array")

    errors: list[str] = []
    errors.extend(validate_schema(catalog_raw, schema_raw))

    model_objects = [model for model in models if isinstance(model, dict)]
    errors.extend(validate_unique_ids(model_objects))
    errors.extend(validate_required_fields(model_objects))
    errors.extend(validate_semantics(model_objects))
    errors.extend(validate_urls(model_objects, token))

    if errors:
        for error in errors:
            print(f"[validate-embedding] {error}", file=sys.stderr)
        return 1

    print(
        f"[validate-embedding] success: {args.catalog} passed schema, required fields, "
        "id uniqueness, semantics and URL checks",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[validate-embedding] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
