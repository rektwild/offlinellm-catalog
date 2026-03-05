#!/usr/bin/env python3
"""Validate ocr_catalog.json against schema and runtime checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT_DIR / "ocr_catalog.json"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "schema" / "ocr_catalog.schema.json"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "offlinellm-ocr-catalog-validator/1.0"

REQUIRED_FIELDS = [
    "id",
    "name",
    "family",
    "quantization",
    "sizeBytes",
    "contextLength",
    "downloadURL",
    "sha256",
    "mmprojDownloadURL",
    "mmprojSizeBytes",
    "mmprojSha256",
    "group",
    "tags",
    "minRamGb",
    "devices",
    "description",
    "promptTemplate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ocr_catalog.json")
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
    messages: list[str] = []
    for error in sorted(validator.iter_errors(catalog), key=lambda item: list(item.path)):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        messages.append(f"Schema violation at {path}: {error.message}")
    return messages


def validate_required(models: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, model in enumerate(models):
        model_id = model.get("id", f"index-{index}")
        for field in REQUIRED_FIELDS:
            if field not in model:
                errors.append(f"Model '{model_id}' missing required field '{field}'")
    return errors


def validate_unique_ids(models: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for model in models:
        model_id = model.get("id")
        if isinstance(model_id, str):
            if model_id in seen:
                duplicates.add(model_id)
            seen.add(model_id)
    return [f"Duplicate model id: {item}" for item in sorted(duplicates)]


def check_url(url: str) -> None:
    request = urllib.request.Request(
        url=url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Range": "bytes=0-0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", None) or response.getcode()
            if not 200 <= status < 400:
                raise RuntimeError(f"Unexpected HTTP status {status}")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            # Public repos may deny anonymous range probes. Allow and rely on runtime download checks.
            return
        raise


def validate_urls(models: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for model in models:
        model_id = str(model.get("id", "unknown"))
        main_url = model.get("downloadURL")
        mmproj_url = model.get("mmprojDownloadURL")
        for label, value in (("downloadURL", main_url), ("mmprojDownloadURL", mmproj_url)):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Model '{model_id}' invalid {label}")
                continue
            try:
                check_url(value)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Model '{model_id}' {label} check failed: {value} ({exc})")
    return errors


def validate_hashes(models: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for model in models:
        model_id = str(model.get("id", "unknown"))
        for key in ("sha256", "mmprojSha256"):
            value = model.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                errors.append(f"Model '{model_id}' has invalid {key}")
    return errors


def main() -> int:
    args = parse_args()

    catalog = load_json(args.catalog)
    schema = load_json(args.schema)

    if not isinstance(catalog, dict):
        raise RuntimeError("ocr_catalog.json root must be an object")
    if not isinstance(schema, dict):
        raise RuntimeError("schema root must be an object")

    models_raw = catalog.get("models")
    if not isinstance(models_raw, list):
        raise RuntimeError("ocr_catalog.json must include models array")

    model_objects = [item for item in models_raw if isinstance(item, dict)]

    errors: list[str] = []
    errors.extend(validate_schema(catalog, schema))
    errors.extend(validate_required(model_objects))
    errors.extend(validate_unique_ids(model_objects))
    errors.extend(validate_hashes(model_objects))
    errors.extend(validate_urls(model_objects))

    if errors:
        for item in errors:
            print(f"[validate-ocr] {item}", file=sys.stderr)
        return 1

    print(f"[validate-ocr] success: {args.catalog} passed all checks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[validate-ocr] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
