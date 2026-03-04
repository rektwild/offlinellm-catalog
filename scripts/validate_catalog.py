#!/usr/bin/env python3
"""Validate catalog.json against schema and runtime expectations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT_DIR / "catalog.json"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "schema" / "catalog.schema.json"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "offlinellm-catalog-validator/1.0"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate catalog.json")
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


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    catalog_raw = load_json(args.catalog)
    schema_raw = load_json(args.schema)

    if not isinstance(catalog_raw, dict):
        raise RuntimeError("catalog.json root must be an object")
    if not isinstance(schema_raw, dict):
        raise RuntimeError("schema root must be an object")

    models = catalog_raw.get("models")
    if not isinstance(models, list):
        raise RuntimeError("catalog.json must include a models array")

    errors: list[str] = []
    errors.extend(validate_schema(catalog_raw, schema_raw))

    model_objects = [model for model in models if isinstance(model, dict)]
    errors.extend(validate_unique_ids(model_objects))
    errors.extend(validate_required_fields(model_objects))
    errors.extend(validate_urls(model_objects, token))

    if errors:
        for error in errors:
            print(f"[validate] {error}", file=sys.stderr)
        return 1

    print(
        f"[validate] success: {args.catalog} passed schema, required fields, id uniqueness and URL checks",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[validate] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
