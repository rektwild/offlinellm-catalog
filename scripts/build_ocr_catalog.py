#!/usr/bin/env python3
"""Build ocr_catalog.json from ocr_curated_models.yaml."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CURATED_PATH = ROOT_DIR / "ocr_curated_models.yaml"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "ocr_catalog.json"
REQUIRED_FIELDS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ocr_catalog.json from curated file")
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_curated(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Curated file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise RuntimeError("ocr_curated_models.yaml must contain top-level models list")

    models: list[dict[str, Any]] = []
    for entry in payload["models"]:
        if not isinstance(entry, dict):
            raise RuntimeError("Each model entry must be an object")
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            model_id = entry.get("id", "unknown")
            raise RuntimeError(f"Model '{model_id}' missing fields: {sorted(missing)}")
        models.append(entry)

    return sorted(models, key=lambda item: str(item.get("id", "")))


def load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("Existing ocr catalog must be an object")
    return data


def utc_today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def compute_version(existing: dict[str, Any] | None, new_models: list[dict[str, Any]]) -> int:
    if existing is None:
        return 1

    version = existing.get("version")
    if not isinstance(version, int) or version < 1:
        raise RuntimeError("Existing ocr catalog version must be a positive integer")

    previous = existing.get("models")
    previous_models = sorted(previous, key=lambda item: str(item.get("id", ""))) if isinstance(previous, list) else []
    if previous_models == new_models:
        return version
    return version + 1


def main() -> int:
    args = parse_args()
    models = load_curated(args.curated)
    existing = load_existing(args.output)
    version = compute_version(existing, models)

    catalog = {
        "version": version,
        "updatedAt": utc_today(),
        "models": models,
    }

    args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[build-ocr] wrote {args.output} version={version} models={len(models)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[build-ocr] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
