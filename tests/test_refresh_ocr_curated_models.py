from __future__ import annotations

import datetime as dt
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import refresh_ocr_curated_models as rom  # noqa: E402


def build_policy() -> rom.DiscoveryPolicy:
    return rom.DiscoveryPolicy(
        endpoint="https://huggingface.co/api/models",
        search="llava gguf",
        sort="downloads",
        direction="-1",
        limit=240,
        full=True,
        target_count=6,
        minimum_count=3,
        fallback_context_length=4096,
        likes_weight=120,
        bucket_targets={"small": 2, "medium": 2, "large": 2},
        trusted_authors={"xtuner"},
        min_downloads=150,
        min_likes=0,
        recency_from=dt.date(2023, 1, 1),
        blocklist_repo_substrings=[],
        quantization_priority=["F16", "Q8_0", "Q4_K_M"],
        size_min_gb=0.3,
        size_max_gb=24.0,
        mmproj_path_contains=["mmproj"],
        reject_path_regex=[],
        segmentation={
            "small": {
                "maxSizeGb": 5.0,
                "devices": ["ipad", "mac"],
                "minRamGb": 8,
                "group": "advanced",
            },
            "medium": {
                "maxSizeGb": 12.0,
                "devices": ["mac"],
                "minRamGb": 12,
                "group": "advanced",
            },
            "large": {
                "maxSizeGb": 24.0,
                "devices": ["mac"],
                "minRamGb": 16,
                "group": "advanced",
            },
        },
        prompt_template="Extract all readable text from this image.",
    )


def make_candidate(
    repo_id: str,
    score: int,
    *,
    key: str = "xtuner/llava-llama-3-8b-v1_1",
    bucket: str = "small",
) -> rom.OCRCandidate:
    return rom.OCRCandidate(
        repo_id=repo_id,
        author=repo_id.split("/", 1)[0],
        downloads=100_000,
        likes=100,
        last_modified="2026-01-01T00:00:00.000Z",
        base_model="xtuner/Llava-Llama-3-8B",
        family="llava",
        context_length=4096,
        main_path="llava-q4_k_m.gguf",
        main_size_bytes=int(4.0 * 1_073_741_824),
        main_sha256="a" * 64,
        mmproj_path="mmproj-f16.gguf",
        mmproj_size_bytes=int(0.5 * 1_073_741_824),
        mmproj_sha256="b" * 64,
        quantization="Q4_K_M",
        candidate_key=key,
        score=score,
        bucket=bucket,
    )


class RefreshOCRCuratedModelsTests(unittest.TestCase):
    def test_select_ocr_pair_requires_mmproj(self) -> None:
        nodes = [
            {"path": "model-Q4_K_M.gguf", "size": 1, "type": "file"},
        ]

        selected = rom.select_ocr_pair(
            tree_nodes=nodes,
            mmproj_tokens=["mmproj"],
            quant_priority=["Q4_K_M"],
            reject_regex=[],
        )

        self.assertIsNone(selected)

    def test_select_ocr_pair_prefers_quant_and_f16_mmproj(self) -> None:
        nodes = [
            {"path": "model-Q8_0.gguf", "size": 1, "type": "file"},
            {"path": "model-f16.gguf", "size": 1, "type": "file"},
            {"path": "mmproj-Q8_0.gguf", "size": 1, "type": "file"},
            {"path": "mmproj-f16.gguf", "size": 1, "type": "file"},
        ]

        selected = rom.select_ocr_pair(
            tree_nodes=nodes,
            mmproj_tokens=["mmproj"],
            quant_priority=["F16", "Q8_0"],
            reject_regex=[],
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        main, mmproj, quant = selected
        self.assertEqual(main["path"], "model-f16.gguf")
        self.assertEqual(mmproj["path"], "mmproj-f16.gguf")
        self.assertEqual(quant, "F16")

    def test_generate_model_id_has_ocr_suffix(self) -> None:
        model_id = rom.generate_model_id("xtuner/Llava-Llama-3-8B", "Q4_K_M")
        self.assertEqual(model_id, "xtuner-llava-llama-3-8b-q4-k-m-ocr")

    def test_probe_download_url_rejects_http_403(self) -> None:
        with mock.patch(
            "refresh_ocr_curated_models.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://huggingface.co/model/file.gguf",
                code=403,
                msg="Forbidden",
                hdrs=None,
                fp=None,
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                rom.probe_download_url("https://huggingface.co/model/file.gguf")
            self.assertIn("HTTP 403", str(ctx.exception))

    def test_dedupe_prefers_existing_repo_for_churn_reduction(self) -> None:
        first = make_candidate("xtuner/llava-llama-3-8b-v1_1-gguf", score=150_000)
        second = make_candidate("second-state/llava-llama-3-8b-v1_1-gguf", score=160_000)

        deduped = rom.dedupe_candidates([first, second], existing_repos={first.repo_id.lower()})

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].repo_id, first.repo_id)
        self.assertTrue(deduped[0].retained)

    def test_bucket_constraints_match_policy(self) -> None:
        policy = build_policy()

        group_small, devices_small, min_ram_small = rom.map_bucket_to_constraints("small", policy)
        self.assertEqual(group_small, "advanced")
        self.assertEqual(devices_small, ["ipad", "mac"])
        self.assertEqual(min_ram_small, 8)

        group_medium, devices_medium, min_ram_medium = rom.map_bucket_to_constraints("medium", policy)
        self.assertEqual(group_medium, "advanced")
        self.assertEqual(devices_medium, ["mac"])
        self.assertEqual(min_ram_medium, 12)

        group_large, devices_large, min_ram_large = rom.map_bucket_to_constraints("large", policy)
        self.assertEqual(group_large, "advanced")
        self.assertEqual(devices_large, ["mac"])
        self.assertEqual(min_ram_large, 16)


if __name__ == "__main__":
    unittest.main()
