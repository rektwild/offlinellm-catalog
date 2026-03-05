from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import refresh_curated_models as rcm  # noqa: E402


def build_policy() -> rcm.DiscoveryPolicy:
    return rcm.DiscoveryPolicy(
        endpoint="https://huggingface.co/api/models",
        search="gguf",
        model_filter="text-generation",
        sort="downloads",
        direction="-1",
        limit=300,
        full=True,
        target_count=24,
        minimum_count=20,
        minimum_recommended_count=8,
        fallback_context_length=8192,
        likes_weight=250,
        bucket_targets={"small": 8, "medium": 8, "large": 8},
        trusted_authors={"bartowski"},
        min_downloads=10000,
        min_likes=25,
        recency_from=dt.date(2024, 1, 1),
        blocklist_repo_substrings=[],
        base_model_owner_allowlist={"meta-llama"},
        quantization_priority=["Q4_K_M", "Q4_K_S", "Q4_0"],
        size_min_gb=0.7,
        size_max_gb=32.0,
        reject_path_contains=["mmproj"],
        reject_path_regex=[],
        segmentation={
            "small": {
                "maxSizeGb": 3.5,
                "devices": ["iphone", "ipad", "mac"],
                "minRamGb": 4,
                "group": "recommended",
            },
            "medium": {
                "maxSizeGb": 8.0,
                "devices": ["ipad", "mac"],
                "lowRamThresholdGb": 4.5,
                "minRamGbLow": 6,
                "minRamGbHigh": 8,
                "group": "advanced",
            },
            "large": {
                "maxSizeGb": 32.0,
                "devices": ["mac"],
                "highRamThresholdGb": 16.0,
                "minRamGbMid": 12,
                "minRamGbHigh": 16,
                "group": "advanced",
            },
        },
    )


def make_candidate(
    repo_id: str,
    score: int,
    *,
    key: str = "meta-llama/llama-3.2-3b-instruct",
    bucket: str = "small",
) -> rcm.CandidateModel:
    return rcm.CandidateModel(
        repo_id=repo_id,
        author=repo_id.split("/", 1)[0],
        downloads=100_000,
        likes=100,
        last_modified="2026-01-01T00:00:00.000Z",
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        family="llama3",
        context_length=131072,
        file_name="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_bytes=int(2.0 * 1_073_741_824),
        sha256=None,
        candidate_key=key,
        score=score,
        bucket=bucket,
    )


class RefreshCuratedModelsTests(unittest.TestCase):
    def test_select_quantized_gguf_prefers_q4_k_m(self) -> None:
        nodes: list[dict[str, Any]] = [
            {"path": "model-Q4_0.gguf", "size": 1, "type": "file"},
            {"path": "model-Q4_K_S.gguf", "size": 1, "type": "file"},
            {"path": "model-Q4_K_M.gguf", "size": 1, "type": "file"},
        ]

        selected = rcm.select_quantized_gguf(
            tree_nodes=nodes,
            quant_priority=["Q4_K_M", "Q4_K_S", "Q4_0"],
            reject_contains=[],
            reject_regex=[],
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        node, quant = selected
        self.assertEqual(node["path"], "model-Q4_K_M.gguf")
        self.assertEqual(quant, "Q4_K_M")

    def test_generate_model_id_is_stable(self) -> None:
        model_id = rcm.generate_model_id("meta-llama/Llama-3.2-3B-Instruct", "Q4_K_M")
        self.assertEqual(model_id, "meta-llama-llama-3-2-3b-instruct-q4-k-m")

    def test_dedupe_prefers_existing_repo_for_churn_reduction(self) -> None:
        first = make_candidate("bartowski/Llama-3.2-3B-Instruct-GGUF", score=150_000)
        second = make_candidate("lmstudio-community/Llama-3.2-3B-Instruct-GGUF", score=160_000)

        deduped = rcm.dedupe_candidates([first, second], existing_repos={first.repo_id.lower()})

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].repo_id, first.repo_id)
        self.assertTrue(deduped[0].retained)

    def test_bucket_constraints_match_policy(self) -> None:
        policy = build_policy()

        group_small, devices_small, min_ram_small = rcm.map_bucket_to_constraints("small", 2.2, policy)
        self.assertEqual(group_small, "recommended")
        self.assertEqual(devices_small, ["iphone", "ipad", "mac"])
        self.assertEqual(min_ram_small, 4)

        group_medium_low, _, min_ram_medium_low = rcm.map_bucket_to_constraints("medium", 4.2, policy)
        group_medium_high, _, min_ram_medium_high = rcm.map_bucket_to_constraints("medium", 7.8, policy)
        self.assertEqual(group_medium_low, "advanced")
        self.assertEqual(min_ram_medium_low, 6)
        self.assertEqual(min_ram_medium_high, 8)

        _, devices_large, min_ram_large_mid = rcm.map_bucket_to_constraints("large", 12.0, policy)
        _, _, min_ram_large_high = rcm.map_bucket_to_constraints("large", 20.0, policy)
        self.assertEqual(devices_large, ["mac"])
        self.assertEqual(min_ram_large_mid, 12)
        self.assertEqual(min_ram_large_high, 16)


if __name__ == "__main__":
    unittest.main()
