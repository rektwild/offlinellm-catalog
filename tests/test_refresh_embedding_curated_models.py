from __future__ import annotations

import datetime as dt
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import refresh_embedding_curated_models as rem  # noqa: E402


def build_policy() -> rem.DiscoveryPolicy:
    return rem.DiscoveryPolicy(
        endpoint="https://huggingface.co/api/models",
        search_terms=["embedding gguf"],
        sort="downloads",
        direction="-1",
        limit=240,
        full=True,
        target_count=20,
        minimum_count=12,
        fallback_context_length=8192,
        likes_weight=150,
        bucket_targets={"small": 8, "medium": 7, "large": 5},
        trusted_authors={"qwen"},
        min_downloads=500,
        min_likes=0,
        recency_from=dt.date(2024, 1, 1),
        blocklist_repo_substrings=[],
        quantization_priority=["F16", "Q8_0", "Q4_K_M"],
        size_min_gb=0.05,
        size_max_gb=10.0,
        reject_path_contains=["mmproj"],
        reject_path_regex=[],
        segmentation={
            "small": {
                "maxSizeGb": 0.8,
                "devices": ["iphone", "ipad", "mac"],
                "minRamGb": 4,
                "group": "recommended",
            },
            "medium": {
                "maxSizeGb": 2.5,
                "devices": ["ipad", "mac"],
                "minRamGb": 6,
                "group": "advanced",
            },
            "large": {
                "maxSizeGb": 10.0,
                "devices": ["mac"],
                "minRamGb": 8,
                "group": "advanced",
            },
        },
    )


def make_candidate(
    repo_id: str,
    score: int,
    *,
    key: str = "qwen/qwen3-embedding-0.6b",
    bucket: str = "small",
) -> rem.CandidateModel:
    return rem.CandidateModel(
        repo_id=repo_id,
        author=repo_id.split("/", 1)[0],
        downloads=100_000,
        likes=100,
        last_modified="2026-01-01T00:00:00.000Z",
        base_model="Qwen/Qwen3-Embedding-0.6B",
        family="qwen",
        context_length=32768,
        file_name="Qwen3-Embedding-0.6B-f16.gguf",
        quantization="F16",
        size_bytes=int(0.6 * 1_073_741_824),
        sha256=None,
        candidate_key=key,
        score=score,
        bucket=bucket,
    )


class RefreshEmbeddingCuratedModelsTests(unittest.TestCase):
    def test_select_embedding_gguf_prefers_f16(self) -> None:
        nodes = [
            {"path": "model-Q4_K_M.gguf", "size": 1, "type": "file"},
            {"path": "model-Q8_0.gguf", "size": 1, "type": "file"},
            {"path": "model-f16.gguf", "size": 1, "type": "file"},
        ]

        selected = rem.select_embedding_gguf(
            tree_nodes=nodes,
            quant_priority=["F16", "Q8_0", "Q4_K_M"],
            reject_contains=[],
            reject_regex=[],
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        node, quant = selected
        self.assertEqual(node["path"], "model-f16.gguf")
        self.assertEqual(quant, "F16")

    def test_generate_model_id_is_stable(self) -> None:
        model_id = rem.generate_model_id("Qwen/Qwen3-Embedding-0.6B", "F16")
        self.assertEqual(model_id, "qwen-qwen3-embedding-0-6b-f16")

    def test_build_download_url_preserves_case(self) -> None:
        url = rem.build_download_url("ggml-org/bge-m3-Q8_0-GGUF", "bge-m3-q8_0.gguf")
        self.assertIn("bge-m3-q8_0.gguf", url)
        self.assertNotIn("bge-m3-Q8_0.gguf", url)

    def test_probe_download_url_rejects_http_401(self) -> None:
        with mock.patch(
            "refresh_embedding_curated_models.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://huggingface.co/model/file.gguf",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                rem.probe_download_url("https://huggingface.co/model/file.gguf")
            self.assertIn("HTTP 401", str(ctx.exception))

    def test_dedupe_prefers_existing_repo_for_churn_reduction(self) -> None:
        first = make_candidate("Qwen/Qwen3-Embedding-0.6B-GGUF", score=150_000)
        second = make_candidate("second-state/Qwen3-Embedding-0.6B-GGUF", score=160_000)

        deduped = rem.dedupe_candidates([first, second], existing_repos={first.repo_id.lower()})

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].repo_id, first.repo_id)
        self.assertTrue(deduped[0].retained)

    def test_bucket_constraints_match_policy(self) -> None:
        policy = build_policy()

        group_small, devices_small, min_ram_small = rem.map_bucket_to_constraints("small", policy)
        self.assertEqual(group_small, "recommended")
        self.assertEqual(devices_small, ["iphone", "ipad", "mac"])
        self.assertEqual(min_ram_small, 4)

        group_medium, devices_medium, min_ram_medium = rem.map_bucket_to_constraints("medium", policy)
        self.assertEqual(group_medium, "advanced")
        self.assertEqual(devices_medium, ["ipad", "mac"])
        self.assertEqual(min_ram_medium, 6)

        group_large, devices_large, min_ram_large = rem.map_bucket_to_constraints("large", policy)
        self.assertEqual(group_large, "advanced")
        self.assertEqual(devices_large, ["mac"])
        self.assertEqual(min_ram_large, 8)


if __name__ == "__main__":
    unittest.main()
