from __future__ import annotations

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import validate_ocr_catalog as voc  # noqa: E402


class ValidateOCRCatalogTests(unittest.TestCase):
    def test_check_url_rejects_http_401(self) -> None:
        with mock.patch(
            "validate_ocr_catalog.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://huggingface.co/model/file.gguf",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                voc.check_url("https://huggingface.co/model/file.gguf")
            self.assertIn("HTTP 401", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
