"""Regression tests for the full-article backup corpus."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_unchunked_corpus import build
from scripts import build_full_corpus as full_corpus


class UnchunkedCorpusTests(unittest.TestCase):
    def test_extended_full_corpus_topic_rules(self) -> None:
        self.assertEqual(full_corpus.classify_full({
            "law_id": "61-CP", "law_name": "Nghị định về mua bán và kinh doanh nhà ở",
            "fields": ["Chưa phân loại"],
        }, {}), (full_corpus.TOPICS[6], "document_title_extended"))
        self.assertEqual(full_corpus.classify_full({
            "law_id": "60/2015/TT-BTNMT", "law_name": "Thông tư về điều tra đất đai",
            "fields": ["Chưa phân loại"],
        }, {}), (full_corpus.TOPICS[7], "document_title_extended"))
        self.assertEqual(full_corpus.classify_full({
            "law_id": "1/2026/TT-BTC", "law_name": "Thông tư về quản lý ngân sách",
            "fields": ["Quản lý ngân sách"],
        }, {}), (None, None))
        self.assertEqual(full_corpus.classify_full({
            "law_id": "2/2026/TT-BYT", "law_name": "Quản lý Nhà nước về an toàn thực phẩm",
            "fields": ["Chưa phân loại"],
        }, {}), (None, None))
        self.assertEqual(full_corpus.classify_full({
            "law_id": "3/2026/QĐ-TTg", "law_name": "Tổng điều tra dân số và nhà ở",
            "fields": ["Chưa phân loại"],
        }, {}), (full_corpus.TOPICS[1], "document_title_extended"))

    def test_full_corpus_keeps_distinct_versions_and_validity(self) -> None:
        legacy = {
            "law_id": "L/2024", "law_name": "Sample law", "article": "Điều 1",
            "article_number": "1", "content": "Old text", "content_chars": 8,
            "topic": full_corpus.TOPICS[2], "quality_status": "ready",
            "source_urls": ["https://legacy/1"],
        }
        processed = [
            {
                "law_id": "L/2024", "law_name": "Sample law", "article": "Điều 1",
                "article_number": "1", "content": content, "citation": "Sample law",
                "source_urls": [f"https://vbpl.vn/{index}"], "document_id": f"doc-{index}",
                "record_id": f"vbpl:{index}", "unit_type": "article",
                "validity_status": status, "validity_code": "CHL",
                "issue_date": issued,
            }
            for index, content, status, issued in [
                (1, "Old text", "Expired", "2024-01-01"),
                (2, "New text", "Current", "2025-01-01"),
            ]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path = root / "legacy.jsonl"
            processed_path = root / "processed.jsonl"
            legacy_path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")
            processed_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in processed) + "\n", encoding="utf-8")
            with (
                patch.object(full_corpus, "CANDIDATES_PATH", root / "candidates.json"),
                patch.object(full_corpus, "RAW_DIR", root / "raw"),
                patch.object(full_corpus, "LABEL_PATH", root / "labels.json"),
            ):
                summary = full_corpus.build(
                    legacy_path, processed_path, root / "full.json", root / "manifest.json", root / "report.md"
                )
            rows = json.loads((root / "full.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["output_records"], 1)
        self.assertEqual(summary["keys_with_alternatives"], 1)
        self.assertEqual(rows[0]["content"], "New text")
        self.assertEqual(rows[0]["validity_status"], "Current")
        self.assertEqual(rows[0]["variants"][0]["content"], "Old text")
        self.assertEqual(len(rows[0]["variants"][0]["equivalent_sources"]), 1)

    def test_reassembles_chunks_and_keeps_content_variants_under_one_key(self) -> None:
        rows = [
            {
                "law_id": "L/2024", "article": "Điều 1", "article_number": "1",
                "content": "VBPL text", "unit_type": "article", "source_system": "vbpl_gateway",
                "quality_status": "ready", "record_id": "vbpl:1", "issue_date": "2024-01-01",
                "source_urls": ["https://vbpl.vn/1"],
            },
            *[
                {
                    "law_id": "L/2024", "article": "Điều 1", "article_number": "1",
                    "content": text, "unit_type": "article_chunk", "source_system": "phapdien",
                    "quality_status": "ready", "parent_record_id": "legacy:1",
                    "chunk_number": number, "record_id": f"legacy:1:chunk:{number}",
                    "source_urls": ["https://legacy/1"],
                }
                for number, text in [(2, "Legacy second"), (1, "Legacy first")]
            ],
            *[
                {
                    "law_id": "L/2024", "article": None, "article_number": None,
                    "content": text, "unit_type": "document_chunk", "source_system": "vbpl_gateway",
                    "quality_status": "ready", "document_id": "doc-1",
                    "chunk_number": number, "record_id": f"doc-1:chunk:{number}",
                    "issue_date": "2024-01-01", "source_urls": ["https://vbpl.vn/doc-1"],
                }
                for number, text in [(2, "Document end"), (1, "Document start")]
            ],
            {
                "law_id": "L/2024", "article": None, "article_number": None,
                "content": "Other document", "unit_type": "document_chunk",
                "source_system": "vbpl_gateway", "quality_status": "ready",
                "document_id": "doc-2", "chunk_number": 1, "record_id": "doc-2:chunk:1",
                "issue_date": "2025-01-01", "source_urls": ["https://vbpl.vn/doc-2"],
            },
            {
                "law_id": "L/2024", "article": "Điều 2", "article_number": "2",
                "content": "Separate article", "unit_type": "article", "source_system": "phapdien",
                "quality_status": "ready", "source_urls": ["https://legacy/2"],
            },
            {
                "law_id": "", "article": "Điều 3", "content": "Unusable",
                "unit_type": "article", "quality_status": "quarantine",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "active.json"
            output = root / "backup.json"
            manifest = root / "manifest.json"
            source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            summary = build(source, output, manifest)
            backup = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["output_record_count"], 3)
        self.assertEqual(summary["keys_with_alternatives"], 2)
        self.assertEqual(summary["skipped_non_ready_records"], 1)
        self.assertEqual(len({(row["law_id"], row["article"]) for row in backup}), 3)
        self.assertTrue(all(row["unit_type"] not in {"article_chunk", "document_chunk"} for row in backup))
        article = next(row for row in backup if row["article"] == "Điều 1")
        self.assertEqual(article["content"], "VBPL text")
        self.assertEqual(article["variants"][0]["content"], "Legacy first\nLegacy second")
        self.assertEqual(article["variants"][0]["source_record_ids"], [
            "legacy:1:chunk:1", "legacy:1:chunk:2",
        ])
        document = next(row for row in backup if row["article"] is None)
        self.assertEqual(document["content"], "Other document")
        self.assertEqual(document["variants"][0]["content"], "Document start\nDocument end")

    def test_missing_chunk_is_rejected(self) -> None:
        rows = [
            {
                "law_id": "L/2024", "article": "Điều 1", "article_number": "1",
                "content": "Only second part", "unit_type": "article_chunk",
                "quality_status": "ready", "source_system": "vbpl_gateway",
                "parent_record_id": "vbpl:1", "chunk_number": 2,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "active.json"
            source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                build(source, root / "backup.json", root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
