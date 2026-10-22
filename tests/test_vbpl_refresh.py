"""Regression tests for VBPL discovery and article extraction."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from bs4 import BeautifulSoup

from notebooks import vbpl_gateway_crawler_v2 as crawler
from scripts import preprocess_vbpl_corpus as preprocessor
from scripts import refresh_vbpl_corpus as refresher
from scripts import merge_vbpl_into_active_corpus as merger
from scripts.merge_vbpl_into_active_corpus import classify
from scripts.preprocess_vbpl_corpus import chunk_paragraphs, extract_articles, normalize_text
from scripts.split_vbpl_records import split_records


def sitemap(*urls: str) -> str:
    locations = "".join(f"<url><loc>{url}</loc></url>" for url in urls)
    return f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{locations}</urlset>'


class Response:
    def __init__(self, text: str = "", agency: str | None = None) -> None:
        self.text = text
        self.agency = agency

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"success": True, "data": {"agencyName": self.agency}}


class Session:
    def __init__(self, responses: dict[str, Response]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url: str, **kwargs) -> Response:
        self.requested.append(url)
        return self.responses[url]


class VBPLRefreshTests(unittest.TestCase):
    def test_extracts_gateway_id_from_both_sitemap_url_forms(self) -> None:
        self.assertEqual(
            crawler.extract_item_id("https://vbpl.vn/van-ban/chi-tiet/--173680"),
            "173680",
        )
        self.assertEqual(
            crawler.extract_item_id("https://vbpl.vn/van-ban/chi-tiet/173680"),
            "173680",
        )
        self.assertEqual(
            crawler.extract_item_id("https://vbpl.vn/van-ban/chi-tiet/vbpqdinhchinh_74"),
            "vbpqdinhchinh_74",
        )

    def test_final_sitemap_refresh_crawls_new_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "one.json").write_text("{}", encoding="utf-8")
            candidates_path = root / "candidates.json"

            def write_candidates(*ids: str) -> None:
                candidates_path.write_text(json.dumps({
                    "candidates": [{"url": f"https://vbpl.vn/van-ban/chi-tiet/--{item_id}"} for item_id in ids]
                }), encoding="utf-8")

            write_candidates("one")
            checkpoint_path = root / "checkpoint.json"
            checkpoint_path.write_text(json.dumps({"failed_ids": {}}), encoding="utf-8")

            def run_stage(command: list[str], **kwargs):
                if "--refresh-candidates" in command:
                    write_candidates("one", "two")
                elif "--max-docs" in command:
                    (raw_dir / "two.json").write_text("{}", encoding="utf-8")
                return type("Result", (), {"returncode": 0})()

            with (
                patch.object(refresher, "ROOT", root),
                patch.object(refresher, "STATUS_PATH", root / "status.json"),
                patch.object(refresher, "CHECKPOINT_PATH", checkpoint_path),
                patch.object(refresher, "CANDIDATES_PATH", candidates_path),
                patch.object(refresher, "RAW_DIR", raw_dir),
                patch.object(refresher, "POST_CRAWL_STAGES", ()),
                patch.object(refresher.subprocess, "run", side_effect=run_stage),
            ):
                refresher.main()
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["coverage"]["candidate_urls"], 2)
            self.assertEqual(status["coverage"]["raw_documents"], 2)

    def test_preprocess_also_writes_full_articles_and_div_fallback(self) -> None:
        metadata = {
            "so_hieu": "L/2024", "ten_van_ban": "Sample law", "loai_van_ban": "Law",
            "co_quan_ban_hanh": "Issuer", "ngay_ban_hanh": "2024-01-01",
            "trang_thai_hieu_luc": "Current", "trang_thai_code": "CHL",
            "url_chi_tiet": "https://vbpl.vn/van-ban/chi-tiet/sample",
        }
        records = [
            {"doc_id": "one", "thuoc_tinh": metadata, "_extra": {}, "noi_dung": {
                "content_html": "<p class='prov-article'>Điều 1. Title</p><p>" + "Long sentence. " * 500 + "</p>",
            }},
            {"doc_id": "two", "thuoc_tinh": metadata, "_extra": {}, "noi_dung": {
                "content_html": "<div>Fallback document text.</div>",
            }},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ready.jsonl"
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
            with patch.object(preprocessor, "ROOT", root):
                preprocessor.preprocess(source, root / "processed", root / "report.md")
                full_summary = preprocessor.preprocess_full_only(
                    source, root / "full_only", root / "full_report.md"
                )
            with (root / "processed" / "legal_corpus_full.jsonl").open(encoding="utf-8") as source:
                full = [json.loads(line) for line in source]
            with (root / "processed" / "legal_corpus.jsonl").open(encoding="utf-8") as source:
                chunked = [json.loads(line) for line in source]
            self.assertFalse((root / "full_only" / "legal_corpus.jsonl").exists())
        self.assertEqual([row["unit_type"] for row in full], ["article", "document"])
        self.assertGreater(len(full[0]["content"]), 6_000)
        self.assertEqual(full[1]["content"], "Fallback document text.")
        self.assertTrue(any(row["unit_type"] == "article_chunk" for row in chunked))
        self.assertTrue(any(row["unit_type"] == "document_chunk" for row in chunked))
        self.assertEqual(full_summary["records"], 2)

    def test_duplicate_chunked_article_is_removed_as_a_whole(self) -> None:
        rows = [
            {
                "law_id": "L/2024", "article_number": "1", "source_system": source,
                "unit_type": "article_chunk", "parent_record_id": parent,
                "record_id": f"{parent}:chunk:{number}", "chunk_number": number,
                "content": content, "source_urls": [url],
            }
            for source, parent, url, parts in [
                ("phapdien", "old", "https://legacy/1", ["First.", "Second."]),
                ("vbpl_gateway", "new", "https://vbpl.vn/1", ["First. ", "Second."]),
            ]
            for number, content in enumerate(parts, start=1)
        ]
        kept, articles_removed, chunks_removed = merger.remove_duplicate_legacy_chunk_articles(rows)
        self.assertEqual((articles_removed, chunks_removed), (1, 2))
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(row["source_system"] == "vbpl_gateway" for row in kept))
        self.assertTrue(all(row["source_urls"] == ["https://legacy/1", "https://vbpl.vn/1"] for row in kept))

    def test_long_legacy_article_gets_ordered_balanced_chunks(self) -> None:
        content = ("Opening " * 650) + ("Closing " * 150)
        row = {"content": content, "unit_type": "article", "source_system": "phapdien"}
        chunks = merger.chunk_legacy_article(row, "source-node-1")
        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk["chunk_number"] for chunk in chunks], [1, 2])
        self.assertEqual(len({chunk["parent_record_id"] for chunk in chunks}), 1)
        self.assertTrue(all(2_000 <= len(chunk["content"]) <= 6_000 for chunk in chunks))
        self.assertEqual(normalize_text("\n".join(chunk["content"] for chunk in chunks)), normalize_text(content))

    def test_merge_keeps_distinct_article_content_and_prefers_vbpl_for_same_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topic = merger.TOPICS[2]

            def legacy(content: str) -> dict:
                return {
                    "law_id": "L/2024", "law_name": "Sample law", "article": "Article 1",
                    "article_number": "1", "content": content,
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "topic": topic, "quality_status": "ready", "source_urls": ["https://legacy/1"],
                }

            def processed(record_id: str, content: str) -> dict:
                return {
                    "law_id": "L/2024", "law_name": "Sample law", "article": "Article 1",
                    "article_number": "1", "content": content,
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "citation": "Sample law", "source_urls": ["https://vbpl.vn/sample/1"],
                    "document_id": "one", "record_id": record_id, "unit_type": "article",
                    "validity_status": "Current", "validity_code": "CHL", "issue_date": "2024-01-01",
                }

            canonical = root / "canonical.jsonl"
            canonical.write_text("\n".join(json.dumps(row) for row in [
                legacy("Clause one. Clause two."),
                legacy("Clause one. Clause two."),
                legacy("Different rule."),
            ]) + "\n", encoding="utf-8")
            processed_path = root / "processed.jsonl"
            processed_path.write_text("\n".join(json.dumps(row) for row in [
                processed("vbpl:1", "Clause one.\nClause two."),
                processed("vbpl:2", "Another rule."),
                processed("vbpl:3", "Clause one.  Clause two."),
            ]) + "\n", encoding="utf-8")
            current_path = root / "current.json"
            current_path.write_text("[]", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"source": "canonical.jsonl"}), encoding="utf-8")

            with (
                patch.object(merger, "ROOT", root),
                patch.object(merger, "CURRENT_PATH", current_path),
                patch.object(merger, "PROCESSED_PATH", processed_path),
                patch.object(merger, "MANIFEST_PATH", manifest_path),
                patch.object(merger, "REPORT_PATH", root / "report.md"),
                patch.object(merger, "CANDIDATES_PATH", root / "candidates.json"),
                patch.object(merger, "RAW_DIR", root / "raw"),
                patch.object(merger, "CHECKPOINT_PATH", root / "checkpoint.json"),
            ):
                result = merger.merge()

            rows = json.loads(current_path.read_text(encoding="utf-8"))
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["duplicate_legacy_content"], 1)
            self.assertEqual(result["replaced_legacy_by_vbpl"], 1)
            self.assertEqual(result["duplicate_normalized_processed"], 1)
            preferred = next(row for row in rows if row.get("record_id") == "vbpl:1")
            self.assertEqual(preferred["source_urls"], ["https://legacy/1", "https://vbpl.vn/sample/1"])
            self.assertEqual({" ".join(row["content"].split()) for row in rows}, {
                "Clause one. Clause two.", "Different rule.", "Another rule."
            })

    def test_article_chunks_preserve_order_after_long_paragraph(self) -> None:
        paragraphs = ["FIRST " * 15, "SECOND " * 1_000]
        chunks = chunk_paragraphs(paragraphs, 6_000)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(normalize_text("\n".join(chunks)), normalize_text("\n".join(paragraphs)))
        self.assertTrue(all(2_000 <= len(chunk) <= 6_000 for chunk in chunks))

    def test_article_chunks_balance_short_tail(self) -> None:
        paragraphs = ["Opening " * 650, "Closing " * 150]
        chunks = chunk_paragraphs(paragraphs, 6_000)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(2_000 <= len(chunk) <= 6_000 for chunk in chunks))
        self.assertEqual(normalize_text("\n".join(chunks)), normalize_text("\n".join(paragraphs)))

    def test_persistent_rate_limit_raises_http_error(self) -> None:
        class RateLimitedResponse:
            status_code = 429

        class RateLimitedSession:
            def get(self, url: str, **kwargs) -> RateLimitedResponse:
                return RateLimitedResponse()

        with patch.object(crawler, "MAX_RETRIES", 1), patch.object(crawler.time, "sleep"):
            with self.assertRaises(requests.HTTPError):
                crawler.crawl_one(RateLimitedSession(), "example")

    def test_checkpoint_retries_temporary_windows_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.json"
            original_replace = Path.replace
            attempts = 0

            def replace_after_locks(path: Path, target: Path) -> Path:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporary file lock")
                return original_replace(path, target)

            with (
                patch.object(crawler, "CHECKPOINT", checkpoint),
                patch.object(crawler.time, "sleep"),
                patch.object(Path, "replace", replace_after_locks),
            ):
                crawler.save_checkpoint({"completed_ids": ["123"]})
            self.assertEqual(attempts, 3)
            self.assertEqual(json.loads(checkpoint.read_text(encoding="utf-8"))["completed_ids"], ["123"])

    def test_discovery_includes_url_without_topic_slug_and_stops_at_local_sitemap(self) -> None:
        root = crawler.PORTAL_ROOT
        gateway = crawler.GATEWAY_DETAIL_TEMPLATE
        central = f"{root}/van-ban/chi-tiet/van-ban--central-1"
        local = f"{root}/van-ban/chi-tiet/quyet-dinh--local-1"
        session = Session({
            crawler.SITEMAP_URL: Response(sitemap(f"{root}/sitemap/1.xml", f"{root}/sitemap/13.xml", f"{root}/sitemap/14.xml")),
            f"{root}/sitemap/1.xml": Response(sitemap(central)),
            f"{root}/sitemap/13.xml": Response(sitemap(local)),
            gateway.format(item_id="central-1"): Response(agency="Quốc hội"),
            gateway.format(item_id="local-1"): Response(agency="UBND Tỉnh Quảng Ngãi"),
        })
        with patch.object(crawler.time, "sleep"):
            candidates = crawler.discover_urls(session, "all_central_documents")
        self.assertEqual([row["url"] for row in candidates], [central])
        self.assertNotIn(f"{root}/sitemap/14.xml", session.requested)

    def test_unmarked_bold_article_headings_are_extracted(self) -> None:
        record = {
            "doc_id": "example",
            "thuoc_tinh": {
                "so_hieu": "53/2014/QH13", "ten_van_ban": "Luật Công chứng",
                "loai_van_ban": "Luật", "co_quan_ban_hanh": "Quốc hội",
                "ngay_ban_hanh": "2014-06-20T00:00:00",
                "trang_thai_hieu_luc": "Hết hiệu lực toàn bộ", "trang_thai_code": "HHL",
                "url_chi_tiet": "https://vbpl.vn/van-ban/chi-tiet/--example",
            },
            "_extra": {},
        }
        soup = BeautifulSoup(
            "<p><strong>Chương I</strong></p><p><strong>QUY ĐỊNH CHUNG</strong></p>"
            "<p><strong>Điều 1. Phạm vi điều chỉnh</strong></p><p>Nội dung 1.</p>"
            "<p><strong>Điều 2. Giải thích từ ngữ</strong></p><p>Nội dung 2.</p>",
            "html.parser",
        )
        rows = extract_articles(record, soup)
        self.assertEqual([(row["article"], row["content"]) for row in rows], [
            ("Điều 1", "Nội dung 1."), ("Điều 2", "Nội dung 2.")
        ])
        self.assertEqual(rows[0]["chapter"], "Chương I")

    def test_title_recovers_document_mislabeled_by_gateway(self) -> None:
        topic, reason = classify(
            {"law_id": "53/2014/QH13", "law_name": "Luật Công chứng số 53/2014/QH13", "fields": ["Công chức"]},
            {},
        )
        self.assertEqual((topic, reason), ("Bổ trợ tư pháp", "document_title"))

    def test_split_keeps_missing_metadata_in_enrichment_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "json"
            source_dir.mkdir()
            base = {
                "doc_id": "one",
                "noi_dung": {"content_html": "<p>Điều 1. Nội dung</p>"},
                "thuoc_tinh": {
                    "so_hieu": "1/2024/QH15", "ten_van_ban": "Luật mẫu", "loai_van_ban": "Luật",
                    "co_quan_ban_hanh": "Quốc hội", "ngay_ban_hanh": "2024-01-01",
                    "trang_thai_hieu_luc": "Còn hiệu lực", "trang_thai_code": "CHL",
                },
            }
            (source_dir / "one.json").write_text(json.dumps(base), encoding="utf-8")
            missing = json.loads(json.dumps(base))
            missing["doc_id"] = "two"
            missing["thuoc_tinh"]["trang_thai_code"] = None
            (source_dir / "two.json").write_text(json.dumps(missing), encoding="utf-8")
            ready = root / "ready.jsonl"
            enrichment = root / "enrichment.jsonl"
            self.assertEqual(split_records(source_dir, ready, enrichment), (1, 1))
            queued = json.loads(enrichment.read_text(encoding="utf-8"))
            self.assertEqual(queued["_enrichment"]["missing_required_fields"], ["thuoc_tinh.trang_thai_code"])


if __name__ == "__main__":
    unittest.main()
