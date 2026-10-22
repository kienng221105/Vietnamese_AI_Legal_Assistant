"""Checks conservative metadata repairs for the active legal corpus."""

import unittest

from scripts.repair_active_corpus import citation_level, repair


class RepairActiveCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.focus = {"Dân sự"}
        self.article = {
            "law_id": "91/2015/QH13", "article": "Điều 1", "article_number": "1",
            "content": "Nội dung Điều 1.", "content_sha256": "digest",
            "source_urls": ["https://vbpl.vn/doc/1"], "topic": "Dân sự",
            "source_system": "phapdien", "unit_type": "article", "quality_status": "ready",
        }

    def test_assigns_stable_legacy_id_without_changing_source_fields(self) -> None:
        repaired, summary = repair([self.article], self.focus)
        self.assertEqual(summary["legacy_record_ids_added"], 1)
        self.assertTrue(repaired[0]["record_id"].startswith("phapdien:article:"))
        self.assertEqual(repaired[0]["citation_level"], "article")
        self.assertEqual(repaired[0]["content"], self.article["content"])
        repeated, repeated_summary = repair(repaired, self.focus)
        self.assertEqual(repeated, repaired)
        self.assertNotIn("legacy_record_ids_added", repeated_summary)

    def test_uncertain_article_number_is_preserved_and_demoted(self) -> None:
        rows = [
            {**self.article, "record_id": "one", "article_number": None},
            {**self.article, "record_id": "two", "article_number": "1 0"},
        ]
        repaired, summary = repair(rows, self.focus)
        self.assertEqual(summary["demoted_missing_article_number"], 1)
        self.assertEqual(summary["demoted_invalid_article_number"], 1)
        self.assertEqual([row["article_number"] for row in repaired], [None, "1 0"])
        self.assertEqual([row["quality_status"] for row in repaired], ["recoverable"] * 2)
        self.assertEqual([row["citation_level"] for row in repaired], ["none"] * 2)
        repeated, repeated_summary = repair(repaired, self.focus)
        self.assertEqual(repeated, repaired)
        self.assertNotIn("demoted_missing_article_number", repeated_summary)

    def test_unnumbered_document_remains_in_corpus_but_is_not_citable(self) -> None:
        row = {**self.article, "law_id": "Không số", "record_id": "one"}
        repaired, _ = repair([row], self.focus)
        self.assertEqual(repaired[0]["quality_status"], "ready")
        self.assertEqual(repaired[0]["law_id"], "Không số")
        self.assertEqual(repaired[0]["citation_level"], "none")
        self.assertEqual(citation_level({**self.article, "unit_type": "document_chunk"}, self.focus), "document")

    def test_marks_duplicate_recoverable_chunk_without_deleting_it(self) -> None:
        base = {**self.article, "unit_type": "document_chunk", "article": None,
                "article_number": None, "source_system": "vbpl_gateway",
                "document_id": "doc-1", "chunk_number": 1}
        ready = {**base, "record_id": "ready"}
        recoverable = {**base, "record_id": "copy", "quality_status": "recoverable"}
        repaired, summary = repair([ready, recoverable], self.focus)
        self.assertEqual(len(repaired), 2)
        self.assertEqual(repaired[1]["duplicate_of_record_id"], "ready")
        self.assertEqual(summary["recoverable_document_chunks_marked_duplicate"], 1)
        self.assertEqual([row["citation_level"] for row in repaired], ["document", "none"])

    def test_generated_id_collision_fails_without_mutating_input(self) -> None:
        row = {**self.article, "record_id": "one"}
        with self.assertRaises(ValueError):
            repair([row, row], self.focus)
        self.assertEqual(row["record_id"], "one")


if __name__ == "__main__":
    unittest.main()
