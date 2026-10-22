import unittest

from scripts.augment_active_from_full_processed import augment
from scripts.merge_vbpl_into_active_corpus import TOPICS


def source_row(
    law_id="01/2025/NĐ-CP",
    article_number="1",
    content="Quy định mẫu.",
    unit_type="article",
    record_id="vbpl:test:1",
    law_name="Nghị định mẫu",
):
    return {
        "law_id": law_id,
        "law_name": law_name,
        "article": f"Điều {article_number}" if article_number else None,
        "article_number": article_number,
        "content": content,
        "record_id": record_id,
        "document_id": "doc-1",
        "unit_type": unit_type,
        "quality_flags": [],
        "source_urls": ["https://vbpl.vn/doc-1"],
        "issue_date": "2025-01-01T00:00:00",
        "validity_status": "Còn hiệu lực",
        "fields": [],
        "majors": [],
    }


class ActiveCorpusAugmentationTests(unittest.TestCase):
    def test_adds_missing_labeled_law_and_assigns_question_topic(self):
        test_rows = [{
            "topic": "Dân sự",
            "relevant_docs": ["01/2025/NĐ-CP"],
            "relevant_articles": [],
        }]
        updated, summary = augment([], [source_row()], test_rows)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["topic"], "Dân sự")
        self.assertEqual(updated[0]["topic_assignment_source"], "labeled_test_reference")
        self.assertEqual(summary["added_law_id_count"], 1)

    def test_does_not_add_unparsed_article_as_an_article(self):
        updated, summary = augment([], [source_row(article_number=None)], [], {"01/2025/NĐ-CP"})
        self.assertEqual(updated, [])
        self.assertEqual(summary["unparsed_article_excluded"], 1)

    def test_explicit_out_of_scope_document_is_kept_separately(self):
        row = source_row(
            law_id="59/2020/QH14",
            article_number="218",
            record_id="vbpl:enterprise:218",
            law_name="Luật Doanh nghiệp",
        )
        updated, _ = augment([], [row], [], {"59/2020/QH14"})
        self.assertEqual(updated[0]["topic"], "Ngoài phạm vi 8 chủ đề")
        self.assertEqual(updated[0]["topic_assignment_source"], "explicit_coverage_addition")
        self.assertEqual(updated[0]["validity_tag"], "in_force")

    def test_deduplicates_exact_content_after_whitespace_normalization(self):
        old = {
            **source_row(content="Quy định   mẫu."),
            "topic": TOPICS[2],
            "record_id": "old:1",
        }
        updated, summary = augment([old], [source_row()], [], {"01/2025/NĐ-CP"})
        self.assertEqual(updated, [old])
        self.assertEqual(summary["identical_content_already_present"], 1)

    def test_older_unreferenced_topic_document_is_not_added(self):
        old_active = [{"law_id": "02/2026/NĐ-CP", "issue_date": "2026-08-24T00:00:00"}]
        candidate = source_row(
            law_id="01/2025/NĐ-CP",
            law_name="Nghị định mẫu",
            record_id="vbpl:old:1",
        )
        updated, summary = augment(old_active, [candidate], [], recent_after="2026-08-24")
        self.assertEqual(updated, old_active)
        self.assertEqual(summary["older_unreferenced_source_units_skipped"], 1)

    def test_excludes_future_issue_dates_when_cutoff_is_given(self):
        future = source_row(law_id="01/2026/NĐ-CP", record_id="future", law_name="Luật đất đai")
        future["issue_date"] = "2026-11-01T00:00:00"
        current = source_row(law_id="02/2026/NĐ-CP", record_id="current", law_name="Luật đất đai")
        current["issue_date"] = "2026-09-28T00:00:00"
        updated, summary = augment(
            [], [future, current], [], recent_after="1900-01-01",
            issue_date_through="2026-09-28",
        )
        self.assertEqual([row["law_id"] for row in updated], ["02/2026/NĐ-CP"])
        self.assertEqual(summary["future_issue_date_excluded"], 1)

    def test_repeating_an_oversized_article_does_not_add_chunks(self):
        article = source_row(content="Nội dung pháp luật. " * 400)
        first, first_summary = augment([], [article], [], {"01/2025/NĐ-CP"})
        self.assertGreater(first_summary["added_records"], 1)
        second, second_summary = augment(first, [article], [], {"01/2025/NĐ-CP"})
        self.assertEqual(second, first)
        self.assertEqual(second_summary["added_records"], 0)


if __name__ == "__main__":
    unittest.main()
