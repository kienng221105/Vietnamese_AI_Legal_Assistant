import unittest

from scripts.tag_active_corpus_validity import annotate, document_metadata, validity_tag


def legacy_row(law_id="59/2020/QH14", item_id="142881"):
    return {
        "law_id": law_id,
        "source_system": "phapdien",
        "source_urls": [f"http://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID={item_id}#Dieu_1"],
        "content": "Nội dung Điều 1.",
        "quality_status": "ready",
    }


class ValidityTagTests(unittest.TestCase):
    def test_maps_portal_codes_and_rejects_disagreement(self):
        self.assertEqual(validity_tag("Còn hiệu lực", "CHL"), "in_force")
        self.assertEqual(validity_tag("Hết hiệu lực một phần", "HHL1P"), "partially_expired")
        self.assertEqual(validity_tag("Hết hiệu lực toàn bộ", "CHL"), "unknown")
        self.assertEqual(validity_tag(None, None), "unknown")

    def test_enriches_only_when_item_id_and_law_id_both_match(self):
        metadata = {"142881": ("59/2020/QH14", "Hết hiệu lực một phần", "HHL1P")}
        updated, summary = annotate([legacy_row(), legacy_row("60/2005/QH11")], metadata)
        self.assertEqual(updated[0]["validity_tag"], "partially_expired")
        self.assertEqual(updated[0]["validity_tag_source"], "vbpl_gateway_itemid_match")
        self.assertIsNone(updated[1]["validity_status"])
        self.assertEqual(updated[1]["validity_tag"], "unknown")
        self.assertEqual(summary["reasons"]["source_law_id_mismatch"], 1)

    def test_gateway_record_uses_its_own_metadata(self):
        row = {
            "source_system": "vbpl_gateway",
            "validity_status": "Hết hiệu lực toàn bộ",
            "validity_code": "HHL",
            "content": "Nội dung.",
        }
        updated, _ = annotate([row], {})
        self.assertEqual(updated[0]["validity_tag"], "expired")
        self.assertEqual(updated[0]["validity_tag_source"], "vbpl_gateway_metadata")
        self.assertEqual(updated[0]["content"], row["content"])

    def test_conflicting_gateway_metadata_is_not_used(self):
        source = [
            {"document_id": "142881", "law_id": "59/2020/QH14", "validity_status": "Còn hiệu lực", "validity_code": "CHL"},
            {"document_id": "142881", "law_id": "59/2020/QH14", "validity_status": "Hết hiệu lực toàn bộ", "validity_code": "HHL"},
        ]
        metadata, conflicts = document_metadata(source, {"142881"})
        self.assertEqual(metadata, {})
        self.assertEqual(conflicts, {"142881"})


if __name__ == "__main__":
    unittest.main()
