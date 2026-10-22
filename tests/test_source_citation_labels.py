"""Checks for conservative citation pairing in public Q&A answers."""

import unittest

from scripts.extract_source_citation_candidates import extract_citations


class SourceCitationLabelTests(unittest.TestCase):
    def test_pairs_each_article_with_its_own_document(self) -> None:
        answer = (
            "Theo Điều 21 Nghị định số 63/2011/NĐ-CP; "
            "khoản 3 Điều 1 Nghị định số 124/2018/NĐ-CP."
        )
        documents, articles = extract_citations(answer)
        self.assertEqual(documents, {"63/2011/NĐ-CP", "124/2018/NĐ-CP"})
        self.assertEqual(articles, {
            ("63/2011/NĐ-CP", "Điều 21"),
            ("124/2018/NĐ-CP", "Điều 1"),
        })

    def test_does_not_attach_an_unnumbered_law_to_later_decree(self) -> None:
        answer = (
            "Điều 122 Bộ luật Dân sự quy định điều kiện giao dịch; "
            "Điều 146 Nghị định 181/2004/NĐ-CP quy định việc đăng ký."
        )
        _, articles = extract_citations(answer)
        self.assertEqual(articles, {("181/2004/NĐ-CP", "Điều 146")})


if __name__ == "__main__":
    unittest.main()
