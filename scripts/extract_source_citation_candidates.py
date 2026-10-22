"""Collect citation candidates from old public Q&A answers for later review."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "test" / "project_scope_questions_without_topic.json"
SOURCE_PATH = ROOT / "data" / "external" / "pbgdpl-vn-legal-qna" / "data" / "qa.jsonl"
EVIDENCE_PATH = ROOT / "data" / "review" / "source_citation_candidates.jsonl"

LAW_ID = re.compile(r"(?<!\w)\d{1,4}/(?:19|20)\d{2}/[A-ZĐa-zđ][A-ZĐa-zđ0-9-]*\b")
ARTICLE = re.compile(r"\bĐiều\s+(\d+[a-zđ]?)\b", re.IGNORECASE)
MAX_CITATION_DISTANCE = 100


def question_key(question: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFC", question).casefold()
        if char.isalnum()
    )


def link_source(question: str, by_question: dict[str, dict]) -> tuple[dict | None, str]:
    key = question_key(question)
    if key in by_question:
        return by_question[key], "exact"
    pool = [
        candidate for candidate in by_question
        if abs(len(candidate) - len(key)) <= max(12, len(key) // 10)
        and candidate[:8] == key[:8]
    ]
    if not pool:
        pool = [
            candidate for candidate in by_question
            if abs(len(candidate) - len(key)) <= max(12, len(key) // 10)
        ]
    scores = sorted(
        ((difflib.SequenceMatcher(None, key, candidate, autojunk=False).ratio(), candidate)
         for candidate in pool),
        reverse=True,
    )
    if not scores or scores[0][0] < 0.97:
        return None, "unmatched"
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.01:
        return None, "ambiguous"
    return by_question[scores[0][1]], "near_exact"


def same_citation_clause(text: str, start: int, end: int) -> bool:
    between = text[min(start, end):max(start, end)]
    return not re.search(r"[;.!?\n]|\bĐiều\s+\d+", between, re.IGNORECASE)


def extract_citations(answer: str) -> tuple[set[str], set[tuple[str, str]]]:
    laws = list(LAW_ID.finditer(answer))
    articles = list(ARTICLE.finditer(answer))
    document_ids = {match.group().upper() for match in laws}
    article_pairs: set[tuple[str, str]] = set()

    for article in articles:
        candidates = []
        for law in laws:
            if law.start() >= article.end():
                gap = law.start() - article.end()
                left, right = article.end(), law.start()
            else:
                gap = article.start() - law.end()
                left, right = law.end(), article.start()
            if gap < 0 or gap > MAX_CITATION_DISTANCE:
                continue
            if same_citation_clause(answer, left, right):
                candidates.append((gap, law))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 10:
            continue
        law_id = candidates[0][1].group().upper()
        article_name = f"Điều {article.group(1).lower()}"
        article_pairs.add((law_id, article_name))

    return document_ids, article_pairs


def extract_candidates() -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in SOURCE_PATH.open(encoding="utf-8")]
    by_question = {question_key(row["question_text"]): row for row in source_rows}
    evidence_rows = []
    counts: Counter[str] = Counter()

    for question in questions:
        source, match_type = link_source(question["question"], by_question)
        counts[f"match_{match_type}"] += 1
        docs, articles = extract_citations(source["answer_text"]) if source else (set(), set())
        evidence_rows.append({
            "id": question["id"],
            "source_item_id": source["item_id"] if source else None,
            "source_url": source["source_url"] if source else None,
            "question_match": match_type,
            "extracted_law_ids": sorted(docs),
            "extracted_article_pairs": [list(pair) for pair in sorted(articles)],
            "needs_semantic_and_legal_review": True,
        })
        counts["questions_with_docs"] += bool(docs)
        counts["questions_with_articles"] += bool(articles)

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for row in evidence_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"questions": len(questions), **counts}


if __name__ == "__main__":
    print(json.dumps(extract_candidates(), ensure_ascii=False, indent=2))
