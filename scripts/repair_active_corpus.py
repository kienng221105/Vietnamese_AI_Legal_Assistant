"""Repair safe active-corpus metadata and materialize citation-ready records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATH = ROOT / "data/current/legal_corpus.json"
MANIFEST_PATH = ROOT / "data/current/manifest.json"
CITABLE_PATH = ROOT / "data/current/legal_corpus_citable.json"
ARTICLE_NUMBER = re.compile(r"\d+[A-Za-zĐđ]?\Z")
UNCITABLE_LAW_IDS = {"Không số"}


def legacy_record_id(row: dict[str, Any]) -> str:
    identity = [
        row.get("law_id"),
        row.get("article_number"),
        row.get("content_sha256"),
        row.get("source_urls"),
    ]
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"phapdien:article:{digest}"


def citation_level(row: dict[str, Any], focus_topics: set[str]) -> str:
    if (
        row.get("quality_status") != "ready"
        or row.get("topic") not in focus_topics
        or not row.get("law_id")
        or row["law_id"].strip() in UNCITABLE_LAW_IDS
        or not str(row.get("content") or "").strip()
        or not row.get("source_urls")
    ):
        return "none"
    unit_type = row.get("unit_type")
    if unit_type in {"article", "article_chunk"}:
        return "article" if ARTICLE_NUMBER.fullmatch(str(row.get("article_number") or "")) else "none"
    if unit_type in {"document", "document_chunk"}:
        return "document"
    return "none"


def repair(rows: list[dict[str, Any]], focus_topics: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired = [dict(row) for row in rows]
    seen_ids = {row["record_id"] for row in repaired if row.get("record_id")}
    if len(seen_ids) != sum(bool(row.get("record_id")) for row in repaired):
        raise ValueError("Active corpus already contains duplicate record IDs")
    counters: Counter[str] = Counter()

    ready_chunks = {
        (row.get("document_id"), row.get("law_id"), row.get("chunk_number"), row.get("content_sha256")):
        row["record_id"]
        for row in repaired
        if row.get("unit_type") == "document_chunk"
        and row.get("quality_status") == "ready"
        and row.get("record_id")
    }
    for row in repaired:
        if (
            row.get("source_system") == "phapdien"
            and not row.get("record_id")
            and str(row.get("content") or "").strip()
            and row.get("source_urls")
        ):
            record_id = legacy_record_id(row)
            if record_id in seen_ids:
                raise ValueError(f"Generated record ID collision: {record_id}")
            row["record_id"] = record_id
            seen_ids.add(record_id)
            counters["legacy_record_ids_added"] += 1

        if row.get("quality_status") == "ready" and row.get("unit_type") in {"article", "article_chunk"}:
            number = str(row.get("article_number") or "")
            if not ARTICLE_NUMBER.fullmatch(number):
                flag = "missing_article_number" if not number else "invalid_article_number"
                row["quality_status"] = "recoverable"
                row["quality_flags"] = list(dict.fromkeys([*(row.get("quality_flags") or []), flag]))
                counters[f"demoted_{flag}"] += 1

        if row.get("unit_type") == "document_chunk" and row.get("quality_status") == "recoverable":
            key = (row.get("document_id"), row.get("law_id"), row.get("chunk_number"), row.get("content_sha256"))
            duplicate_of = ready_chunks.get(key)
            if duplicate_of and duplicate_of != row.get("record_id"):
                row["duplicate_of_record_id"] = duplicate_of
                counters["recoverable_document_chunks_marked_duplicate"] += 1

        row["citation_level"] = citation_level(row, focus_topics)
        counters[f"citation_{row['citation_level']}"] += 1

    counters["records"] = len(repaired)
    return repaired, dict(counters)


def write_json_atomic(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(dry_run: bool = False) -> dict[str, Any]:
    rows = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("record_count") != len(rows):
        raise ValueError("Active corpus does not match manifest record count")
    repaired, summary = repair(rows, set(manifest["focus_topics"]))
    citable = [row for row in repaired if row["citation_level"] != "none"]
    if len(citable) != summary["citation_article"] + summary["citation_document"]:
        raise ValueError("Citation counts do not match citable records")
    if dry_run:
        return summary

    manifest["quality_status_counts"] = dict(Counter(row.get("quality_status") for row in repaired))
    manifest["citation_level_counts"] = dict(Counter(row["citation_level"] for row in repaired))
    manifest["metadata_repair"] = {
        **summary,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "citation_corpus": str(CITABLE_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json_atomic(CITABLE_PATH, citable)
    write_json_atomic(ACTIVE_PATH, repaired)
    write_json_atomic(MANIFEST_PATH, manifest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.dry_run), ensure_ascii=False, indent=2))
