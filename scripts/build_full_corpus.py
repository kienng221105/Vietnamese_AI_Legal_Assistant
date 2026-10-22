"""Build a full-text, one-record-per-law-article corpus from VBPL and Phap Dien."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .build_unchunked_corpus import article_key, select_primary, write_json
    from .merge_vbpl_into_active_corpus import TOPICS, classify, normalized_content
else:
    from build_unchunked_corpus import article_key, select_primary, write_json
    from merge_vbpl_into_active_corpus import TOPICS, classify, normalized_content


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "data" / "legacy" / "2026-09-22_pre-domain-focus" / "canonical" / "repaired" / "legal_articles.jsonl"
PROCESSED_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "legal_corpus_full.jsonl"
OUTPUT_PATH = ROOT / "data" / "current" / "legal_corpus_full.json"
MANIFEST_PATH = ROOT / "data" / "current" / "manifest_full.json"
REPORT_PATH = ROOT / "reports" / "full_corpus_report.md"
CANDIDATES_PATH = ROOT / "data" / "raw" / "vbpl_gateway" / "candidate_urls_all_central_documents.json"
RAW_DIR = ROOT / "data" / "raw" / "vbpl_gateway" / "json"
LABEL_PATH = ROOT / "data" / "test" / "origin_labeled_test_set.json"
EXTRA_FIELD_TOPICS = {
    "Nhà ở và công sở": TOPICS[6],
    "Quản lý nhà và thị trường bất động sản": TOPICS[6],
    "Quản lý nhà": TOPICS[6],
    "Xây dựng nhà ở": TOPICS[6],
    "Quản lý hoạt động xây dựng": TOPICS[6],
    "Kinh tế xây dựng": TOPICS[6],
    "Quy hoạch đô thị": TOPICS[6],
    "Quy hoạch đô thị và nông thôn": TOPICS[6],
    "Hành chính tư pháp": TOPICS[3],
    "Giám định tư pháp": TOPICS[0],
    "Tố tụng hành chính": TOPICS[5],
    "Chi phí tố tụng": TOPICS[5],
    "Gia đình": TOPICS[1],
}
EXTRA_TITLE_RULES = (
    (TOPICS[7], re.compile(r"đất đai|quyền sử dụng đất|đăng ký đất|quy hoạch sử dụng đất|giá đất", re.IGNORECASE)),
    (TOPICS[6], re.compile(r"nhà ở|chung cư|quản lý nhà(?!\s+nước)|quy hoạch đô thị|công trình xây dựng", re.IGNORECASE)),
)


def classify_full(row: dict[str, Any], known_law_topics: dict[str, str]) -> tuple[str | None, str | None]:
    topic, reason = classify(row, {})
    if topic is not None:
        return topic, reason
    for field in row.get("fields", []):
        topic = EXTRA_FIELD_TOPICS.get(field)
        if topic is not None:
            return topic, "document_field_extended"
    title = row["law_name"]
    if re.search(r"điều tra dân số và nhà ở", title, re.IGNORECASE):
        return TOPICS[1], "document_title_extended"
    for topic, pattern in EXTRA_TITLE_RULES:
        if pattern.search(title):
            return topic, "document_title_extended"
    return classify(row, known_law_topics)


def source_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "record_id", "document_id", "source_system", "issue_date", "validity_status",
            "effective_from", "effective_to", "source_urls",
        )
    }


def add_unit(groups: dict, row: dict[str, Any], counts: Counter[str]) -> None:
    key = article_key(row)
    if not key[0] or not row.get("content", "").strip():
        counts["missing_identity_or_content"] += 1
        return
    content_key = normalized_content(row["content"])
    variants = groups.setdefault(key, {})
    existing = variants.get(content_key)
    if existing is None:
        variants[content_key] = row
        return

    counts["equivalent_content_sources"] += 1
    preferred, other = (row, existing) if select_primary(row) > select_primary(existing) else (existing, row)
    preferred["source_urls"] = list(dict.fromkeys([
        *preferred.get("source_urls", []), *other.get("source_urls", []),
    ]))
    preferred["equivalent_sources"] = [
        *preferred.get("equivalent_sources", []),
        *other.get("equivalent_sources", []),
        source_summary(other),
    ]
    variants[content_key] = preferred


def build(
    legacy_path: Path = LEGACY_PATH,
    processed_path: Path = PROCESSED_PATH,
    output_path: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    groups: dict[tuple[str, str | None], dict[str, dict[str, Any]]] = {}
    known_law_topics: dict[str, str] = {}
    counts: Counter[str] = Counter()

    with legacy_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("topic") not in TOPICS or row.get("quality_status") != "ready":
                continue
            counts["legacy_ready_records"] += 1
            if row.get("law_id"):
                known_law_topics[row["law_id"]] = row["topic"]
            unit = {
                key: value for key, value in row.items()
                if key not in {
                    "source_node_id", "source_file", "source_row", "mapc", "node_title",
                    "node_type", "law_article_key", "is_indexable", "quality_flags",
                    "repair_actions_applied",
                }
            }
            unit.update({
                "unit_type": "article", "source_system": "phapdien",
                "issue_date": None, "effective_from": None, "effective_to": None,
                "validity_status": None, "document_id": None,
                "content_chars": len(row.get("content") or ""),
                "content_sha256": hashlib.sha256((row.get("content") or "").encode()).hexdigest(),
            })
            add_unit(groups, unit, counts)

    document_ids: set[str] = set()
    excluded_document_ids: set[str] = set()
    processed_law_ids: set[str] = set()
    topic_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    with processed_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            counts["processed_full_records"] += 1
            if row.get("law_id"):
                processed_law_ids.add(row["law_id"].strip().upper())
            topic, reason = classify_full(row, known_law_topics)
            if topic is None:
                counts["unclassified_or_out_of_scope"] += 1
                excluded_document_ids.add(row["document_id"])
                continue
            unit = {
                "law_id": row["law_id"], "law_name": row["law_name"],
                "chapter": row.get("chapter"), "article": row.get("article"),
                "article_number": row.get("article_number"),
                "article_title": row.get("article_title"), "content": row["content"],
                "content_chars": len(row["content"]),
                "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                "citation": row["citation"], "source_urls": row["source_urls"],
                "section": topic, "topic": topic, "quality_status": "ready",
                "document_id": row["document_id"], "record_id": row["record_id"],
                "unit_type": row["unit_type"], "source_system": "vbpl_gateway",
                "issue_date": row["issue_date"],
                "validity_status": row["validity_status"],
                "validity_code": row["validity_code"],
                "effective_from": row.get("effective_from"),
                "effective_to": row.get("effective_to"),
                "document_type": row.get("document_type"),
                "document_type_code": row.get("document_type_code"),
                "issuing_agency": row.get("issuing_agency"),
                "managing_ministry": row.get("managing_ministry"),
                "fields": row.get("fields", []),
                "majors": row.get("majors", []),
                "document_url": row.get("document_url"),
                "pdf_url": row.get("pdf_url"),
                "doc_url": row.get("doc_url"),
                "fetched_at_utc": row.get("fetched_at_utc"),
                "topic_assignment_source": reason,
            }
            document_ids.add(row["document_id"])
            topic_counts[topic] += 1
            assignment_counts[reason or "unknown"] += 1
            add_unit(groups, unit, counts)

    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1] or "")):
        variants = sorted(groups[key].values(), key=select_primary, reverse=True)
        primary = dict(variants[0])
        primary["source_record_id"] = primary.pop("record_id", None)
        primary["record_id"] = "full:" + hashlib.sha256(
            json.dumps(key, ensure_ascii=False).encode()
        ).hexdigest()[:24]
        primary["variant_count"] = len(variants)
        primary["variants"] = variants[1:]
        output.append(primary)

    if len({article_key(row) for row in output}) != len(output):
        raise ValueError("Duplicate law_id/article key in full corpus")
    display_keys = {
        (row["law_id"].strip().upper(), " ".join((row.get("article") or "").split()).casefold())
        for row in output
    }
    if len(display_keys) != len(output):
        raise ValueError("Duplicate displayed law_id/article pair in full corpus")
    if any(row["unit_type"] in {"article_chunk", "document_chunk"} for row in output):
        raise ValueError("Chunk record leaked into full corpus")

    candidate_ids: set[str] | None = None
    candidate_cache_created_at = None
    if CANDIDATES_PATH.exists():
        cache = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        candidate_ids = {
            row["url"].rsplit("/", 1)[-1].rsplit("--", 1)[-1]
            for row in cache.get("candidates", [])
        }
        candidate_cache_created_at = cache.get("created_at_utc")
    raw_ids = {path.stem for path in RAW_DIR.glob("*.json")}
    missing_candidates = len(candidate_ids - raw_ids) if candidate_ids is not None else None
    label_audit = None
    if LABEL_PATH.exists():
        labeled = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        label_law_ids = {
            law_id.strip().strip("*").upper()
            for question in labeled
            for law_id in question.get("relevant_docs", [])
            if isinstance(law_id, str) and law_id.strip()
        }
        output_law_ids = {row["law_id"].strip().upper() for row in output}
        missing = label_law_ids - output_law_ids
        label_audit = {
            "unique_referenced_law_ids": len(label_law_ids),
            "present_in_corpus": len(label_law_ids & output_law_ids),
            "missing_but_processed": sorted(missing & processed_law_ids),
            "missing_from_processed": sorted(missing - processed_law_ids),
            "note": "Historical answer citations can be outside the eight corpus topics; gaps require review, not automatic inclusion.",
        }
    manifest = {
        "source_legacy": str(legacy_path), "source_vbpl": str(processed_path),
        "output": str(output_path), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "focus_topics": list(TOPICS), "source_counts": dict(counts),
        "candidate_urls": len(candidate_ids) if candidate_ids is not None else None,
        "candidate_cache_created_at_utc": candidate_cache_created_at,
        "raw_documents": len(raw_ids),
        "missing_candidate_urls": missing_candidates,
        "vbpl_documents_in_scope": len(document_ids),
        "vbpl_documents_unclassified_or_out_of_scope": len(excluded_document_ids),
        "vbpl_topic_record_counts": dict(topic_counts),
        "topic_assignment_source_counts": dict(assignment_counts),
        "topic_counts": dict(Counter(row["topic"] for row in output)),
        "quality_status_counts": dict(Counter(row.get("quality_status", "unknown") for row in output)),
        "validity_status_counts": dict(Counter(row.get("validity_status") or "unknown" for row in output)),
        "source_counts": dict(Counter(row.get("source_system", "unknown") for row in output)),
        "unit_type_counts": dict(Counter(row.get("unit_type", "unknown") for row in output)),
        "output_records": len(output),
        "primary_validity_status_counts": dict(Counter(
            row.get("validity_status") or "unknown" for row in output
        )),
        "keys_with_alternatives": sum(row["variant_count"] > 1 for row in output),
        "alternative_unit_count": sum(row["variant_count"] - 1 for row in output),
        "latest_issue_date_in_scope": max((row.get("issue_date") or "" for row in output), default=""),
        "max_primary_content_chars": max((row["content_chars"] for row in output), default=0),
        "coverage_complete": missing_candidates == 0 if missing_candidates is not None else False,
        "selection_policy": "Prefer VBPL, then later issue_date, then longer content; distinct text remains in variants.",
        "label_reference_audit": label_audit,
    }
    write_json(output_path, output)
    write_json(manifest_path, manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Báo cáo corpus nguyên văn cấp Điều",
            "",
            f"Cập nhật: {manifest['generated_at_utc']}.",
            f"- File: `{output_path}`; {len(output):,} record, mỗi cặp `law_id`–Điều xuất hiện một lần.",
            f"- Có {manifest['keys_with_alternatives']:,} Điều có nội dung khác nhau; các bản còn lại nằm trong `variants`.",
            f"- Văn bản VBPL thuộc 8 chủ đề đã xử lý: {len(document_ids):,}; ngày ban hành mới nhất: {manifest['latest_issue_date_in_scope'] or 'chưa rõ'}.",
            "- Trạng thái hiệu lực của record chính: " + ", ".join(
                f"{status}: {count:,}" for status, count in sorted(manifest["primary_validity_status_counts"].items())
            ) + ".",
            f"- Candidate URL còn thiếu raw: {manifest['missing_candidate_urls']:,}." if manifest["missing_candidate_urls"] is not None else "- Chưa có candidate cache.",
            f"- Ngoài phạm vi hoặc chưa phân loại: {len(excluded_document_ids):,} văn bản ({counts['unclassified_or_out_of_scope']:,} record); cần rà các ứng viên giao thoa trước khi xác nhận độ phủ 8 chủ đề.",
            (
                f"- Đối chiếu nhãn test: {label_audit['present_in_corpus']:,}/{label_audit['unique_referenced_law_ids']:,} `law_id` có trong corpus; "
                f"{len(label_audit['missing_but_processed']):,} đã preprocess nhưng chưa đưa vào 8 chủ đề, "
                f"{len(label_audit['missing_from_processed']):,} chưa thấy trong dữ liệu processed."
                if label_audit else "- Chưa có tập nhãn gốc để đối chiếu."
            ),
            "- `coverage_complete` chỉ đo độ phủ của candidate cache hiện có; cần làm mới sitemap, đối chiếu Công báo/văn bản hợp nhất và rà nhãn chủ đề trước khi xác nhận đầy đủ.",
            "- Record chính được chọn bằng quy tắc kỹ thuật, không xác nhận đó là bản pháp lý hiện hành; cần xét `validity_status`, ngày hiệu lực và các `variants`.",
        ]) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=LEGACY_PATH)
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    manifest = build(args.legacy, args.processed, args.output, args.manifest, args.report)
    summary = {key: manifest[key] for key in (
        "output", "output_records", "vbpl_documents_in_scope",
        "missing_candidate_urls", "coverage_complete",
    )}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
