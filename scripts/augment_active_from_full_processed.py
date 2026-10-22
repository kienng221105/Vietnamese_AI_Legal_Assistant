"""Add eligible, already-crawled VBPL units to the existing active corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .build_full_corpus import classify_full
    from .merge_vbpl_into_active_corpus import TOPICS, normalized_content
    from .preprocess_vbpl_corpus import expand_oversized_units
    from .tag_active_corpus_validity import validity_tag
else:
    from build_full_corpus import classify_full
    from merge_vbpl_into_active_corpus import TOPICS, normalized_content
    from preprocess_vbpl_corpus import expand_oversized_units
    from tag_active_corpus_validity import validity_tag


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATH = ROOT / "data" / "current" / "legal_corpus.json"
MANIFEST_PATH = ROOT / "data" / "current" / "manifest.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "legal_corpus_full.jsonl"
LABEL_PATH = ROOT / "data" / "test" / "origin_labeled_test_set.json"
REPORT_PATH = ROOT / "reports" / "active_corpus_patch_report.md"
OUT_OF_SCOPE_TOPIC = "Ngoài phạm vi 8 chủ đề"
TOPIC_ALIASES = {
    "Hôn nhân, gia đình, trẻ em, bình đẳng giới": TOPICS[1],
}


def canonical_law_id(value: Any) -> str:
    return "".join(str(value or "").split()).strip("*").upper()


def topic_references(test_rows: Iterable[dict[str, Any]]) -> dict[str, Counter[str]]:
    references: dict[str, Counter[str]] = defaultdict(Counter)
    for question in test_rows:
        topic = TOPIC_ALIASES.get(question.get("topic"), question.get("topic"))
        if not topic:
            continue
        for law_id in question.get("relevant_docs", []):
            if isinstance(law_id, str) and canonical_law_id(law_id):
                references[canonical_law_id(law_id)][topic] += 1
        for article in question.get("relevant_articles", []):
            if isinstance(article, dict) and canonical_law_id(article.get("law_id")):
                references[canonical_law_id(article["law_id"])][topic] += 1
    return references


def content_signature(law_id: Any, article_number: Any, content: str) -> tuple[str, str, str]:
    return (
        canonical_law_id(law_id),
        str(article_number or "").strip().casefold(),
        normalized_content(content),
    )


def existing_signatures(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    signatures = {
        content_signature(row.get("law_id"), row.get("article_number"), row.get("content", ""))
        for row in rows
        if row.get("content") and row.get("article_number")
    }
    chunks: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("parent_record_id") and row.get("article_number"):
            key = (
                canonical_law_id(row.get("law_id")),
                str(row.get("article_number") or "").strip().casefold(),
                str(row["parent_record_id"]),
            )
            chunks[key].append(row)
    for (law_id, article_number, _), parts in chunks.items():
        text = "\n".join(
            row.get("content", "")
            for row in sorted(parts, key=lambda item: item.get("chunk_number", 0))
        )
        signatures.add(content_signature(law_id, article_number, text))
    return signatures


def make_active_row(
    source: dict[str, Any], topic: str, assignment_source: str,
) -> dict[str, Any]:
    content = source["content"]
    quality_flags = source.get("quality_flags", [])
    row = {
        "law_id": source["law_id"],
        "law_name": source["law_name"],
        "chapter": source.get("chapter"),
        "article": source.get("article"),
        "article_number": source.get("article_number"),
        "article_title": source.get("article_title"),
        "content": content,
        "citation": source.get("citation"),
        "source_urls": source.get("source_urls", []),
        "section": topic,
        "topic": topic,
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "quality_status": "recoverable" if quality_flags else "ready",
        "quality_flags": quality_flags,
        "document_id": source.get("document_id"),
        "record_id": source["record_id"],
        "unit_type": source["unit_type"],
        "source_system": "vbpl_gateway",
        "validity_status": source.get("validity_status"),
        "validity_code": source.get("validity_code"),
        "validity_tag": validity_tag(source.get("validity_status"), source.get("validity_code")),
        "validity_tag_source": "vbpl_gateway_metadata",
        "issue_date": source.get("issue_date"),
        "effective_from": source.get("effective_from"),
        "effective_to": source.get("effective_to"),
        "topic_assignment_source": assignment_source,
    }
    if source.get("document_url"):
        row["document_url"] = source["document_url"]
    if source.get("fields"):
        row["fields"] = source["fields"]
    if source.get("majors"):
        row["majors"] = source["majors"]
    return row


def augment(
    active_rows: list[dict[str, Any]],
    processed_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    explicit_law_ids: set[str] | None = None,
    recent_after: str | None = None,
    excluded_law_ids: set[str] | None = None,
    issue_date_through: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    references = topic_references(test_rows)
    explicit = {canonical_law_id(item) for item in (explicit_law_ids or set())}
    excluded = {canonical_law_id(item) for item in (excluded_law_ids or set())}
    active_dates = [row.get("issue_date") for row in active_rows if row.get("issue_date")]
    cutoff = recent_after or max(active_dates, default=None)
    existing_topics = {
        row["law_id"]: row["topic"]
        for row in active_rows
        if row.get("law_id") and row.get("topic") in TOPICS
    }
    signatures = existing_signatures(active_rows)
    record_ids = {row.get("record_id") for row in active_rows if row.get("record_id")}
    additions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    processed_law_ids: set[str] = set()
    dates: list[str] = []
    selected_dates: list[str] = []
    selected_documents: set[str] = set()
    added_law_ids: set[str] = set()
    added_quality: Counter[str] = Counter()
    longest_added_content = 0

    for source in processed_rows:
        if issue_date_through and source.get("issue_date") and source["issue_date"][:10] > issue_date_through:
            counters["future_issue_date_excluded"] += 1
            continue
        law_id_key = canonical_law_id(source.get("law_id"))
        if not law_id_key:
            counters["missing_law_id"] += 1
            continue
        processed_law_ids.add(law_id_key)
        if source.get("issue_date"):
            dates.append(source["issue_date"])
        is_test_reference = law_id_key in references
        is_explicit = law_id_key in explicit
        is_recent = bool(cutoff and source.get("issue_date") and source["issue_date"][:10] >= cutoff[:10])
        if law_id_key in excluded:
            counters["excluded_after_topic_review"] += 1
            continue
        if not (is_test_reference or is_explicit or is_recent):
            counters["older_unreferenced_source_units_skipped"] += 1
            continue
        topic, reason = classify_full(source, existing_topics)
        cited_topics = references.get(law_id_key)
        if topic is None and cited_topics:
            topic = cited_topics.most_common(1)[0][0]
            reason = "labeled_test_reference"
        if topic is None and law_id_key in explicit:
            topic = OUT_OF_SCOPE_TOPIC
            reason = "explicit_coverage_addition"
        if topic is None:
            counters["not_selected_by_topic_or_reference"] += 1
            continue

        is_article = source.get("unit_type") == "article"
        if is_article and not source.get("article_number"):
            counters["unparsed_article_excluded"] += 1
            continue
        if source.get("unit_type") not in {"article", "document"}:
            counters["unsupported_unit_type"] += 1
            continue
        if not source.get("content", "").strip():
            counters["empty_content"] += 1
            continue

        signature = content_signature(
            law_id_key, source.get("article_number"), source["content"],
        )
        if signature in signatures:
            counters["identical_content_already_present"] += 1
            continue

        candidate = make_active_row(source, topic, reason or "topic_classifier")
        expanded = expand_oversized_units([candidate])
        existing_chunk_ids = [row["record_id"] for row in expanded if row["record_id"] in record_ids]
        if existing_chunk_ids:
            counter_key = (
                "already_present_by_record_id" if len(existing_chunk_ids) == len(expanded)
                else "partial_record_id_collision"
            )
            counters[counter_key] += 1
            signatures.add(signature)
            continue
        if source.get("document_id"):
            selected_documents.add(str(source["document_id"]))
        if source.get("issue_date"):
            selected_dates.append(source["issue_date"])
        for row in expanded:
            record_ids.add(row["record_id"])
            additions.append(row)
            added_quality[row.get("quality_status", "unknown")] += 1
            longest_added_content = max(longest_added_content, len(row.get("content", "")))
        signatures.add(signature)
        added_law_ids.add(law_id_key)
        counters["added_source_units"] += 1
        counters["added_active_records"] += len(expanded)
        counters[f"assigned:{topic}"] += 1

    result = [*active_rows, *additions]
    missing_test_refs = sorted(set(references) - processed_law_ids - {
        canonical_law_id(row.get("law_id")) for row in active_rows
    })
    summary = {
        **dict(counters),
        "before_records": len(active_rows),
        "after_records": len(result),
        "added_records": len(additions),
        "added_law_ids": sorted(added_law_ids),
        "added_law_id_count": len(added_law_ids),
        "selected_source_documents": len(selected_documents),
        "added_quality_status_counts": dict(added_quality),
        "max_added_content_chars": longest_added_content,
        "referenced_law_ids": len(references),
        "referenced_ids_missing_from_active_and_processed": missing_test_refs,
        "latest_issue_date_in_source": max(dates, default=None),
        "issue_date_through": issue_date_through,
        "latest_issue_date_added": max(selected_dates, default=None),
        "topic_refresh_cutoff": cutoff,
        "included_explicit_law_ids": sorted(explicit & processed_law_ids),
        "explicit_law_ids_not_found_in_processed": sorted(explicit - processed_law_ids),
    }
    return result, summary


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_patch_report(
    report_path: Path,
    summary: dict[str, Any],
    processed_path: Path,
    generated_at_utc: str,
) -> None:
    unresolved = summary["referenced_ids_missing_from_active_and_processed"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join([
        "# Báo cáo vá corpus active từ dữ liệu Gateway đã crawl",
        "",
        f"- Thời điểm: {generated_at_utc}.",
        f"- Nguồn đã preprocess: `{processed_path}`; ngày ban hành mới nhất trong nguồn: {summary['latest_issue_date_in_source']}.",
        f"- Chọn văn bản được nhãn test tham chiếu, số hiệu yêu cầu tường minh hoặc văn bản từ mốc `{summary['topic_refresh_cutoff']}` trở đi.",
        f"- Record active trước/sau: {summary['before_records']:,}/{summary['after_records']:,}; thêm {summary['added_records']:,} record từ {summary['selected_source_documents']:,} văn bản.",
        f"- Chất lượng record mới: {summary['added_quality_status_counts']}; nội dung dài nhất sau chunk: {summary['max_added_content_chars']:,} ký tự.",
        f"- Đơn vị nguồn trùng nội dung đã có: {summary.get('identical_content_already_present', 0):,}.",
        f"- Dòng không parse được số Điều đã bỏ qua: {summary.get('unparsed_article_excluded', 0):,}.",
        f"- Đối chiếu {summary['referenced_law_ids']:,} số hiệu trong nhãn test; còn {len(unresolved):,} số hiệu chưa có trong active lẫn nguồn preprocess.",
        "- Các số hiệu test còn thiếu: " + (", ".join(unresolved) or "không có") + ".",
        "- Bản vá chỉ dùng dữ liệu đã crawl và preprocess sẵn; không tiếp tục crawl candidate còn thiếu.",
        "- Các Điều không parse được `article_number` không được đưa vào corpus active; cần xử lý riêng theo cấu trúc HTML nguồn.",
    ]) + "\n", encoding="utf-8")


def run(
    active_path: Path = ACTIVE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    processed_path: Path = PROCESSED_PATH,
    labels_path: Path = LABEL_PATH,
    report_path: Path = REPORT_PATH,
    explicit_law_ids: set[str] | None = None,
    recent_after: str | None = None,
    excluded_law_ids: set[str] | None = None,
    dry_run: bool = False,
    issue_date_through: str | None = None,
) -> dict[str, Any]:
    active = json.loads(active_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    updated, summary = augment(
        active, read_jsonl(processed_path), labels, explicit_law_ids,
        recent_after, excluded_law_ids, issue_date_through,
    )
    if dry_run or not summary["added_records"]:
        return summary

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = len(updated)
    manifest["unit_type_counts"] = dict(Counter(row.get("unit_type", "legacy") for row in updated))
    manifest["quality_status_counts"] = dict(Counter(row.get("quality_status", "unknown") for row in updated))
    manifest["validity_status_counts"] = dict(Counter(row.get("validity_status") or "unknown" for row in updated))
    if any("validity_tag" in row for row in updated):
        manifest["validity_tag_counts"] = dict(Counter(row.get("validity_tag") or "unknown" for row in updated))
    manifest["topic_counts"] = dict(Counter(row.get("topic", "unknown") for row in updated))
    manifest["patch_source"] = str(processed_path.relative_to(ROOT)).replace("\\", "/")
    manifest["last_raw_patch"] = {
        **summary,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    active_temp = active_path.with_suffix(active_path.suffix + ".tmp")
    manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        active_temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        active_temp.replace(active_path)
        manifest_temp.replace(manifest_path)
    finally:
        active_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)

    write_patch_report(
        report_path, summary, processed_path,
        manifest["last_raw_patch"]["generated_at_utc"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", type=Path, default=ACTIVE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--labels", type=Path, default=LABEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--include-law-id", action="append", default=[])
    parser.add_argument("--exclude-law-id", action="append", default=[])
    parser.add_argument("--recent-after", help="Mốc ngày YYYY-MM-DD; mặc định là ngày mới nhất đang có trong corpus active.")
    parser.add_argument("--issue-date-through", help="Chỉ thêm văn bản ban hành đến ngày YYYY-MM-DD.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Tạo lại báo cáo từ manifest của bản vá gần nhất.")
    args = parser.parse_args()
    if args.report_only:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        summary = manifest["last_raw_patch"]
        write_patch_report(args.report, summary, args.processed, summary["generated_at_utc"])
        print(f"Đã tạo {args.report}")
        return
    summary = run(
        args.active, args.manifest, args.processed, args.labels, args.report,
        {canonical_law_id(item) for item in args.include_law_id}, args.recent_after,
        {canonical_law_id(item) for item in args.exclude_law_id}, args.dry_run,
        args.issue_date_through,
    )
    printable = {
        key: value for key, value in summary.items()
        if key not in {"added_law_ids", "referenced_ids_missing_from_active_and_processed"}
    }
    printable["added_law_id_count"] = summary.get("added_law_id_count", 0)
    printable["unresolved_test_reference_count"] = len(
        summary.get("referenced_ids_missing_from_active_and_processed", [])
    )
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
