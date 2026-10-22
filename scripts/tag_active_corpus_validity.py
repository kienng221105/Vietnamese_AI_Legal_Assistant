"""Attach document-level validity tags to the active legal corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlparse


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATH = ROOT / "data" / "current" / "legal_corpus.json"
MANIFEST_PATH = ROOT / "data" / "current" / "manifest.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "legal_corpus_full.jsonl"
REPORT_PATH = ROOT / "reports" / "active_corpus_validity_report.md"

CODE_TO_STATUS_TAG = {
    "CHL": ("Còn hiệu lực", "in_force"),
    "HHL": ("Hết hiệu lực toàn bộ", "expired"),
    "HHL1P": ("Hết hiệu lực một phần", "partially_expired"),
    "CCHL": ("Chưa có hiệu lực", "not_yet_effective"),
    "TNHL": ("Ngưng hiệu lực", "suspended"),
}
STATUS_TO_TAG = {status: tag for status, tag in CODE_TO_STATUS_TAG.values()}


def canonical_law_id(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def validity_tag(status: Any, code: Any) -> str:
    normalized_status = str(status or "").strip()
    normalized_code = str(code or "").strip().upper()
    if normalized_code:
        expected = CODE_TO_STATUS_TAG.get(normalized_code)
        if expected is None or (normalized_status and normalized_status != expected[0]):
            return "unknown"
        return expected[1]
    return STATUS_TO_TAG.get(normalized_status, "unknown")


def source_item_id(row: dict[str, Any]) -> str | None:
    item_ids = {
        value
        for url in row.get("source_urls", [])
        for key, value in parse_qsl(urlparse(url).query)
        if key.casefold() == "itemid" and value
    }
    return next(iter(item_ids)) if len(item_ids) == 1 else None


def document_metadata(
    source_rows: Iterable[dict[str, Any]], wanted_ids: set[str],
) -> tuple[dict[str, tuple[str, str | None, str | None]], set[str]]:
    metadata: dict[str, tuple[str, str | None, str | None]] = {}
    conflicting_ids: set[str] = set()
    for row in source_rows:
        document_id = str(row.get("document_id") or "")
        if document_id not in wanted_ids or document_id in conflicting_ids:
            continue
        value = (
            canonical_law_id(row.get("law_id")),
            row.get("validity_status"),
            row.get("validity_code"),
        )
        if document_id in metadata and metadata[document_id] != value:
            metadata.pop(document_id)
            conflicting_ids.add(document_id)
        else:
            metadata[document_id] = value
    return metadata, conflicting_ids


def annotate(
    rows: list[dict[str, Any]],
    metadata: dict[str, tuple[str, str | None, str | None]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    tag_sources: Counter[str] = Counter()
    changed = 0

    for original in rows:
        row = dict(original)
        status = row.get("validity_status")
        code = row.get("validity_code")
        tag_source = row.get("validity_tag_source")

        if row.get("source_system") == "phapdien" and not status:
            item_id = source_item_id(row)
            source = metadata.get(item_id) if item_id else None
            if source is None:
                reasons["no_matching_source_document"] += 1
            elif canonical_law_id(row.get("law_id")) != source[0]:
                reasons["source_law_id_mismatch"] += 1
            elif validity_tag(source[1], source[2]) == "unknown":
                reasons["invalid_source_status"] += 1
            else:
                status, code = source[1], source[2]
                tag_source = "vbpl_gateway_itemid_match"
                reasons["phapdien_enriched_by_source_itemid"] += 1

        row["validity_status"] = status
        row["validity_code"] = code
        row["validity_tag"] = validity_tag(status, code)
        if not tag_source:
            tag_source = "vbpl_gateway_metadata" if row.get("source_system") == "vbpl_gateway" else "unverified"
        row["validity_tag_source"] = tag_source

        tags[row["validity_tag"]] += 1
        tag_sources[tag_source] += 1
        changed += row != original
        result.append(row)

    return result, {
        "record_count": len(rows),
        "changed_records": changed,
        "validity_tag_counts": dict(tags),
        "validity_tag_source_counts": dict(tag_sources),
        "reasons": dict(reasons),
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_report(path: Path, summary: dict[str, Any], source_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "# Báo cáo gắn tag hiệu lực cho corpus active",
        "",
        f"- Cập nhật: {summary['generated_at_utc']}.",
        f"- Corpus: {summary['record_count']:,} record; {summary['changed_records']:,} record được bổ sung hoặc cập nhật trường hiệu lực.",
        f"- Metadata đối chiếu: `{source_path}`.",
        "- `validity_tag` phản ánh hiệu lực **cấp văn bản nguồn**, không xác nhận hiệu lực của từng Điều/khoản.",
        "- Nguồn Pháp điển chỉ được bổ sung trạng thái khi `ItemID` trong URL nguồn và `law_id` cùng khớp metadata Gateway.",
        "- Tag `unknown` giữ nguyên khi không có nguồn khớp hoặc metadata mâu thuẫn; không suy hiệu lực từ nội dung Điều.",
        "",
        "| Trạng thái VBPL | `validity_tag` |",
        "|---|---|",
        *[f"| {status} | `{tag}` |" for status, tag in CODE_TO_STATUS_TAG.values()],
        "| Chưa xác định | `unknown` |",
        "",
        "## Phân bố tag",
        "",
        "| Tag | Số record |",
        "|---|---:|",
        *[f"| `{tag}` | {count:,} |" for tag, count in sorted(summary["validity_tag_counts"].items())],
        "",
        "## Nguồn xác nhận",
        "",
        *[f"- `{source}`: {count:,}." for source, count in sorted(summary["validity_tag_source_counts"].items())],
        "",
        "## Kết quả đối chiếu Pháp điển với Gateway",
        "",
        *[f"- `{reason}`: {count:,}." for reason, count in sorted(summary["reasons"].items())],
    ]) + "\n", encoding="utf-8")


def run(
    active_path: Path = ACTIVE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    processed_path: Path = PROCESSED_PATH,
    report_path: Path = REPORT_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    active = json.loads(active_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("record_count") != len(active):
        raise ValueError("Active corpus count does not match its manifest")

    wanted_ids = {
        item_id
        for row in active
        if row.get("source_system") == "phapdien" and not row.get("validity_status")
        if (item_id := source_item_id(row))
    }
    metadata, conflicts = document_metadata(read_jsonl(processed_path), wanted_ids)
    updated, summary = annotate(active, metadata)
    summary["source_document_ids_matched"] = len(metadata)
    summary["source_document_ids_conflicting"] = len(conflicts)
    if dry_run or not summary["changed_records"]:
        return summary

    generated_at = datetime.now(timezone.utc).isoformat()
    summary["generated_at_utc"] = generated_at
    manifest["validity_status_counts"] = dict(Counter(row.get("validity_status") or "unknown" for row in updated))
    manifest["validity_tag_counts"] = summary["validity_tag_counts"]
    manifest["validity_tagging"] = summary

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
    write_report(report_path, summary, processed_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", type=Path, default=ACTIVE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.active, args.manifest, args.processed, args.report, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
