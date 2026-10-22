"""Report discovered, downloaded, processed and active VBPL coverage."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "vbpl_gateway"
CANDIDATES_PATH = RAW_ROOT / "candidate_urls_all_central_documents.json"
CHECKPOINT_PATH = RAW_ROOT / "checkpoint_v2.json"
READY_PATH = RAW_ROOT / "records_ready_for_conversion.jsonl"
PROCESSED_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "legal_documents.jsonl"
FULL_PREPROCESS_MANIFEST_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "manifest_full.json"
ACTIVE_PATH = ROOT / "data" / "current" / "legal_corpus.json"
REPORT_PATH = ROOT / "reports" / "vbpl_coverage_report.md"


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def main() -> None:
    cached = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = cached["candidates"]
    candidate_ids = {row["url"].rsplit("/", 1)[-1].rsplit("--", 1)[-1] for row in candidates}
    raw_ids = {path.stem for path in (RAW_ROOT / "json").glob("*.json")}
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    active = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    pending = candidate_ids - raw_ids
    failed = set(checkpoint.get("failed_ids", {}))
    law_53 = [row for row in active if row.get("law_id") == "53/2014/QH13"]
    full_preprocessed_documents = None
    if FULL_PREPROCESS_MANIFEST_PATH.exists():
        full_preprocessed_documents = json.loads(
            FULL_PREPROCESS_MANIFEST_PATH.read_text(encoding="utf-8")
        ).get("documents")
    lines = [
        "# Báo cáo độ phủ VBPL Gateway",
        "",
        f"- URL ứng viên trung ương: {len(candidate_ids):,}.",
        f"- Raw JSON đã tải: {len(raw_ids):,}.",
        f"- URL chưa có raw: {len(pending):,}.",
        f"- ID lỗi Gateway còn ghi trong checkpoint: {len(failed):,}.",
        f"- Văn bản ready ở lần tách gần nhất: {count_jsonl(READY_PATH):,}.",
        f"- Văn bản đã preprocess ở luồng chunked cũ: {count_jsonl(PROCESSED_PATH):,}.",
        f"- Văn bản đã preprocess nguyên Điều từ raw ready: {full_preprocessed_documents:,}."
        if full_preprocessed_documents is not None else "- Chưa có manifest preprocess nguyên Điều.",
        f"- Record trong corpus active: {len(active):,}.",
        f"- Luật Công chứng `53/2014/QH13` trong active: {len(law_53):,} record, "
        f"trong đó {sum(bool(row.get('article')) for row in law_53):,} record có Điều.",
        "",
        "Số URL ứng viên chưa tải là thước đo tiến độ crawl, không phải số văn bản chắc chắn thuộc tám chủ đề. "
        "Các ID Gateway trả HTTP 500 cần xử lý riêng; không được tính là đã crawl thành công.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã cập nhật {REPORT_PATH}: {len(pending):,} URL chưa có raw")


if __name__ == "__main__":
    main()
