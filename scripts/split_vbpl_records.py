"""Split VBPL raw records into conversion-ready and enrichment queues."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "vbpl_gateway" / "json"
DEFAULT_READY = (
    ROOT / "data" / "raw" / "vbpl_gateway" / "records_ready_for_conversion.jsonl"
)
DEFAULT_ENRICHMENT = (
    ROOT / "data" / "raw" / "vbpl_gateway" / "records_need_enrichment.jsonl"
)


def has_value(value: Any) -> bool:
    return value is not None and value != ""


def has_meaningful_html(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    without_noncontent = re.sub(
        r"<(script|style)[^>]*>.*?</\\1>", " ", value, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<[^>]+>", " ", without_noncontent)
    return bool(re.sub(r"\s+", " ", html.unescape(text)).strip())


REQUIRED_FIELDS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "noi_dung.content_html": lambda record: has_meaningful_html(
        record.get("noi_dung", {}).get("content_html")
    ),
    "thuoc_tinh.so_hieu": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("so_hieu")
    ),
    "thuoc_tinh.ten_van_ban": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("ten_van_ban")
    ),
    "thuoc_tinh.loai_van_ban": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("loai_van_ban")
    ),
    "thuoc_tinh.co_quan_ban_hanh": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("co_quan_ban_hanh")
    ),
    "thuoc_tinh.ngay_ban_hanh": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("ngay_ban_hanh")
    ),
    "thuoc_tinh.trang_thai_hieu_luc": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("trang_thai_hieu_luc")
    ),
    "thuoc_tinh.trang_thai_code": lambda record: has_value(
        record.get("thuoc_tinh", {}).get("trang_thai_code")
    ),
}


def missing_fields(record: dict[str, Any]) -> list[str]:
    return [name for name, check in REQUIRED_FIELDS.items() if not check(record)]


def replace_with_retry(temporary_path: Path, path: Path) -> None:
    for attempt in range(10):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))


def split_records(
    input_directory: Path, ready_path: Path, enrichment_path: Path
) -> tuple[int, int]:
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    enrichment_path.parent.mkdir(parents=True, exist_ok=True)
    ready_temp = ready_path.with_suffix(f"{ready_path.suffix}.tmp")
    enrichment_temp = enrichment_path.with_suffix(f"{enrichment_path.suffix}.tmp")
    ready_count = enrichment_count = 0
    try:
        with (
            ready_temp.open("w", encoding="utf-8", newline="\n") as ready_output,
            enrichment_temp.open("w", encoding="utf-8", newline="\n") as enrichment_output,
        ):
            for source_path in sorted(input_directory.glob("*.json")):
                with source_path.open(encoding="utf-8") as source:
                    record = json.load(source)
                missing = missing_fields(record)
                if missing:
                    record["_enrichment"] = {
                        "missing_required_fields": missing,
                        "source_file": source_path.name,
                    }
                    output = enrichment_output
                    enrichment_count += 1
                else:
                    output = ready_output
                    ready_count += 1
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        replace_with_retry(ready_temp, ready_path)
        replace_with_retry(enrichment_temp, enrichment_path)
    finally:
        ready_temp.unlink(missing_ok=True)
        enrichment_temp.unlink(missing_ok=True)
    return ready_count, enrichment_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ready-output", type=Path, default=DEFAULT_READY)
    parser.add_argument("--enrichment-output", type=Path, default=DEFAULT_ENRICHMENT)
    args = parser.parse_args()

    ready_count, enrichment_count = split_records(
        args.input_dir, args.ready_output, args.enrichment_output
    )
    print(f"Ready: {ready_count}")
    print(f"Needs enrichment: {enrichment_count}")


if __name__ == "__main__":
    main()
