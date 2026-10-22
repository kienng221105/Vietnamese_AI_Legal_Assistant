"""Convert ready VBPL gateway records into an article-oriented retrieval corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag

if __package__:
    from .split_vbpl_records import replace_with_retry
else:
    from split_vbpl_records import replace_with_retry



ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "vbpl_gateway" / "records_ready_for_conversion.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "vbpl_gateway"
DEFAULT_REPORT = ROOT / "reports" / "vbpl_preprocessing_report.md"
DEFAULT_FULL_REPORT = ROOT / "reports" / "vbpl_full_preprocessing_report.md"
FALLBACK_CHUNK_CHARS = 6_000
MIN_ARTICLE_CHUNK_CHARS = 2_000
ARTICLE_PATTERN = re.compile(r"^Điều\s+([^.:]+)(?:[.:]\s*(.*))?$", re.IGNORECASE)
UNMARKED_ARTICLE_PATTERN = re.compile(r"^Điều\s+\d+[a-zđ]?(?:[.:]|$)", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def direct_text_blocks(soup: BeautifulSoup) -> Iterable[Tag]:
    for tag in soup.find_all(["p", "li"]):
        if tag.find_parent(["p", "li"]) is None:
            yield tag


def tag_classes(tag: Tag) -> set[str]:
    return set(tag.get("class", []))


def is_unmarked_heading(tag: Tag, text: str, pattern: re.Pattern[str]) -> bool:
    if not pattern.match(text):
        return False
    emphasized = tag.find(["strong", "b"])
    return emphasized is not None and normalize_text(emphasized.get_text(" ", strip=True)) == text


def as_list_of_names(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [item["name"] for item in values if isinstance(item, dict) and item.get("name")]


def make_citation(metadata: dict[str, Any]) -> str:
    parts = [metadata.get("loai_van_ban"), metadata.get("so_hieu"), metadata.get("ten_van_ban")]
    citation = " ".join(part for part in parts if part)
    if metadata.get("ngay_ban_hanh"):
        citation += f", ban hành ngày {metadata['ngay_ban_hanh'][:10]}"
    if metadata.get("co_quan_ban_hanh"):
        citation += f" bởi {metadata['co_quan_ban_hanh']}"
    if metadata.get("hieu_luc_tu"):
        citation += f", có hiệu lực từ {metadata['hieu_luc_tu'][:10]}"
    return citation


def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["thuoc_tinh"]
    extra = record.get("_extra", {})
    document_id = record["doc_id"]
    return {
        "document_id": document_id,
        "law_id": metadata["so_hieu"],
        "law_name": metadata["ten_van_ban"],
        "document_type": metadata["loai_van_ban"],
        "document_type_code": metadata.get("loai_code"),
        "issuing_agency": metadata["co_quan_ban_hanh"],
        "managing_ministry": metadata.get("bo_nganh"),
        "issue_date": metadata["ngay_ban_hanh"],
        "effective_from": metadata.get("hieu_luc_tu"),
        "effective_to": metadata.get("hieu_luc_den"),
        "validity_status": metadata["trang_thai_hieu_luc"],
        "validity_code": metadata["trang_thai_code"],
        "document_url": metadata.get("url_chi_tiet"),
        "pdf_url": metadata.get("url_pdf"),
        "doc_url": metadata.get("url_doc"),
        "fields": as_list_of_names(extra.get("document_fields")),
        "majors": as_list_of_names(extra.get("document_majors")),
        "fetched_at_utc": extra.get("fetched_at_utc"),
        "citation": make_citation(metadata),
    }


def make_record_id(document_id: str, unit_type: str, ordinal: int, anchor: str | None) -> str:
    raw = f"{document_id}|{unit_type}|{ordinal}|{anchor or ''}"
    return f"vbpl:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def parse_article_heading(text: str) -> tuple[str | None, str, str | None]:
    match = ARTICLE_PATTERN.match(text)
    if not match:
        return None, text, None
    number = normalize_text(match.group(1))
    title = normalize_text(match.group(2) or "") or None
    return number, f"Điều {number}", title


def extract_articles(record: dict[str, Any], soup: BeautifulSoup) -> list[dict[str, Any]]:
    base = base_metadata(record)
    rows: list[dict[str, Any]] = []
    chapter = None
    chapter_title = None
    awaiting_chapter_title = False
    current: dict[str, Any] | None = None
    use_unmarked_headings = soup.select_one(".prov-article") is None

    def finalize() -> None:
        nonlocal current
        if current is None:
            return
        content = "\n".join(current.pop("paragraphs")).strip()
        if not content:
            content = current["article_title"] or current.pop("heading_text")
            current["quality_flags"] = [
                "article_heading_as_content" if content else "empty_article_content"
            ]
        else:
            current.pop("heading_text")
        current["content"] = content
        current["content_chars"] = len(content)
        current["content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        rows.append(current)
        current = None

    for tag in direct_text_blocks(soup):
        text = normalize_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        classes = tag_classes(tag)

        is_chapter = "prov-chapter" in classes or (
            use_unmarked_headings
            and is_unmarked_heading(tag, text, re.compile(r"^Chương\s+[IVXLCDM\d]+$", re.I))
        )
        if is_chapter:
            finalize()
            chapter = text
            chapter_title = None
            awaiting_chapter_title = True
            continue

        is_article = "prov-article" in classes or (
            use_unmarked_headings and is_unmarked_heading(tag, text, UNMARKED_ARTICLE_PATTERN)
        )
        if is_article:
            finalize()
            article_number, article, article_title = parse_article_heading(text)
            anchor = tag.get("id")
            source_url = f"{base['document_url']}#{anchor}" if anchor else base["document_url"]
            current = {
                **base,
                "record_id": make_record_id(base["document_id"], "article", len(rows) + 1, anchor),
                "unit_type": "article",
                "chapter": chapter,
                "chapter_title": chapter_title,
                "article": article,
                "article_number": article_number,
                "article_title": article_title,
                "heading_text": text,
                "source_urls": [source_url] if source_url else [],
                "paragraphs": [],
                "quality_flags": [],
            }
            awaiting_chapter_title = False
            continue

        if awaiting_chapter_title:
            chapter_title = text
            awaiting_chapter_title = False
            continue

        if current is not None:
            current["paragraphs"].append(text)

    finalize()
    return rows


def visible_paragraphs(soup: BeautifulSoup) -> list[str]:
    return [
        text
        for tag in direct_text_blocks(soup)
        if (text := normalize_text(tag.get_text(" ", strip=True)))
    ]


def split_near_boundary(
    text: str, minimum: int, maximum: int, target: int, right_maximum: int | None = None
) -> tuple[str, str] | None:
    boundaries: list[tuple[int, int, int, int]] = []
    for match in re.finditer(r"\s+", text):
        left_length = match.start()
        right_length = len(text) - match.end()
        if not (minimum <= left_length <= maximum and right_length >= minimum):
            continue
        if right_maximum is not None and right_length > right_maximum:
            continue
        priority = (
            0 if "\n" in match.group()
            else 1 if text[left_length - 1] in ".;:?!"
            else 2
        )
        boundaries.append((priority, abs(left_length - target), match.start(), match.end()))
    if not boundaries:
        return None
    _, _, start, end = min(boundaries)
    return text[:start].rstrip(), text[end:].lstrip()


def chunk_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    minimum = min(MIN_ARTICLE_CHUNK_CHARS, max_chars // 3)
    chunks: list[str] = []
    current = ""
    for original_paragraph in paragraphs:
        paragraph = original_paragraph.strip()
        if not paragraph:
            continue
        pieces: list[str] = []
        while len(paragraph) > max_chars:
            remaining_count = (len(paragraph) + max_chars - 1) // max_chars
            target = len(paragraph) // remaining_count
            split = split_near_boundary(
                paragraph,
                minimum,
                max_chars,
                target,
            )
            if split is None:
                break  # An unbreakable token must not lose characters.
            part, paragraph = split
            pieces.append(part)
        pieces.append(paragraph)
        for piece in pieces:
            combined = f"{current}\n{piece}" if current else piece
            if current and len(combined) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = combined
    if current:
        chunks.append(current)

    index = 0
    while len(chunks) > 1 and index < len(chunks):
        if len(chunks[index]) >= minimum:
            index += 1
            continue
        left_index = index - 1 if index else 0
        right_index = index if index else 1
        combined = f"{chunks[left_index]}\n{chunks[right_index]}"
        if len(combined) <= max_chars:
            chunks[left_index:right_index + 1] = [combined]
        else:
            split = split_near_boundary(
                combined,
                max(minimum, len(combined) - max_chars),
                max_chars,
                len(combined) // 2,
                max_chars,
            )
            if split is None:
                index += 1
                continue
            chunks[left_index:right_index + 1] = list(split)
        index = max(0, left_index - 1)
    return chunks


def expand_oversized_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        chunks = chunk_paragraphs(row["content"].splitlines(), FALLBACK_CHUNK_CHARS)
        if len(chunks) == 1:
            expanded.append(row)
            continue

        for chunk_number, content in enumerate(chunks, start=1):
            chunk = {
                **row,
                "parent_record_id": row["record_id"],
                "record_id": f"{row['record_id']}:chunk:{chunk_number}",
                "unit_type": f"{row['unit_type']}_chunk",
                "chunk_number": chunk_number,
                "content": content,
                "content_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "quality_flags": [*row["quality_flags"], "chunked_from_oversized_unit"],
            }
            expanded.append(chunk)
    return expanded


def extract_document_chunks(record: dict[str, Any], soup: BeautifulSoup) -> list[dict[str, Any]]:
    document = extract_document_record(record, soup)
    if not document["content"]:
        return []
    chunks = chunk_paragraphs(document["content"].splitlines(), FALLBACK_CHUNK_CHARS)
    rows = []
    for ordinal, content in enumerate(chunks, start=1):
        rows.append(
            {
                **document,
                "record_id": make_record_id(document["document_id"], "document_chunk", ordinal, None),
                "unit_type": "document_chunk",
                "chunk_number": ordinal,
                "content": content,
                "content_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    return rows


def extract_document_record(record: dict[str, Any], soup: BeautifulSoup) -> dict[str, Any]:
    base = base_metadata(record)
    content = "\n".join(visible_paragraphs(soup)) or normalize_text(soup.get_text(" ", strip=True))
    return {
        **base,
        "record_id": make_record_id(base["document_id"], "document", 1, None),
        "unit_type": "document",
        "chapter": None,
        "chapter_title": None,
        "article": None,
        "article_number": None,
        "article_title": None,
        "source_urls": [base["document_url"]] if base["document_url"] else [],
        "content": content,
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "quality_flags": ["no_article_marker"],
    }


def preprocess(input_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "legal_documents.jsonl"
    corpus_path = output_dir / "legal_corpus.jsonl"
    full_corpus_path = output_dir / "legal_corpus_full.jsonl"
    counters: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    validation: Counter[str] = Counter()
    record_ids: set[str] = set()
    documents_temp = documents_path.with_suffix(".jsonl.tmp")
    corpus_temp = corpus_path.with_suffix(".jsonl.tmp")
    full_corpus_temp = full_corpus_path.with_suffix(".jsonl.tmp")

    try:
        with (
            input_path.open(encoding="utf-8") as source,
            documents_temp.open("w", encoding="utf-8", newline="\n") as document_output,
            corpus_temp.open("w", encoding="utf-8", newline="\n") as corpus_output,
            full_corpus_temp.open("w", encoding="utf-8", newline="\n") as full_corpus_output,
        ):
            for line in source:
                record = json.loads(line)
                soup = BeautifulSoup(record["noi_dung"]["content_html"], "html.parser")
                base = base_metadata(record)
                articles = extract_articles(record, soup)
                full_rows = articles or [extract_document_record(record, soup)]
                rows = expand_oversized_units(articles or extract_document_chunks(record, soup))
                document = {
                    **base,
                    "source_file": f"{record['doc_id']}.json",
                    "unit_count": len(rows),
                    "unit_type": "article" if articles else "document_chunk",
                }
                document_output.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
                counters["documents"] += 1
                counters["article_documents"] += bool(articles)
                counters["fallback_documents"] += not bool(articles)
                for full_row in full_rows:
                    if full_row["content"].strip():
                        full_corpus_output.write(json.dumps(full_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counters["full_corpus_records"] += 1
                counters["empty_article_records"] += sum(
                    "empty_article_content" in row["quality_flags"] for row in articles
                )
                for row in rows:
                    corpus_output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counters["corpus_records"] += 1
                    counters["article_records"] += row["unit_type"] in {"article", "article_chunk"}
                    counters["document_chunk_records"] += row["unit_type"] == "document_chunk"
                    counters["oversized_unit_chunks"] += "chunked_from_oversized_unit" in row["quality_flags"]
                    quality_flags.update(row["quality_flags"])
                    validation["empty_content_records"] += not row["content"].strip()
                    validation["records_over_6000_chars"] += row["content_chars"] > FALLBACK_CHUNK_CHARS
                    validation["article_chunks_under_2000_chars"] += (
                        row["unit_type"] == "article_chunk" and row["content_chars"] < MIN_ARTICLE_CHUNK_CHARS
                    )
                    validation["records_missing_core_metadata"] += any(
                        not row.get(field) for field in ("law_id", "issue_date", "validity_code")
                    )
                    validation["invalid_source_urls"] += not row["source_urls"] or not all(
                        url.startswith("https://vbpl.vn/") for url in row["source_urls"]
                    )
                    validation["duplicate_record_ids"] += row["record_id"] in record_ids
                    record_ids.add(row["record_id"])
        replace_with_retry(documents_temp, documents_path)
        replace_with_retry(corpus_temp, corpus_path)
        replace_with_retry(full_corpus_temp, full_corpus_path)
    finally:
        documents_temp.unlink(missing_ok=True)
        corpus_temp.unlink(missing_ok=True)
        full_corpus_temp.unlink(missing_ok=True)

    report = {
        "source": str(input_path.relative_to(ROOT)),
        "documents_output": str(documents_path.relative_to(ROOT)),
        "corpus_output": str(corpus_path.relative_to(ROOT)),
        "full_corpus_output": str(full_corpus_path.relative_to(ROOT)),
        "schema_version": "vbpl-gateway-article-v1",
        "full_schema_version": "vbpl-gateway-full-article-v1",
        **dict(counters),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation_labels = {
        "empty_content_records": "Record có nội dung rỗng",
        "duplicate_record_ids": "Record ID trùng lặp",
        "records_over_6000_chars": "Record vượt quá 6.000 ký tự",
        "article_chunks_under_2000_chars": "Chunk cấp Điều ngắn hơn 2.000 ký tự",
        "records_missing_core_metadata": "Record thiếu metadata cốt lõi",
        "invalid_source_urls": "URL nguồn không hợp lệ",
    }
    report_lines = [
        "# Báo cáo tiền xử lý dữ liệu VBPL",
        "",
        "## Phạm vi",
        "",
        f"- Nguồn đầu vào: `{report['source']}`.",
        f"- Số văn bản đầu vào: {counters['documents']:,}.",
        f"- Số record phục vụ truy xuất: {counters['corpus_records']:,}.",
        f"- Số record đầy đủ cấp Điều/văn bản, không chia chunk: {counters['full_corpus_records']:,}.",
        "",
        "## Các bước tiền xử lý",
        "",
        "1. Chỉ giữ record đã đạt điều kiện: HTML có nội dung thực và đầy đủ metadata cốt lõi.",
        "2. Phân tích marker HTML `prov-chapter` và `prov-article`; với HTML không có marker, nhận diện heading Chương/Điều được in đậm.",
        "3. Dùng chính heading Điều làm nội dung khi portal đặt toàn bộ quy định trên heading đó.",
        "4. Tạo chunk theo đoạn văn cho văn bản không có marker Điều.",
        f"5. Chỉ chia Điều dài quá {FALLBACK_CHUNK_CHARS:,} ký tự; ưu tiên ranh giới đoạn/câu và cân lại chunk ngắn để đạt tối thiểu {MIN_ARTICLE_CHUNK_CHARS:,} ký tự khi có thể.",
        "6. Giữ lại URL nguồn, metadata hiệu lực, định danh văn bản và hash nội dung SHA-256.",
        "",
        "## Cấu trúc đầu ra",
        "",
        f"- Record được suy ra từ Điều: {counters['article_records']:,}.",
        f"- Record chunk cấp văn bản: {counters['document_chunk_records']:,}.",
        f"- Chunk được tách từ đơn vị quá dài: {counters['oversized_unit_chunks']:,}.",
        "",
        "## Kiểm tra chất lượng",
        "",
        *[f"- {validation_labels[key]}: {value:,}." for key, value in validation.items()],
        "",
        "## Cờ cần rà soát",
        "",
        *[f"- `{key}`: {value:,}." for key, value in sorted(quality_flags.items())],
        "",
        "`article_heading_as_content` là trường hợp dự kiến đối với quyết định ngắn có toàn bộ quy định nằm trong heading Điều. "
        "`no_article_marker` chỉ các chunk dự phòng cấp văn bản; cần rà soát riêng nếu yêu cầu corpus chỉ gồm cấp Điều.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return report


def preprocess_full_only(input_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "legal_corpus_full.jsonl"
    temporary = corpus_path.with_suffix(".jsonl.tmp")
    counters: Counter[str] = Counter()
    record_ids: set[str] = set()
    completed = False
    try:
        with input_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8", newline="\n") as output:
            for line in source:
                record = json.loads(line)
                soup = BeautifulSoup(record["noi_dung"]["content_html"], "html.parser")
                articles = extract_articles(record, soup)
                rows = articles or [extract_document_record(record, soup)]
                counters["documents"] += 1
                counters["article_documents"] += bool(articles)
                counters["fallback_documents"] += not bool(articles)
                for row in rows:
                    if not row["content"].strip():
                        counters["empty_content_records"] += 1
                        continue
                    if row["record_id"] in record_ids:
                        raise ValueError(f"Duplicate full record ID: {row['record_id']}")
                    record_ids.add(row["record_id"])
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counters["records"] += 1
                    counters["article_records"] += row["unit_type"] == "article"
                    counters["document_records"] += row["unit_type"] == "document"
                    counters["max_content_chars"] = max(counters["max_content_chars"], len(row["content"]))
        completed = True
    finally:
        if not completed:
            temporary.unlink(missing_ok=True)
    replace_with_retry(temporary, corpus_path)

    manifest = {
        "source": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "output": str(corpus_path.relative_to(ROOT)).replace("\\", "/"),
        "schema_version": "vbpl-gateway-full-article-v1",
        **dict(counters),
    }
    (output_dir / "manifest_full.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join([
        "# Báo cáo tiền xử lý VBPL không chunk",
        "",
        f"- Nguồn: `{manifest['source']}`; {counters['documents']:,} văn bản.",
        f"- Đầu ra: `{manifest['output']}`; {counters['records']:,} record đầy đủ.",
        f"- Record cấp Điều: {counters['article_records']:,}; record cấp văn bản chưa nhận diện Điều: {counters['document_records']:,}.",
        f"- Nội dung dài nhất: {counters['max_content_chars']:,} ký tự; không chia chunk.",
        f"- Văn bản/Điều không có nội dung sau parse: {counters['empty_content_records']:,}.",
    ]) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()
    report_path = args.report_output or (DEFAULT_FULL_REPORT if args.full_only else DEFAULT_REPORT)
    processor = preprocess_full_only if args.full_only else preprocess
    print(
        json.dumps(
            processor(args.input, args.output_dir, report_path),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
