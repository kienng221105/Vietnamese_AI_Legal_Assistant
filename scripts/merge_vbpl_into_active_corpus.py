"""Merge preprocessed VBPL documents belonging to the active legal topics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .preprocess_vbpl_corpus import (
        FALLBACK_CHUNK_CHARS, MIN_ARTICLE_CHUNK_CHARS, chunk_paragraphs,
    )
else:
    from preprocess_vbpl_corpus import (
        FALLBACK_CHUNK_CHARS, MIN_ARTICLE_CHUNK_CHARS, chunk_paragraphs,
    )

ROOT = Path(__file__).resolve().parents[1]
CURRENT_PATH = ROOT / "data" / "current" / "legal_corpus.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "legal_corpus.jsonl"
MANIFEST_PATH = ROOT / "data" / "current" / "manifest.json"
REPORT_PATH = ROOT / "reports" / "active_corpus_report.md"
CANDIDATES_PATH = ROOT / "data" / "raw" / "vbpl_gateway" / "candidate_urls_all_central_documents.json"
RAW_DIR = ROOT / "data" / "raw" / "vbpl_gateway" / "json"
CHECKPOINT_PATH = ROOT / "data" / "raw" / "vbpl_gateway" / "checkpoint_v2.json"

TOPICS = (
    "Bổ trợ tư pháp",
    "Dân số, gia đình, trẻ em, bình đẳng giới",
    "Dân sự",
    "Hành chính tư pháp",
    "Thi hành án",
    "Tố tụng và các phương thức giải quyết tranh chấp",
    "Xây dựng, nhà ở, đô thị",
    "Đất đai",
)

# The gateway's documentFields can be absent or wrong; use a specific title when available.
TITLE_RULES = (
    ("Bổ trợ tư pháp", r"công chứng|luật sư|trợ giúp pháp lý|giám định tư pháp|thừa phát lại|đấu giá tài sản"),
    ("Hành chính tư pháp", r"hộ tịch|quốc tịch|lý lịch tư pháp|chứng thực|đăng ký khai sinh|đăng ký khai tử"),
    ("Thi hành án", r"thi hành án|chấp hành viên"),
    ("Tố tụng và các phương thức giải quyết tranh chấp", r"tố tụng|trọng tài thương mại|hòa giải, đối thoại tại tòa án"),
    ("Dân số, gia đình, trẻ em, bình đẳng giới", r"hôn nhân và gia đình|bình đẳng giới|quyền trẻ em|luật trẻ em|nuôi con nuôi|luật dân số|chính sách dân số"),
    ("Đất đai", r"luật đất đai|quyền sử dụng đất|quản lý đất đai|giao đất|thu hồi đất|đăng ký đất đai"),
    ("Xây dựng, nhà ở, đô thị", r"luật xây dựng|luật nhà ở|quản lý nhà chung cư|quy hoạch đô thị|giấy phép xây dựng"),
    ("Dân sự", r"bộ luật dân sự|giao dịch dân sự|bồi thường thiệt hại ngoài hợp đồng|thừa kế"),
)

FIELD_TOPICS = {
    "Bổ trợ tư pháp": "Bổ trợ tư pháp",
    "Công chứng": "Bổ trợ tư pháp",
    "Trợ giúp pháp lý": "Bổ trợ tư pháp",
    "Luật sư": "Bổ trợ tư pháp",
    "Hộ tịch, quốc tịch, chứng thực": "Hành chính tư pháp",
    "Hộ tịch": "Hành chính tư pháp",
    "Lý lịch tư pháp": "Hành chính tư pháp",
    "Nuôi con nuôi": "Dân số, gia đình, trẻ em, bình đẳng giới",
    "Bảo vệ và chăm sóc trẻ em": "Dân số, gia đình, trẻ em, bình đẳng giới",
    "Dân số - Kế hoạch hóa gia đình": "Dân số, gia đình, trẻ em, bình đẳng giới",
    "Thi hành án dân sự": "Thi hành án",
    "Tố tụng dân sự": "Tố tụng và các phương thức giải quyết tranh chấp",
    "Dân sự - Kinh tế": "Dân sự",
    "Xây dựng": "Xây dựng, nhà ở, đô thị",
    "Phát triển đô thị": "Xây dựng, nhà ở, đô thị",
    "Quy hoạch - Kiến trúc": "Xây dựng, nhà ở, đô thị",
    "Kiến trúc, quy hoạch xây dựng": "Xây dựng, nhà ở, đô thị",
    "Đất đai": "Đất đai",
}


def classify(row: dict, existing_law_topics: dict[str, str]) -> tuple[str | None, str | None]:
    law_id = row["law_id"]
    if law_id in existing_law_topics:
        return existing_law_topics[law_id], "existing_law_id"
    title = row["law_name"].casefold()
    for topic, pattern in TITLE_RULES:
        if re.search(pattern, title):
            return topic, "document_title"
    for field in row.get("fields", []):
        topic = FIELD_TOPICS.get(field)
        if topic:
            return topic, "document_field"
    return None, None


def normalized_content(content: str) -> str:
    return " ".join(content.split()).casefold()


def chunk_legacy_article(row: dict, source_key: str) -> list[dict]:
    if len(row.get("content") or "") <= FALLBACK_CHUNK_CHARS:
        return [row]
    chunks = chunk_paragraphs(row["content"].splitlines(), FALLBACK_CHUNK_CHARS)
    if len(chunks) <= 1:
        return [row]
    parent_id = f"phapdien:{hashlib.sha256(source_key.encode()).hexdigest()[:24]}"
    return [
        {
            **row,
            "record_id": f"{parent_id}:chunk:{number}",
            "parent_record_id": parent_id,
            "chunk_number": number,
            "unit_type": "article_chunk",
            "content": content,
            "content_chars": len(content),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
        for number, content in enumerate(chunks, start=1)
    ]


def remove_duplicate_legacy_chunk_articles(rows: list[dict]) -> tuple[list[dict], int, int]:
    chunk_groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        if (
            row.get("unit_type") != "article_chunk"
            or not row.get("law_id")
            or not row.get("article_number")
        ):
            continue
        key = (row["source_system"], row["law_id"], row["article_number"], row["parent_record_id"])
        chunk_groups.setdefault(key, []).append(row)

    vbpl_by_content: dict[tuple[str, str, str], list[dict]] = {}
    legacy_groups: list[tuple[tuple[str, str, str], list[dict]]] = []
    for (source, law_id, article_number, _), chunks in chunk_groups.items():
        ordered = sorted(chunks, key=lambda item: item["chunk_number"])
        content_key = (law_id, article_number, normalized_content("\n".join(
            chunk["content"] for chunk in ordered
        )))
        if source == "vbpl_gateway":
            vbpl_by_content[content_key] = ordered
        elif source == "phapdien":
            legacy_groups.append((content_key, ordered))

    removed_ids: set[str] = set()
    removed_articles = 0
    for content_key, legacy_chunks in legacy_groups:
        vbpl_chunks = vbpl_by_content.get(content_key)
        if not vbpl_chunks:
            continue
        legacy_urls = list(dict.fromkeys(
            url for chunk in legacy_chunks for url in chunk.get("source_urls", [])
        ))
        for chunk in vbpl_chunks:
            chunk["source_urls"] = list(dict.fromkeys([*legacy_urls, *chunk.get("source_urls", [])]))
        removed_ids.update(chunk["record_id"] for chunk in legacy_chunks)
        removed_articles += 1
    retained = [row for row in rows if row.get("record_id") not in removed_ids]
    return retained, removed_articles, len(removed_ids)


def merge() -> dict:
    previous = json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
    previous_ids = {
        row["record_id"] for row in previous
        if row.get("record_id") and row.get("source_system") != "phapdien"
    }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    canonical_path = ROOT / manifest["source"]
    removed_fields = set(manifest.get("removed_fields", []))
    current = []
    seen_legacy_content: set[tuple[str, str | None, str]] = set()
    duplicate_legacy_content = 0
    with canonical_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("topic") not in TOPICS:
                continue
            if row.get("law_id") and row.get("article_number") and row.get("content", "").strip():
                key = (row["law_id"], row["article_number"], normalized_content(row["content"]))
                if key in seen_legacy_content:
                    duplicate_legacy_content += 1
                    continue
                seen_legacy_content.add(key)
            active_row = {key: value for key, value in row.items() if key not in removed_fields}
            active_row.update({
                "unit_type": "article",
                "source_system": "phapdien",
            })
            source_key = str(
                row.get("source_node_id")
                or f"{row.get('source_file')}:{row.get('source_row')}:{row['content_sha256']}"
            )
            current.extend(chunk_legacy_article(active_row, source_key))
    retained_ids: set[str] = set()
    existing_content = {
        (row.get("law_id"), row.get("article_number"), row.get("content_sha256"))
        for row in current
    }
    normalized_articles = {
        (row["law_id"], row["article_number"], normalized_content(row["content"])): index
        for index, row in enumerate(current)
        if row.get("unit_type") == "article" and row.get("law_id")
        and row.get("article_number") and row.get("content", "").strip()
    }
    existing_law_topics = {
        row["law_id"]: row["topic"]
        for row in current
        if row.get("law_id") and row.get("topic") in TOPICS
    }
    counts: Counter[str] = Counter()
    counts["duplicate_legacy_content"] = duplicate_legacy_content
    added_by_topic: Counter[str] = Counter()

    with PROCESSED_PATH.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            counts["processed_records"] += 1
            topic, reason = classify(row, existing_law_topics)
            if topic is None:
                counts["out_of_scope"] += 1
                continue
            key = (row["law_id"], row.get("article_number"), row["content_sha256"])
            content = row["content"]
            if not content.strip():
                counts["empty_content"] += 1
                continue
            normalized_key = (
                (row["law_id"], row["article_number"], normalized_content(content))
                if row["unit_type"] == "article" and row.get("article_number")
                else None
            )
            duplicate_index = normalized_articles.get(normalized_key) if normalized_key else None
            if duplicate_index is not None and current[duplicate_index]["source_system"] == "vbpl_gateway":
                counts["duplicate_normalized_processed"] += 1
                continue
            # An identical chunk may be a shared prefix of two otherwise different article versions.
            if duplicate_index is None and key in existing_content and row["unit_type"] != "article_chunk":
                counts["duplicate_content"] += 1
                continue
            active_row = {
                "law_id": row["law_id"],
                "law_name": row["law_name"],
                "chapter": row.get("chapter"),
                "article": row.get("article"),
                "article_number": row.get("article_number"),
                "article_title": row.get("article_title"),
                "content": content,
                "citation": row["citation"],
                "source_urls": row["source_urls"],
                "section": topic,
                "topic": topic,
                "content_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "quality_status": "ready",
                "document_id": row["document_id"],
                "record_id": row["record_id"],
                "unit_type": row["unit_type"],
                "source_system": "vbpl_gateway",
                "validity_status": row["validity_status"],
                "validity_code": row["validity_code"],
                "issue_date": row["issue_date"],
                "effective_from": row.get("effective_from"),
                "effective_to": row.get("effective_to"),
                "topic_assignment_source": reason,
            }
            if row.get("parent_record_id"):
                active_row["parent_record_id"] = row["parent_record_id"]
            if row.get("chunk_number") is not None:
                active_row["chunk_number"] = row["chunk_number"]
            if duplicate_index is not None:
                active_row["source_urls"] = list(dict.fromkeys(
                    [*current[duplicate_index].get("source_urls", []), *active_row["source_urls"]]
                ))
                current[duplicate_index] = active_row
                counts["replaced_legacy_by_vbpl"] += 1
            else:
                current.append(active_row)
                if normalized_key is not None:
                    normalized_articles[normalized_key] = len(current) - 1
            retained_ids.add(row["record_id"])
            existing_content.add(key)
            if row["record_id"] in previous_ids:
                counts["retained_processed"] += 1
            else:
                counts["added"] += 1
                added_by_topic[topic] += 1

    current, removed_articles, removed_chunks = remove_duplicate_legacy_chunk_articles(current)
    counts["duplicate_legacy_chunk_articles"] = removed_articles
    counts["duplicate_legacy_chunk_records"] = removed_chunks
    counts["superseded_processed_records"] = len(previous_ids - retained_ids)

    manifest["record_count"] = len(current)
    manifest.pop("clause_record_count", None)
    manifest.pop("point_record_count", None)
    manifest["unit_type_counts"] = dict(Counter(row.get("unit_type", "legacy") for row in current))
    manifest["focus_topics"] = list(TOPICS)
    manifest["topic_counts"] = dict(Counter(row["topic"] for row in current))
    manifest["quality_status_counts"] = dict(
        Counter(row.get("quality_status") for row in current)
    )
    manifest["validity_status_counts"] = dict(
        Counter(row.get("validity_status") or "unknown" for row in current)
    )
    manifest["additional_source"] = str(PROCESSED_PATH.relative_to(ROOT)).replace("\\", "/")
    manifest["source_processed_record_count"] = counts["processed_records"]
    manifest["last_merge"] = {**counts, "added_by_topic": dict(added_by_topic)}

    candidate_count = None
    if CANDIDATES_PATH.exists():
        cached = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        candidate_count = len(cached.get("candidates", []))
    raw_count = sum(1 for _ in RAW_DIR.glob("*.json"))
    failed_count = 0
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        failed_count = len(checkpoint.get("failed_ids", {}))
    source_counts = Counter("Pháp điển cũ" if row.get("source_system") == "phapdien" else "VBPL Gateway" for row in current)
    article_variants: dict[tuple[str, str], set[str]] = {}
    normalized_record_counts: Counter[tuple[str, str, str]] = Counter()
    for row in current:
        if row.get("law_id") and row.get("article_number") and row.get("content", "").strip():
            content_key = normalized_content(row["content"])
            article_variants.setdefault((row["law_id"], row["article_number"]), set()).add(
                content_key
            )
            normalized_record_counts[(row["law_id"], row["article_number"], content_key)] += 1
    variant_group_count = sum(len(contents) > 1 for contents in article_variants.values())
    normalized_duplicate_groups = sum(count > 1 for count in normalized_record_counts.values())
    law_53_count = sum(row.get("law_id") == "53/2014/QH13" for row in current)
    report_lines = [
        "# Báo cáo corpus active",
        "",
        f"Cập nhật: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        f"Corpus `data/current/legal_corpus.json` có **{len(current):,} record** từ "
        f"{source_counts['Pháp điển cũ']:,} record Pháp điển cũ và {source_counts['VBPL Gateway']:,} record VBPL Gateway.",
        "",
        "## Cấp đơn vị truy hồi",
        "",
        "- Đơn vị chuẩn là Điều; nội dung không bị tách rời theo Khoản/Điểm.",
        f"- Điều dài được chia thành `article_chunk` mục tiêu {MIN_ARTICLE_CHUNK_CHARS:,}–{FALLBACK_CHUNK_CHARS:,} ký tự theo ranh giới đoạn/câu; `parent_record_id` và `chunk_number` giữ liên kết với Điều gốc.",
        f"- Có {variant_group_count:,} cặp `law_id`–Điều có nhiều nội dung khác nhau; các record này được giữ riêng.",
        f"- Có {normalized_duplicate_groups:,} nhóm record có nội dung tương đương sau chuẩn hóa khoảng trắng; cần xét Điều cha trước khi gộp các chunk này.",
        f"- Trùng nội dung đã loại: {counts['duplicate_legacy_content']:,} bản Pháp điển, {counts['duplicate_normalized_processed']:,} bản VBPL, {counts['duplicate_legacy_chunk_articles']:,} Điều dài Pháp điển; {counts['replaced_legacy_by_vbpl']:,} bản Pháp điển được thay bằng VBPL cùng nội dung đã chuẩn hóa khoảng trắng.",
        f"- Phân bố loại record: {', '.join(f'`{kind}` {count:,}' for kind, count in sorted(manifest['unit_type_counts'].items()))}.",
        "",
        "## Phân bố chủ đề",
        "",
        "| Chủ đề | Số record |",
        "|---|---:|",
        *[f"| {topic} | {manifest['topic_counts'].get(topic, 0):,} |" for topic in TOPICS],
        "",
        "## Chất lượng và hiệu lực",
        "",
        *[f"- Chất lượng `{status}`: {count:,}." for status, count in sorted(manifest["quality_status_counts"].items())],
        *[f"- Hiệu lực `{status}`: {count:,}." for status, count in sorted(manifest["validity_status_counts"].items())],
        "",
        f"- Luật Công chứng `53/2014/QH13`: {law_53_count:,} record trong active.",
        "- `unknown` nghĩa là nguồn Pháp điển cũ chưa có metadata hiệu lực, không có nghĩa văn bản còn hiệu lực.",
        "- Văn bản hết hiệu lực toàn bộ hoặc một phần vẫn được giữ để truy vấn lịch sử.",
        "- Record `quarantine`/`recoverable` vẫn có trong tệp active; khi lập index chính chỉ lấy `quality_status == ready`.",
        "",
        "## Phạm vi crawl",
        "",
        f"- URL trung ương đã phát hiện: {candidate_count:,}." if candidate_count is not None else "- Chưa có candidate cache đầy đủ.",
        f"- Raw JSON hiện có: {raw_count:,}; ID lỗi Gateway đang ghi trong checkpoint: {failed_count:,}.",
        "- Số URL ứng viên nhiều hơn số raw không đồng nghĩa mọi văn bản còn thiếu đều thuộc 8 chủ đề.",
        "- Cần gán nhãn truy hồi dựa trên corpus sau lần refresh cuối cùng, vì `record_id` và số Điều có thể thay đổi khi parser nhận diện thêm Điều từ HTML cũ.",
    ]

    CURRENT_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"total": len(current), **counts, "added_by_topic": dict(added_by_topic)}


if __name__ == "__main__":
    print(json.dumps(merge(), ensure_ascii=False, indent=2))
    from build_unchunked_corpus import build as build_unchunked_corpus

    print(json.dumps(build_unchunked_corpus(), ensure_ascii=False, indent=2))
    from build_full_corpus import build as build_full_corpus

    full_manifest = build_full_corpus()
    print(json.dumps({key: full_manifest[key] for key in (
        "output", "output_records", "missing_candidate_urls", "coverage_complete",
    )}, ensure_ascii=False, indent=2))
    from audit_vbpl_coverage import main as audit_coverage

    audit_coverage()
