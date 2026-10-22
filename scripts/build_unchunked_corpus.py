"""Build a full-text backup with one top-level record per law and article."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .split_vbpl_records import replace_with_retry
else:
    from split_vbpl_records import replace_with_retry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "current" / "legal_corpus.json"
DEFAULT_OUTPUT = ROOT / "data" / "current" / "legal_corpus_unchunked.json"
DEFAULT_MANIFEST = ROOT / "data" / "current" / "manifest_unchunked.json"
ARTICLE_NUMBER = re.compile(r"^điều\s+(\d+(?:\.\d+)?[a-zđ]?)\b", re.IGNORECASE)


def article_key(row: dict[str, Any]) -> tuple[str, str | None]:
    law_id = row["law_id"].strip().upper()
    number = str(row.get("article_number") or "").strip().rstrip(".:").casefold()
    if number:
        return law_id, number
    article = " ".join((row.get("article") or "").split()).casefold()
    if article:
        match = ARTICLE_NUMBER.match(article)
        return law_id, match.group(1) if match else article
    return law_id, None


def complete_unit(rows: list[dict[str, Any]], unit_type: str) -> dict[str, Any]:
    if len(rows) > 1 or rows[0].get("chunk_number") is not None:
        rows = sorted(rows, key=lambda row: row["chunk_number"])
        numbers = [row["chunk_number"] for row in rows]
        if numbers != list(range(1, len(rows) + 1)):
            raise ValueError(f"Missing or duplicate chunks for {rows[0].get('parent_record_id') or rows[0].get('document_id')}")
    first = rows[0]
    identity = (first.get("law_id"), first.get("article_number"), first.get("article"))
    if any((row.get("law_id"), row.get("article_number"), row.get("article")) != identity for row in rows):
        raise ValueError(f"Inconsistent metadata among chunks for {identity}")
    content = "\n".join(row["content"] for row in rows)
    unit = {
        key: value for key, value in first.items()
        if key not in {"record_id", "parent_record_id", "chunk_number", "content", "content_chars", "content_sha256", "source_urls", "unit_type"}
    }
    unit.update({
        "unit_type": unit_type,
        "content": content,
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source_urls": list(dict.fromkeys(url for row in rows for url in row.get("source_urls", []))),
        "source_record_ids": list(dict.fromkeys(
            row["record_id"] for row in rows if row.get("record_id")
        )),
    })
    return unit


def select_primary(unit: dict[str, Any]) -> tuple[Any, ...]:
    return (
        unit.get("source_system") == "vbpl_gateway",
        unit.get("issue_date") or "",
        unit["content_chars"],
        unit.get("document_id") or "",
        unit["content_sha256"],
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        completed = True
        assert temporary is not None
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None and (not completed or not temporary.exists()):
            temporary.unlink(missing_ok=True)


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
          manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    source_rows = json.loads(input_path.read_text(encoding="utf-8"))
    article_chunks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    document_chunks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    units: list[dict[str, Any]] = []
    skipped_non_ready = skipped_missing_law_id = 0

    for row in source_rows:
        if row.get("quality_status") != "ready":
            skipped_non_ready += 1
            continue
        if not row.get("law_id"):
            skipped_missing_law_id += 1
            continue
        if row.get("unit_type") == "article_chunk":
            parent_id = row.get("parent_record_id")
            if not parent_id or row.get("chunk_number") is None:
                raise ValueError(f"Article chunk lacks parent or position: {row.get('record_id')}")
            article_chunks[(row.get("source_system") or "", parent_id)].append(row)
        elif row.get("unit_type") == "document_chunk":
            document_id = row.get("document_id")
            if not document_id or row.get("chunk_number") is None:
                raise ValueError(f"Document chunk lacks document or position: {row.get('record_id')}")
            document_chunks[(row.get("source_system") or "", document_id)].append(row)
        else:
            units.append(complete_unit([row], "article" if row.get("article") else "document"))

    units.extend(complete_unit(rows, "article") for rows in article_chunks.values())
    units.extend(complete_unit(rows, "document") for rows in document_chunks.values())
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[article_key(unit)].append(unit)

    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1] or "")):
        variants = sorted(groups[key], key=select_primary, reverse=True)
        primary = dict(variants[0])
        primary["record_id"] = "unchunked:" + hashlib.sha256(
            json.dumps(key, ensure_ascii=False).encode()
        ).hexdigest()[:24]
        primary["variant_count"] = len(variants)
        primary["variants"] = variants[1:]
        output.append(primary)

    display_keys = [(row["law_id"].strip().upper(), " ".join((row.get("article") or "").split()).casefold())
                    for row in output]
    if len(display_keys) != len(set(display_keys)):
        raise ValueError("Duplicate law_id/article pair in unchunked corpus")
    if any(row.get("chunk_number") is not None or row.get("unit_type") in {"article_chunk", "document_chunk"}
           for row in output):
        raise ValueError("Chunk record leaked into unchunked corpus")

    manifest = {
        "source": str(input_path),
        "source_modified_at_utc": datetime.fromtimestamp(input_path.stat().st_mtime, timezone.utc).isoformat(),
        "output": str(output_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_record_count": len(source_rows),
        "source_ready_record_count": len(source_rows) - skipped_non_ready,
        "skipped_non_ready_records": skipped_non_ready,
        "skipped_ready_records_without_law_id": skipped_missing_law_id,
        "assembled_article_count": len(article_chunks),
        "assembled_document_count": len(document_chunks),
        "complete_unit_count": len(units),
        "output_record_count": len(output),
        "keys_with_alternatives": sum(len(group) > 1 for group in groups.values()),
        "alternative_unit_count": sum(len(group) - 1 for group in groups.values()),
        "max_primary_content_chars": max((row["content_chars"] for row in output), default=0),
        "selection_policy": "Prefer VBPL, then later issue_date, then longer content; alternatives remain in variants.",
    }
    write_json(output_path, output)
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
