"""Add all crawled Land Law records to the active corpus, preserving validity metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PATH = ROOT / "data" / "current" / "legal_corpus.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "vbpl_gateway" / "legal_corpus.jsonl"
MANIFEST_PATH = ROOT / "data" / "current" / "manifest.json"


def main() -> None:
    current = json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
    land = []
    for line in PROCESSED_PATH.open(encoding="utf-8"):
        row = json.loads(line)
        if "Đất đai" not in row.get("fields", []):
            continue
        content = row["content"]
        land.append({
            "law_id": row["law_id"], "law_name": row["law_name"],
            "chapter": row.get("chapter"), "article": row.get("article"),
            "article_number": row.get("article_number"), "article_title": row.get("article_title"),
            "content": content, "citation": row["citation"], "source_urls": row["source_urls"],
            "section": "Đất đai", "topic": "Đất đai", "content_chars": len(content),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "quality_status": "ready",
            "document_id": row["document_id"], "record_id": row["record_id"],
            "validity_status": row["validity_status"], "validity_code": row["validity_code"],
            "effective_from": row.get("effective_from"), "effective_to": row.get("effective_to"),
        })
    existing = {(r["law_id"], r.get("article_number"), r["content_sha256"]) for r in current}
    current.extend(r for r in land if (r["law_id"], r.get("article_number"), r["content_sha256"]) not in existing)
    CURRENT_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["record_count"] = len(current)
    if "Đất đai" not in manifest["focus_topics"]:
        manifest["focus_topics"].append("Đất đai")
    manifest["topic_counts"]["Đất đai"] = len(land)
    manifest["excluded_topics"] = [x for x in manifest.get("excluded_topics", []) if not x.startswith("Đất đai")]
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã thêm {len(land)} record Đất đai; corpus có {len(current)} record.")


if __name__ == "__main__":
    main()
