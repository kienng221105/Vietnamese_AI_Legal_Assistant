#!/usr/bin/env python3
"""
VBPL Gateway Direct API Crawler v2
===================================
Crawl legal documents from the VBPL gateway API and output
in R2AI-compatible JSON format.

Key differences from v1:
- Calls Gateway API directly (no Playwright/browser needed)
- Gateway API is NOT behind WAF (unlike vbpl.vn portal)
- Outputs individual JSON files per document (R2AI format)
- Robust checkpointing, retry, and rate limiting

Usage:
    python vbpl_gateway_crawler_v2.py
    python vbpl_gateway_crawler_v2.py --max-docs 10  # test run
    python vbpl_gateway_crawler_v2.py --discovery all_central_documents
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests

# Fix Windows console encoding for Vietnamese text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

PORTAL_ROOT = "https://vbpl.vn"
GATEWAY_ROOT = "https://vbpl-bientap-gateway.moj.gov.vn"
GATEWAY_DETAIL_TEMPLATE = f"{GATEWAY_ROOT}/api/qtdc/public/doc/{{item_id}}"
ROBOTS_URL = f"{PORTAL_ROOT}/robots.txt"
SITEMAP_URL = f"{PORTAL_ROOT}/sitemap.xml"

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_ROOT = DATA_ROOT / "raw" / "vbpl_gateway"

# Output directories (R2AI-compatible)
JSON_DETAIL_DIR = OUTPUT_ROOT / "json"          # individual {id}.json files
FILTERED_DIR = OUTPUT_ROOT / "filtered"          # filtered index
FILTERED_JSON_DIR = FILTERED_DIR / "json"        # filtered detail JSONs

# Crawl state
CHECKPOINT = OUTPUT_ROOT / "checkpoint_v2.json"
MASTER_INDEX = OUTPUT_ROOT / "crawl_index.jsonl"

CONTACT_EMAIL = "phanlacontrai@gmail.com"
USER_AGENT = f"VietnameseLegalCorpusResearch/0.2 ({CONTACT_EMAIL})"

REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0  # seconds between retries

FOCUS_KEYWORDS = (
    "dân sự", "đất đai", "dân số", "gia đình", "hôn nhân", "trẻ em", "bình đẳng giới",
    "xây dựng", "nhà ở", "đô thị", "tố tụng", "tranh chấp",
    "bổ trợ tư pháp", "luật sư", "công chứng", "hộ tịch", "quốc tịch",
    "lý lịch tư pháp", "nuôi con nuôi", "chứng thực", "thi hành án",
    "dan-su", "dat-dai", "dan-so", "gia-dinh", "hon-nhan", "tre-em", "binh-dang-gioi",
    "xay-dung", "nha-o", "do-thi", "to-tung", "tranh-chap",
    "bo-tro-tu-phap", "luat-su", "cong-chung", "ho-tich", "quoc-tich",
    "ly-lich-tu-phap", "nuoi-con-nuoi", "chung-thuc", "thi-hanh-an",
)


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_item_id(url: str) -> str:
    """Extract the item ID from a vbpl.vn detail URL.
    
    Handles both numeric IDs (--122250) and UUID IDs
    (--be2a9370-b0b0-11f1-a0a7-318a060316a1).
    """
    return url.rsplit("/", 1)[-1].rsplit("--", 1)[-1]


def raw_record_is_valid(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(record, dict)
        and record.get("doc_id") == path.stem
        and isinstance(record.get("thuoc_tinh"), dict)
        and isinstance(record.get("noi_dung"), dict)
    )


# ──────────────────────────────────────────────────────────────
# Gateway API → R2AI Format Transformer
# ──────────────────────────────────────────────────────────────

def gateway_to_r2ai(item_id: str, gateway_data: dict) -> dict:
    """Transform gateway API response to R2AI-compatible format.
    
    R2AI format:
    {
        "doc_id": "163595",
        "thuoc_tinh": { ... metadata ... },
        "noi_dung": { "id": "...", "title": "...", "content_html": "..." }
    }
    """
    doc_type = gateway_data.get("docType") or {}
    eff_status = gateway_data.get("effStatus") or {}
    organization = gateway_data.get("organization") or {}
    doc_content = gateway_data.get("documentContent") or {}

    # Build PDF/DOC URLs
    pdf_filename = gateway_data.get("documentContentFileName")
    doc_filename = gateway_data.get("documentContentFileDocName")
    url_pdf = f"{PORTAL_ROOT}/file/{pdf_filename}" if pdf_filename else None
    url_doc = f"{PORTAL_ROOT}/file/{doc_filename}" if doc_filename else None

    thuoc_tinh = {
        "id": str(gateway_data.get("id", item_id)),
        "so_hieu": gateway_data.get("docNum"),
        "ten_van_ban": gateway_data.get("title"),
        "loai_van_ban": doc_type.get("name"),
        "loai_code": doc_type.get("code"),
        "ngay_ban_hanh": gateway_data.get("issueDate"),
        "hieu_luc_tu": gateway_data.get("effFrom"),
        "hieu_luc_den": gateway_data.get("effTo"),
        "trang_thai_hieu_luc": eff_status.get("name"),
        "trang_thai_code": eff_status.get("code"),
        "co_quan_ban_hanh": gateway_data.get("agencyName"),
        "bo_nganh": organization.get("name"),
        "ngon_ngu": gateway_data.get("lang"),
        "luot_xem": gateway_data.get("viewCount"),
        "ngay_cap_nhat": gateway_data.get("updatedDate"),
        "trang_thai_xuat_ban": gateway_data.get("status"),
        "is_luat": gateway_data.get("isLw"),
        "is_cu": gateway_data.get("isOld"),
        "has_content": gateway_data.get("hasContent"),
        "file_pdf": pdf_filename,
        "file_doc": doc_filename,
        "url_chi_tiet": f"{PORTAL_ROOT}/van-ban/chi-tiet/--{item_id}",
        "url_pdf": url_pdf,
        "url_doc": url_doc,
    }

    noi_dung = {
        "id": str(doc_content.get("id", item_id)),
        "title": doc_content.get("title"),
        "content_html": doc_content.get("content"),
    }

    return {
        "doc_id": str(item_id),
        "thuoc_tinh": thuoc_tinh,
        "noi_dung": noi_dung,
        # Extra fields not in R2AI but useful
        "_extra": {
            "document_fields": gateway_data.get("documentFields", []),
            "document_majors": gateway_data.get("documentMajors", []),
            "document_issues": gateway_data.get("documentIssues", []),
            "provision_tree": gateway_data.get("provisionTree"),
            "references": gateway_data.get("references"),
            "fetched_at_utc": utc_now(),
        }
    }


# ──────────────────────────────────────────────────────────────
# Preflight Checks
# ──────────────────────────────────────────────────────────────

def robots_allows_public_page(robots_text: str, url: str) -> bool:
    disallowed = []
    active = False
    for raw_line in robots_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().casefold(), value.strip()
        if key == "user-agent":
            active = value == "*"
        elif active and key == "disallow" and value:
            disallowed.append(value)
    path = urlparse(url).path
    return not any(path.startswith(prefix) for prefix in disallowed)


def preflight(session: requests.Session) -> str:
    """Run preflight checks. Returns robots.txt content."""
    if "@example.com" in CONTACT_EMAIL or "@example.invalid" in CONTACT_EMAIL:
        raise ValueError("Set CONTACT_EMAIL before crawling.")
    if REQUEST_DELAY_SECONDS < 1.0:
        raise ValueError("REQUEST_DELAY_SECONDS must be at least 1 second.")

    robots_response = session.get(ROBOTS_URL, timeout=30)
    robots_response.raise_for_status()
    robots_text = robots_response.text

    if not robots_allows_public_page(robots_text, f"{PORTAL_ROOT}/van-ban/chi-tiet/example"):
        raise PermissionError("robots.txt does not allow public detail pages.")

    print(f"✓ robots.txt OK — public detail pages allowed")
    print(f"✓ Gateway API endpoint: {GATEWAY_ROOT}")
    print(f"  (Gateway is on separate domain, not restricted by portal robots.txt)")
    return robots_text


# ──────────────────────────────────────────────────────────────
# URL Discovery via Sitemap
# ──────────────────────────────────────────────────────────────

def sitemap_locations(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [node.text for node in root.findall(".//sm:loc", namespace) if node.text]


def is_local_sitemap(session: requests.Session, urls: list[str]) -> bool:
    """Check the first document's issuer before crossing into provincial sitemaps."""
    first_detail = next(
        (url for url in urls if url.startswith(f"{PORTAL_ROOT}/van-ban/chi-tiet/")),
        None,
    )
    if first_detail is None:
        raise ValueError("Cannot classify sitemap without a document detail URL")
    item_id = extract_item_id(first_detail)
    response = session.get(
        GATEWAY_DETAIL_TEMPLATE.format(item_id=item_id),
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("success") is not True or not isinstance(payload.get("data"), dict):
        raise ValueError(f"Cannot classify sitemap from document {item_id}")
    agency = (payload["data"].get("agencyName") or "").casefold()
    if not agency:
        raise ValueError(f"Missing issuer for sitemap sample {item_id}")
    return agency.startswith(("ubnd", "ủy ban nhân dân", "uỷ ban nhân dân"))


def discover_urls(session: requests.Session, discovery_mode: str) -> list[dict]:
    """Discover document URLs from vbpl.vn sitemaps."""
    print("Discovering URLs from sitemap...")
    root_response = session.get(SITEMAP_URL, timeout=30)
    root_response.raise_for_status()
    sitemap_urls = sitemap_locations(root_response.text)
    rows = []
    seen = set()
    for sitemap_url in sitemap_urls:
        if not sitemap_url.startswith(f"{PORTAL_ROOT}/sitemap/"):
            continue
        time.sleep(REQUEST_DELAY_SECONDS)
        print(f"  Fetching {sitemap_url}...")
        response = session.get(sitemap_url, timeout=60)
        response.raise_for_status()
        urls = sitemap_locations(response.text)
        if not urls or not any(
            url.startswith(f"{PORTAL_ROOT}/van-ban/chi-tiet/") for url in urls
        ):
            continue
        if is_local_sitemap(session, urls):
            print(f"  Provincial documents start at {sitemap_url}; stopping central discovery")
            break
        for url in urls:
            if url in seen or not url.startswith(f"{PORTAL_ROOT}/van-ban/chi-tiet/"):
                continue
            hints = [kw for kw in FOCUS_KEYWORDS if kw in url.casefold()]
            rows.append({"url": url, "topic_hints": hints, "sitemap_url": sitemap_url})
            seen.add(url)

    return rows


def load_or_discover_candidates(
    session: requests.Session, discovery_mode: str, refresh: bool = False
) -> list[dict]:
    """Load cached candidates or discover from sitemap."""
    cache_path = OUTPUT_ROOT / f"candidate_urls_{discovery_mode}.json"
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        candidates = cached.get("candidates", []) if isinstance(cached, dict) else []
        if isinstance(cached, dict) and cached.get("discovery_mode") != discovery_mode:
            candidates = []
        if candidates:
            print(f"✓ Loaded {len(candidates)} cached candidate URLs")
            return candidates

    candidates = discover_urls(session, discovery_mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"discovery_mode": discovery_mode, "created_at_utc": utc_now(), "candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✓ Discovered and cached {len(candidates)} candidate URLs")
    return candidates


# ──────────────────────────────────────────────────────────────
# Checkpoint Management
# ──────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {"completed_ids": [], "failed_ids": {}, "status": "started"}


def save_checkpoint(state: dict):
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(CHECKPOINT)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.25 * (attempt + 1), 1.0))


# ──────────────────────────────────────────────────────────────
# Single Document Crawler
# ──────────────────────────────────────────────────────────────

def crawl_one(session: requests.Session, item_id: str) -> dict:
    """Fetch a single document from the gateway API.
    
    Returns R2AI-compatible dict.
    Raises on failure after retries.
    """
    api_url = GATEWAY_DETAIL_TEMPLATE.format(item_id=item_id)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(api_url, timeout=REQUEST_TIMEOUT_SECONDS)

            if response.status_code == 429:
                last_error = requests.HTTPError(
                    f"Gateway rate limited document {item_id} (HTTP 429)"
                )
                wait = RETRY_BACKOFF * attempt * 2
                print(f"    Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            payload = response.json()

            if payload.get("success") is not True:
                raise ValueError(
                    f"Gateway returned success=False for {item_id}: "
                    f"{payload.get('message', 'unknown error')}"
                )

            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError(f"Gateway returned no data dict for {item_id}")

            return gateway_to_r2ai(item_id, data)

        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"    Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    raise last_error


# ──────────────────────────────────────────────────────────────
# Index Record (lightweight summary)
# ──────────────────────────────────────────────────────────────

def make_index_record(doc: dict, topic_hints: list) -> dict:
    """Create a lightweight index record from an R2AI document."""
    tp = doc.get("thuoc_tinh", {})
    nd = doc.get("noi_dung", {})
    return {
        "doc_id": doc["doc_id"],
        "so_hieu": tp.get("so_hieu"),
        "ten_van_ban": tp.get("ten_van_ban"),
        "loai_van_ban": tp.get("loai_van_ban"),
        "co_quan_ban_hanh": tp.get("co_quan_ban_hanh"),
        "ngay_ban_hanh": tp.get("ngay_ban_hanh"),
        "trang_thai_hieu_luc": tp.get("trang_thai_hieu_luc"),
        "has_content": bool(nd.get("content_html")),
        "content_length": len(nd.get("content_html") or ""),
        "topic_hints": topic_hints,
        "url_chi_tiet": tp.get("url_chi_tiet"),
    }


# ──────────────────────────────────────────────────────────────
# Main Crawl Loop
# ──────────────────────────────────────────────────────────────

def run_crawl(
    max_docs: int | None = None,
    discovery_mode: str = "all_central_documents",
    refresh_candidates: bool = False,
    discover_only: bool = False,
    retry_failed: bool = False,
):
    """Main entry point for the crawl."""
    print("=" * 60)
    print("VBPL Gateway Direct API Crawler v2")
    print("=" * 60)

    # Setup session
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "vi,en;q=0.8",
    })

    # Preflight
    preflight(session)

    # Discover URLs
    if retry_failed:
        failed_ids = load_checkpoint()["failed_ids"]
        # Recover malformed-URL failures before retrying persistent Gateway 500s.
        ordered_ids = sorted(failed_ids, key=lambda item_id: ("500" in failed_ids[item_id], item_id))
        candidates = [
            {"url": f"{PORTAL_ROOT}/van-ban/chi-tiet/--{extract_item_id(item_id)}", "topic_hints": []}
            for item_id in ordered_ids
        ]
    else:
        candidates = load_or_discover_candidates(session, discovery_mode, refresh_candidates)

    if discover_only:
        print(f"Discovered {len(candidates)} candidates; no documents fetched")
        return

    # Create output directories
    JSON_DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    state = load_checkpoint()
    if retry_failed:
        # Older runs stored the whole URL for the few sitemap entries without "--".
        state["failed_ids"] = {
            extract_item_id(item_id): error
            for item_id, error in state["failed_ids"].items()
        }
    completed_ids = set(state["completed_ids"])

    # Only complete, parseable raw records count as completed. A crash during
    # write must leave the item eligible for retry on the next run.
    files_to_check = (
        (JSON_DETAIL_DIR / f"{item_id}.json" for item_id in state["failed_ids"])
        if retry_failed else JSON_DETAIL_DIR.glob("*.json")
    )
    for json_file in files_to_check:
        if not json_file.exists():
            continue
        if raw_record_is_valid(json_file):
            completed_ids.add(json_file.stem)
        else:
            print(f"  Incomplete raw file will be retried: {json_file.name}")

    # Build pending list
    pending = []
    for candidate in candidates:
        item_id = extract_item_id(candidate["url"])
        if item_id not in completed_ids:
            pending.append({**candidate, "item_id": item_id})

    if max_docs is not None:
        pending = pending[:max_docs]
        print(f"  Limited this run to the next {len(pending)} pending documents")

    print(f"\n📊 Status: {len(completed_ids)} completed, {len(pending)} pending")

    if not pending:
        print("✓ No pending candidates")
        state["status"] = "partial" if state["failed_ids"] else "completed"
        state["completed_at_utc"] = utc_now()
        save_checkpoint(state)
        return

    # Crawl loop
    success_count = 0
    fail_count = 0

    with MASTER_INDEX.open("a", encoding="utf-8") as index_file:
        for i, candidate in enumerate(pending, start=1):
            item_id = candidate["item_id"]
            try:
                doc = crawl_one(session, item_id)

                # Save individual JSON (R2AI format)
                doc_path = JSON_DETAIL_DIR / f"{item_id}.json"
                doc_path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # Append to master index
                idx = make_index_record(doc, candidate.get("topic_hints", []))
                index_file.write(json.dumps(idx, ensure_ascii=False) + "\n")
                index_file.flush()

                # Update checkpoint
                state["completed_ids"].append(item_id)
                state["failed_ids"].pop(item_id, None)
                save_checkpoint(state)

                success_count += 1
                title_short = (doc["thuoc_tinh"].get("ten_van_ban") or "")[:60]
                has_content = "✓" if doc["noi_dung"].get("content_html") else "✗"
                print(f"  [{i}/{len(pending)}] {item_id} [{has_content}] {title_short}")

            except Exception as e:
                fail_count += 1
                state["failed_ids"][item_id] = repr(e)
                save_checkpoint(state)
                print(f"  [{i}/{len(pending)}] ✗ FAILED {item_id}: {e}")
                # Don't raise — continue with next document

            # Rate limit
            time.sleep(REQUEST_DELAY_SECONDS)

    # Final checkpoint
    if fail_count == 0 and not state["failed_ids"]:
        state["status"] = "completed"
    else:
        state["status"] = "partial"
    state["completed_at_utc"] = utc_now()
    state["stats"] = {
        "total_candidates": len(candidates),
        "success": success_count,
        "failed": fail_count,
        "previously_completed": len(completed_ids),
    }
    save_checkpoint(state)

    print(f"\n{'=' * 60}")
    print(f"✓ Done! {success_count} succeeded, {fail_count} failed")
    print(f"  JSON files saved to: {JSON_DETAIL_DIR}")
    print(f"  Index: {MASTER_INDEX}")
    print(f"  Checkpoint: {CHECKPOINT}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VBPL Gateway Direct API Crawler v2")
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Maximum number of documents to crawl (default: all)",
    )
    parser.add_argument(
        "--discovery", choices=["focus_slug", "all_central_documents"],
        default="all_central_documents",
        help="URL discovery mode (default: all_central_documents; both modes now use full sitemap discovery)",
    )
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    run_crawl(
        max_docs=args.max_docs,
        discovery_mode=args.discovery,
        refresh_candidates=args.refresh_candidates,
        discover_only=args.discover_only,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
