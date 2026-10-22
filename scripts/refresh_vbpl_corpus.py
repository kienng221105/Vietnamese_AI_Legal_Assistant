"""Resume the VBPL crawl, then rebuild the full-article corpus."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "raw" / "vbpl_gateway" / "pipeline_status.json"
CHECKPOINT_PATH = ROOT / "data" / "raw" / "vbpl_gateway" / "checkpoint_v2.json"
CANDIDATES_PATH = ROOT / "data" / "raw" / "vbpl_gateway" / "candidate_urls_all_central_documents.json"
RAW_DIR = ROOT / "data" / "raw" / "vbpl_gateway" / "json"
CRAWLER = ROOT / "notebooks" / "vbpl_gateway_crawler_v2.py"
POST_CRAWL_STAGES = (
    ("split", ROOT / "scripts" / "split_vbpl_records.py", []),
    ("preprocess_full", ROOT / "scripts" / "preprocess_vbpl_corpus.py", ["--full-only"]),
    ("build_full", ROOT / "scripts" / "build_full_corpus.py", []),
)
BATCH_SIZE = 5_000


def save_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temp.replace(STATUS_PATH)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.25 * (attempt + 1), 1.0))


def main() -> None:
    status = {
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": BATCH_SIZE,
        "completed_batches": 0,
        "stages": {},
    }
    save_status(status)
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))["candidates"]
    candidate_ids = {row["url"].rsplit("/", 1)[-1].rsplit("--", 1)[-1] for row in candidates}
    final_discovery_done = False

    while True:
        raw_ids = {path.stem for path in RAW_DIR.glob("*.json")}
        pending_ids = candidate_ids - raw_ids
        pending_count = len(pending_ids)
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        failed_ids = set(checkpoint.get("failed_ids", {}))
        status["coverage"] = {
            "candidate_urls": len(candidate_ids),
            "raw_documents": len(raw_ids),
            "missing_candidates": pending_count,
            "failed_gateway_ids": len(failed_ids),
        }
        if not pending_count or pending_ids.issubset(failed_ids):
            if not final_discovery_done:
                print("\n=== refresh central sitemap before completion ===", flush=True)
                status["current_stage"] = "refresh_discovery"
                save_status(status)
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", str(CRAWLER),
                     "--discovery", "all_central_documents", "--refresh-candidates", "--discover-only"],
                    cwd=ROOT,
                    check=False,
                )
                if result.returncode:
                    status["state"] = "failed"
                    save_status(status)
                    raise SystemExit(result.returncode)
                candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))["candidates"]
                candidate_ids = {row["url"].rsplit("/", 1)[-1].rsplit("--", 1)[-1] for row in candidates}
                final_discovery_done = True
                continue
            break

        batch_number = status["completed_batches"] + 1
        limit = min(BATCH_SIZE, pending_count)
        print(f"\n=== crawl batch {batch_number}: up to {limit} pending documents ===", flush=True)
        status["current_stage"] = f"crawl_batch_{batch_number}"
        save_status(status)
        result = subprocess.run(
            [
                sys.executable, "-X", "utf8", str(CRAWLER),
                "--discovery", "all_central_documents", "--max-docs", str(limit),
            ],
            cwd=ROOT,
            check=False,
        )
        status["stages"][f"crawl_batch_{batch_number}"] = {
            "exit_code": result.returncode,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if result.returncode:
            status["state"] = "failed"
            save_status(status)
            raise SystemExit(result.returncode)
        status["completed_batches"] = batch_number
        for name, script, args in POST_CRAWL_STAGES:
            print(f"\n=== {name} after crawl batch {batch_number} ===", flush=True)
            status["current_stage"] = name
            save_status(status)
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(script), *args],
                cwd=ROOT,
                check=False,
            )
            status["stages"][f"{name}_batch_{batch_number}"] = {
                "exit_code": result.returncode,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if result.returncode:
                status["state"] = "failed"
                save_status(status)
                raise SystemExit(result.returncode)
        save_status(status)

    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    raw_ids = {path.stem for path in RAW_DIR.glob("*.json")}
    status["coverage"] = {
        "candidate_urls": len(candidate_ids),
        "raw_documents": len(raw_ids),
        "missing_candidates": len(candidate_ids - raw_ids),
        "failed_gateway_ids": len(checkpoint.get("failed_ids", {})),
    }
    status["state"] = "completed_with_gaps" if status["coverage"]["missing_candidates"] else "completed"
    status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    status.pop("current_stage", None)
    save_status(status)


if __name__ == "__main__":
    main()
