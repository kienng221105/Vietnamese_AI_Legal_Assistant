"""Retry failed VBPL document IDs and rebuild the full corpus if raw grows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from refresh_vbpl_corpus import POST_CRAWL_STAGES, ROOT


RAW_DIR = ROOT / "data" / "raw" / "vbpl_gateway" / "json"
CRAWLER = ROOT / "notebooks" / "vbpl_gateway_crawler_v2.py"


def run() -> None:
    before = len(list(RAW_DIR.glob("*.json")))
    subprocess.run(
        [sys.executable, "-X", "utf8", str(CRAWLER), "--retry-failed"],
        cwd=ROOT,
        check=True,
    )
    after = len(list(RAW_DIR.glob("*.json")))
    print(f"Recovered raw documents: {after - before}", flush=True)
    if after > before:
        for name, script, args in POST_CRAWL_STAGES:
            print(f"Rebuilding: {name}", flush=True)
            subprocess.run([sys.executable, "-X", "utf8", str(script), *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    run()
