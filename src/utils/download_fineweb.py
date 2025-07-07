#!/usr/bin/env python
"""
download_fineweb.py
-------------------
Mirror the HuggingFaceFW/fineweb parquet files into a shared folder
(without creating symlinks in each user’s home) and resume cleanly
when re-launched.

Usage (interactive):
    python download_fineweb.py \
        --dest /davinci-1/work/abeatini/MORTE_AL_DAVINCI_TEMP

Typical PBS command inside a job script:
    export HF_HUB_DISABLE_SYMLINKS=1
    export HF_HOME="$PWD/.hf_cache"
    python download_fineweb.py --dest "$PWD/data"
"""

import argparse, os, time, logging, requests
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from urllib3.exceptions import MaxRetryError
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mirror FineWeb parquet blobs")
    parser.add_argument(
        "--dest", required=False, default="/davinci-1/home/abeatini/pycharmProjects/shallowMind/data" ,
        help="Destination directory where data/ will be created"
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Number of parallel download connections (default: 32)",
    )
    args = parser.parse_args()

    dest_root = Path(args.dest).resolve()
    #dest_root.mkdir(parents=True, exist_ok=True)
    # Force-on all HF progress bars, even on non-TTY:
    enable_progress_bars()
    # ------------------------------------------------------------------
    # Minimal logging: console + file in the same folder
    log_file = dest_root / "logs/fineweb_download.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )
    logging.info("Starting FineWeb mirror → %s", dest_root)

    # ------------------------------------------------------------------
    # Required env-vars so everyone writes in shared space
    #os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    #os.environ.setdefault("HF_HOME", str(dest_root / ".hf_cache"))
    PROJ = Path("/davinci-1/home/abeatini/pycharmProjects/shallowMind").expanduser().resolve()
    BASE = PROJ / "data" / "main_cache"
    RAW  = BASE / "raw"  / "HuggingFaceFW_fineweb"    # mirror goes here
    
    # ─── Mirror with retry ─────────────────────────────────────────────────────
    MAX_RETRIES = 5
    RETRY_DELAY = 30   # seconds
    t0 = time.time()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id="HuggingFaceFW/fineweb",
                repo_type="dataset",
                local_dir=RAW,
                allow_patterns="data/*.parquet",
                resume_download=True,
                max_workers=args.workers,
            )
            elapsed = (time.time() - t0) / 60
            logging.info("✅ Finished mirror in %.1f min", elapsed)
            break

        except Exception as e:
            if attempt < MAX_RETRIES:
                logging.warning(
                    "🌐 Network error on attempt %d/%d: %s",
                    attempt, MAX_RETRIES, e
                )
                time.sleep(RETRY_DELAY)
            else:
                logging.error(
                    "❌ Mirror failed after %d attempts: %s",
                    attempt, e, exc_info=True
                )
                raise

if __name__ == "__main__":
    main()
