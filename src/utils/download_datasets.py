# src/utils/download_datasets.py

import logging
import sys
import os
import time
import argparse
import re # Import regular expressions for log file matching
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union, Optional # Keep Union for Py 3.10

from datasets import load_dataset, DownloadMode
try:
    from huggingface_hub import repo_info
    from huggingface_hub.utils import HfHubHTTPError
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    repo_info = None
    HfHubHTTPError = Exception

# --- Colored Logging Setup (Same as before) ---
class ColoredFormatter(logging.Formatter):
    RESET = "\033[0m"; RED = "\033[31m"; YELLOW = "\033[33m"; CYAN = "\033[36m"; BOLD = "\033[1m"
    LEVEL_COLORS = { logging.DEBUG: CYAN + BOLD, logging.INFO: CYAN, logging.WARNING: YELLOW + BOLD, logging.ERROR: RED, logging.CRITICAL: RED + BOLD, }
    def __init__(self, fmt="%(levelname)s: %(message)s", datefmt=None, style='%'):
        fmt = "%(asctime)s - PID:%(process)d - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        formatted_message = super().format(record)
        return f"{color}{formatted_message}{self.RESET}"

# === UPDATED setup_colored_logging function ===
def setup_colored_logging(level=logging.INFO, log_dir: Union[str, Path, None] = None, log_filename_base: str = "download_datasets"):
    """
    Configures the root logger for console (colored) and incremental file output.
    Removes default handlers to prevent duplicate messages.
    Finds the next available log file index (log_0.txt, log_1.txt, etc.).
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # --- Remove existing handlers ---
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            try: handler.close()
            except Exception: pass
            logger.removeHandler(handler)

    # --- Console Handler (Colored) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_log_format = "%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s"
    console_date_format = "%Y-%m-%d %H:%M:%S"
    console_formatter = ColoredFormatter(fmt=console_log_format, datefmt=console_date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler (Incremental Naming) ---
    log_file_path = None
    if log_dir:
        try:
            log_directory = Path(log_dir)
            log_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            # Find the next available log file index
            log_index = 0
            log_file_pattern = re.compile(rf"{re.escape(log_filename_base)}_(\d+)\.log")
            existing_indices = []
            for item in log_directory.iterdir():
                if item.is_file():
                    match = log_file_pattern.match(item.name)
                    if match:
                        existing_indices.append(int(match.group(1)))

            if existing_indices:
                log_index = max(existing_indices) + 1

            log_file_path = log_directory / f"{log_filename_base}_{log_index}.log"

            # Configure file handler
            file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
            file_handler.setLevel(level)
            file_log_format = "%(asctime)s - PID:%(process)d - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
            file_date_format = "%Y-%m-%d %H:%M:%S"
            file_formatter = logging.Formatter(fmt=file_log_format, datefmt=file_date_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            logger.info(f"Logging configured successfully. Console level: {logging.getLevelName(level)}, File output: {log_file_path}")

        except Exception as e:
            logger.error(f"Failed to configure file logging to directory {log_dir}: {e}", exc_info=True)
            log_file_path = None # Ensure it's None if setup failed
    else:
        logger.info(f"Logging configured successfully. Console level: {logging.getLevelName(level)}. File output disabled.")

    # --- Optional: Adjust library logging levels ---
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Return the actual log file path used (or None)
    return log_file_path
# === END UPDATED function ===


# === Configuration: Define Datasets to Download (Same as before, ensure trust_remote_code is correct) ===
DATASETS_TO_DOWNLOAD = [
    {"name": "refinedweb", "config": "default", "splits": ["train"], "trust_remote_code": True},
    {"name": "cerebras/SlimPajama-627B", "config": "default", "splits": ["train"], "trust_remote_code": False},
    {"name": "openwebtext", "config": None, "splits": ["train"], "trust_remote_code": True},
    {"name": "wikitext", "config": "wikitext-103-raw-v1", "splits": ["train", "validation", "test"], "trust_remote_code": False},
    {"name": "bookcorpus", "config": None, "splits": ["train"], "trust_remote_code": True},
    {"name": "oscar", "config": "unshuffled_deduplicated_en", "splits": ["train"], "trust_remote_code": True},
]
# =====================================================

# --- determine_project_root, format_size, get_dataset_size_estimate (Same as before) ---
def determine_project_root(script_path: Path, root_marker: str = '.git') -> Path:
    current_dir = script_path.resolve().parent
    project_root_guess = current_dir.parent.parent
    logging.debug(f"Initial project root guess based on script location: {project_root_guess}")
    check_dir = current_dir
    while check_dir != check_dir.parent:
        if (check_dir / root_marker).exists():
            logging.debug(f"Found root marker '{root_marker}' at: {check_dir}")
            return check_dir
        check_dir = check_dir.parent
    logging.warning(f"Could not find root marker '{root_marker}'. Using directory structure guess for project root: {project_root_guess}")
    return project_root_guess

def format_size(num_bytes: Union[int, None]) -> str:
    if num_bytes is None or num_bytes < 0: return "N/A"
    if num_bytes < 1024.0: return f"{num_bytes} Bytes"
    num_bytes /= 1024.0; unit_i = 0
    units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    while abs(num_bytes) >= 1024.0 and unit_i < len(units) - 1:
        num_bytes /= 1024.0; unit_i += 1
    return f"{num_bytes:3.1f} {units[unit_i]}"

def get_dataset_size_estimate(dataset_name: str, config_name: Union[str, None] = None) -> Union[int, None]:
    if not HAS_HF_HUB or repo_info is None: return None
    try:
        info = repo_info(dataset_name, repo_type="dataset")
        size_bytes = getattr(info, 'sizeOnDisk', None)
        if size_bytes is not None and size_bytes > 0: return size_bytes
        total_size = sum(f.size for f in getattr(info, 'siblings', []) if f.size is not None)
        return total_size if total_size > 0 else None
    except HfHubHTTPError as e:
        if e.response.status_code == 404: logging.warning(f"Dataset '{dataset_name}' not found on Hub.")
        else: logging.warning(f"HTTP error fetching info for '{dataset_name}': {e}.")
        return None
    except Exception as e: logging.warning(f"Could not estimate size for '{dataset_name}': {e}"); return None

# --- download_dataset_worker (Same as before) ---
def download_dataset_worker(dataset_info: dict, raw_cache_dir: str, download_mode: DownloadMode) -> tuple[str, bool]:
    name = dataset_info["name"]; config = dataset_info.get("config")
    splits = dataset_info.get("splits", ["train"]); trust_code = dataset_info.get("trust_remote_code", False)
    dataset_id = f"{name}" + (f" (config: {config})" if config else "")
    success = True; start_time = time.time(); worker_logger = logging.getLogger()
    worker_logger.info(f"Starting download task for {dataset_id} (trust_remote_code={trust_code})...")
    load_args = [name];
    if config: load_args.append(config)
    for split in splits:
        try:
            worker_logger.info(f"Attempting to download {dataset_id} split '{split}'...")
            load_dataset(*load_args, split=split, cache_dir=raw_cache_dir, trust_remote_code=trust_code, download_mode=download_mode)
            worker_logger.info(f"Successfully downloaded/verified {dataset_id} split '{split}'.")
        except Exception as e:
            worker_logger.error(f"FAILED to download {dataset_id} split '{split}': {type(e).__name__} - {e}", exc_info=False)
            if "requires you to execute the dataset script" in str(e) and not trust_code:
                 worker_logger.error(f"--> Hint: Dataset '{name}' likely requires 'trust_remote_code: True' in DATASETS_TO_DOWNLOAD.")
            success = False
    elapsed_time = time.time() - start_time; status_msg = "COMPLETED" if success else "FAILED"
    worker_logger.info(f"{status_msg} download process for {dataset_id} in {elapsed_time:.2f}s")
    return dataset_id, success


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets in parallel.")
    # --- Update help message for base-cache-dir ---
    parser.add_argument(
        "--base-cache-dir", type=str, default="data/main_cache",
        help="Path to the BASE cache directory relative to project root (e.g., data/main_cache). Datasets are stored in <base-cache-dir>/raw."
    )
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2), help="Number of parallel download processes.")
    parser.add_argument("--force-redownload", action="store_true", help="Force redownload even if dataset exists in cache.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--project-root", type=Union[str, None], default=None, help="Optional: Explicitly specify the project root directory.")
    # --- Argument for LOG DIRECTORY instead of specific file ---
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/logs", # Default relative log directory
        help="Directory to store output log files (relative to project root or absolute). Set to 'none' or empty string to disable file logging."
    )

    args = parser.parse_args()

    # --- Determine Project Root ---
    if args.project_root:
        PROJECT_ROOT = Path(args.project_root).resolve()
        if not PROJECT_ROOT.is_dir(): logging.critical(f"Provided project root '{args.project_root}' is not valid. Exiting."); sys.exit(1)
    else:
        try: PROJECT_ROOT = determine_project_root(Path(__file__))
        except Exception as e: logging.critical(f"Error detecting project root: {e}. Use --project-root."); sys.exit(1)
    logging.info(f"Using Project Root: {PROJECT_ROOT}") # Log root after logging is set up

    # --- Resolve Log Directory ---
    log_directory = None
    if args.log_dir and args.log_dir.lower() != 'none':
        log_dir_path = Path(args.log_dir)
        if not log_dir_path.is_absolute():
            log_directory = PROJECT_ROOT / log_dir_path
        else:
            log_directory = log_dir_path
        # Directory creation happens inside setup_colored_logging

    # --- Setup logging (pass resolved log directory) ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    actual_log_file = setup_colored_logging(level=log_level, log_dir=log_directory, log_filename_base="download_datasets")

    # --- Determine Cache Directory ---
    base_cache_path = Path(args.base_cache_dir)
    if not base_cache_path.is_absolute(): base_cache_path = PROJECT_ROOT / base_cache_path
    RAW_CACHE_DIR = base_cache_path / "raw"
    logging.info(f"Base Cache Directory: {base_cache_path}")
    logging.info(f"Target Raw Cache Directory: {RAW_CACHE_DIR}")
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Estimate Size (Same as before) ---
    total_estimated_size = 0; unknown_size_count = 0
    logging.info("-" * 30); logging.info("Estimating dataset sizes...")
    if not HAS_HF_HUB: logging.warning("`huggingface_hub` not found. Cannot estimate dataset sizes.")
    else:
        for ds_info in DATASETS_TO_DOWNLOAD:
            size_bytes = get_dataset_size_estimate(ds_info["name"], ds_info.get("config")); size_str = format_size(size_bytes)
            logging.info(f"  - {ds_info['name']} (config: {ds_info.get('config', 'N/A')}): {size_str}")
            if size_bytes is not None: total_estimated_size += size_bytes
            else: unknown_size_count += 1
        logging.info("-" * 30); logging.info(f"Total Estimated Size: {format_size(total_estimated_size)}")
        if unknown_size_count > 0: logging.warning(f"Could not estimate size for {unknown_size_count} dataset(s).")
        logging.info("Ensure sufficient disk space."); logging.info("-" * 30)

    # --- Start Downloads (Same as before) ---
    download_mode = DownloadMode.FORCE_REDOWNLOAD if args.force_redownload else DownloadMode.REUSE_DATASET_IF_EXISTS
    logging.info(f"Starting parallel download with {args.workers} worker processes.")
    logging.info(f"Download mode: {download_mode.name}")
    start_time_total = time.time(); success_count = 0; failure_count = 0; futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for dataset_info in DATASETS_TO_DOWNLOAD:
            futures.append(executor.submit(download_dataset_worker, dataset_info, str(RAW_CACHE_DIR), download_mode))
        for future in as_completed(futures):
            try:
                dataset_id, success = future.result();
                if success: success_count += 1
                else: failure_count += 1
            except Exception as e: logging.error(f"Exception retrieving result from download worker: {e}", exc_info=True); failure_count += 1

    # --- Final Summary (Same as before) ---
    end_time_total = time.time(); total_elapsed = end_time_total - start_time_total
    final_log_msg = f"Final log file: {actual_log_file}" if actual_log_file else "File logging disabled."
    logging.info("=" * 50); logging.info("Download Summary:")
    logging.info(f"  Target Raw Cache Directory: {RAW_CACHE_DIR}")
    logging.info(f"  {final_log_msg}")
    logging.info(f"  Total datasets attempted: {len(DATASETS_TO_DOWNLOAD)}")
    logging.info(f"  Successfully completed/verified: {success_count}")
    logging.info(f"  Failed downloads: {failure_count}")
    logging.info(f"  Total time elapsed: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    logging.info("=" * 50)
    if failure_count > 0: logging.warning("Some datasets failed to download."); sys.exit(1)
    else: logging.info("All specified datasets downloaded successfully."); sys.exit(0)


if __name__ == "__main__":
    main()


