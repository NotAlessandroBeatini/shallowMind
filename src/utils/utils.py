# src/utils/utils.py (or wherever setup_colored_logging lives)
import re
import logging
import sys
import os
import datetime
from pathlib import Path

def get_cwd():
    #dynamically get base dir of project
    CWD_PATH = Path(os.getcwd())
    CWD_PATH = str(CWD_PATH)
    CWD_PATH = re.sub(r"^.*(?=/home)", "/davinci-1", CWD_PATH)

# --- Define Project Root (reuse logic from train.py/data_manager.py if possible) ---
# This is important for finding the 'logs' directory consistently.
try:
    # Adjust parents[...] based on utils.py's location relative to root
    # If utils.py is in src/utils, root is parents[2]
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
     # Fallback if __file__ not defined
     PROJECT_ROOT = Path(os.getcwd())
# --------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds ANSI color codes to log messages
    based on the log level for console output.
    (Your existing ColoredFormatter code here - unchanged)
    """
    # ... (Your existing ColoredFormatter code) ...
    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    LEVEL_COLORS = {
        logging.DEBUG: CYAN + BOLD,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW + BOLD,
        logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }
    def __init__(self, fmt="%(levelname)s: %(message)s", datefmt=None, style='%', use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color

    def format(self, record):
        if self.use_color:
            color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        else:
            color = "" # No color for file logging
            reset = ""
        formatted_message = super().format(record)
        if self.use_color:
             return f"{color}{formatted_message}{self.RESET}"
        else:
             return formatted_message


# --- Enhanced Setup Function ---
def setup_logging(
    level=logging.INFO,
    log_file_prefix: str = "app", # Default prefix if none provided by caller
    log_dir: str = "logs", # Directory relative to project root
    console_logging: bool = True,
    file_logging: bool = False,
    timestamp_in_logfile_name: bool = True # Option to add timestamp
    ):
    """
    Configures the root logger for console (optional colored) and file logging.

    Args:
        level: The minimum logging level (e.g., logging.INFO, logging.DEBUG).
        log_file_prefix: A prefix for the log file (e.g., 'data_manager', 'train').
                         Often passed as __name__ from the calling module.
        log_dir: The directory (relative to project root) to save log files.
        console_logging: Whether to enable console logging.
        file_logging: Whether to enable file logging.
        timestamp_in_logfile_name: If True, appends a timestamp to the log filename.
    """
    logger = logging.getLogger() # Get root logger
    logger.setLevel(level)

    # --- Remove existing handlers (important!) ---
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            # Check if handler has close method
            if callable(getattr(handler, "close", None)):
                try:
                    handler.close()
                except Exception as e:
                    print(f"Error closing handler {handler}: {e}", file=sys.stderr)
            logger.removeHandler(handler)
    # ---------------------------------------------

    # Define base log format (used by both handlers)
    log_format = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers_added = []

    # --- Configure Console Handler ---
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout) # Use stdout for console
        console_handler.setLevel(level)
        # Use ColoredFormatter for the console
        console_formatter = ColoredFormatter(fmt=log_format, datefmt=date_format, use_color=True)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        handlers_added.append("Console")

    # --- Configure File Handler ---
    if file_logging:
        log_directory = PROJECT_ROOT / log_dir
        try:
            log_directory.mkdir(parents=True, exist_ok=True)

            # Clean prefix (replace dots common in __name__)
            safe_prefix = log_file_prefix.replace('.', '_')

            if timestamp_in_logfile_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"{safe_prefix}_{timestamp}.log"
            else:
                log_filename = f"{safe_prefix}.log" # Overwrites previous log

            log_filepath = log_directory / log_filename

            file_handler = logging.FileHandler(log_filepath, mode='a') # Append mode
            file_handler.setLevel(level)
            # Use standard Formatter (non-colored) for the file
            file_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
            # Alternatively, use ColoredFormatter with use_color=False
            # file_formatter = ColoredFormatter(fmt=log_format, datefmt=date_format, use_color=False)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            handlers_added.append(f"File ({log_filepath})")

        except Exception as e:
            # Log error if file handler setup fails, but continue if console is setup
            logger.error(f"Failed to configure file logging to {log_dir}/{log_filename}: {e}", exc_info=True)

    # --- Optional: Adjust library levels ---
    if level <= logging.INFO: # Only adjust if overall level is INFO or lower
         logging.getLogger("transformers").setLevel(logging.WARNING)
         logging.getLogger("datasets").setLevel(logging.WARNING)
         logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
         logging.getLogger("h5py").setLevel(logging.WARNING)
         logging.getLogger("torch.distributed").setLevel(logging.INFO)
         logging.getLogger("torch.nn.parallel").setLevel(logging.INFO)
         # Add more as needed
    # -------------------------------------

    if handlers_added:
        logger.info(f"Logging configured successfully. Handlers added: {', '.join(handlers_added)}")
    else:
        # This case should ideally not happen if called with default args
        # but good to handle. basicConfig might be an alternative here.
        print("Warning: Logging setup called but no handlers were configured.", file=sys.stderr)
        logging.basicConfig(level=level) # Fallback to basic config if no handlers
        logger.warning("Fell back to basicConfig for logging.")


# --- Example Usage (in another file like data_manager.py or train.py) ---
# import logging
# from src.utils.utils import setup_logging # Adjust import path

# if __name__ == "__main__":
#     # Call setup_logging early, passing the module's name
#     setup_logging(level=logging.INFO, log_file_prefix=__name__, timestamp_in_logfile_name=False) # Overwrite log each run

#     logger = logging.getLogger(__name__) # Get logger specific to this module

#     logger.info("This message goes to console (colored) and data_manager.log")
#     logger.warning("This is a warning.")
#     logger.debug("This message won't show if level is INFO.")

#     try:
#         # ... your code ...
#         logger.info("DataModule test started.")
#         # ...
#         # x = 1 / 0 # Example error
#     except Exception as e:
#         logger.error("An error occurred during the test.", exc_info=True) # Log traceback to file/console

