# data_manager.py:
import logging
# Set root logger to DEBUG for your debug messages, INFO for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import re
from pathlib import Path
import time

try:
    # Infer project root assuming this script is in src/data/
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Optional path correction: map /archive/SSD/... to /davinci-1/...
    root_str = str(PROJECT_ROOT)
    if root_str.startswith("/archive/SSD/"):
        corrected_root = re.sub(r"^/archive/SSD/", "/davinci-1/", root_str)
        PROJECT_ROOT = Path(corrected_root)
        logging.warning(f"Rewritten PROJECT_ROOT path to: {PROJECT_ROOT}")

    logging.info(f"Determined Project Root: {PROJECT_ROOT}")
except NameError:
    # __file__ not defined (e.g., running interactively)
    logging.warning("__file__ is not defined, using current working directory as project root.")
    PROJECT_ROOT = Path(os.getcwd())

sys.path.append(str(PROJECT_ROOT)) # Add project root to path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
# Important: Ensures registration happens by importing the module
# where @register_dataset decorators are used.
import src.data.my_datasets
# Explicitly import registry and helper function if needed directly
from src.data.my_datasets import dataset_registry, _tokenize_and_save_split
from src.utils.utils import setup_logging # Assuming you have this utility


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_configs: dict,
        batch_size: int = 8,
        max_length: int = 512,
        num_workers: int = 4,
        cache_dir: str = "data/main_cache" # Relative to project root or absolute
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_configs = dataset_configs
        self.dataset_names = list(dataset_configs.keys())
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        # --- Add Tokenizer ID ---
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', 'unknown_tokenizer')
        if tokenizer_name is None: # Handle case where name_or_path might be None explicitly
            tokenizer_name = 'unknown_tokenizer'
            logging.warning("Tokenizer name_or_path is None, using 'unknown_tokenizer' as ID.")
        self.tokenizer_id = tokenizer_name.replace("/", "__")


        # Resolve cache_dir relative to PROJECT_ROOT if it's not absolute
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = PROJECT_ROOT / self.cache_dir
            logging.info(f"Relative cache_dir provided. Resolved to: {self.cache_dir}")

        self.raw_cache_dir = self.cache_dir / "raw"
        self.tokenized_cache_dir = self.cache_dir / "tokenized" / self.tokenizer_id

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"DataModule initialized.")
        logging.info(f"  Raw cache: {self.raw_cache_dir}")
        logging.info(f"  Tokenized cache base for '{tokenizer_name}': {self.tokenized_cache_dir}")

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                logging.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # Ensure the pad token ID is set correctly if just assigned
                if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # <<< MODIFICATION: Changed exit() to raise ValueError >>>
                msg = "Tokenizer has no pad_token and no eos_token. Cannot set a default pad token!"
                logging.error(msg)
                # Consider adding a default pad token manually if necessary, e.g.,
                # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                raise ValueError(msg)


    def prepare_data(self):
        """
        Downloads raw data and tokenizes it. Runs only on rank 0.
        Failures during tokenization of a specific split will raise an error
        and stop the whole process.
        """
        logging.info(f"--- Starting prepare_data (runs only on rank 0) ---")
        logging.info(f"Using Raw cache directory: {self.raw_cache_dir}")
        logging.info(f"Using Tokenized cache directory (for {self.tokenizer_id}): {self.tokenized_cache_dir}")

        for name in self.dataset_names:
            if name not in dataset_registry:
                # Make failure to find dataset critical
                msg = f"Dataset '{name}' not found in registry. Available: {list(dataset_registry.keys())}."
                logging.error(msg)
                raise KeyError(msg) # Raise an error to stop execution

            dataset_cls = dataset_registry[name]
            specific_config = self.dataset_configs.get(name, {})
            logging.info(f"Preparing dataset: {name} (Class: {dataset_cls.__name__}) with specific config: {specific_config}")

            try:
                splits_to_process = dataset_cls.get_split_names(**specific_config)
            except Exception as e:
                # Make failure to get split names critical
                logging.error(f"CRITICAL: Failed to get split names for {name} using config {specific_config}: {e}. Stopping.", exc_info=True)
                raise # Re-raise the exception to stop execution

            # Process each defined split ('train', 'validation', 'test', etc.)
            for split_label, split_info in splits_to_process.items():
                logging.info(f"Processing split '{split_label}' for {name} (Info: {split_info})")

                # --- Step 1: Ensure Raw Data Split Exists ---
                raw_split_dataset = None
                try:
                    raw_split_dataset = dataset_cls.download_raw_split(
                        split_info=split_info,
                        raw_cache_dir=str(self.raw_cache_dir),
                        trust_remote_code=getattr(dataset_cls, 'trust_remote_code', False),
                        # Add download_mode if needed by your dataset class, e.g.:
                        # download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
                    )
                    if raw_split_dataset is None:
                         # Make None return critical
                         raise ValueError(f"download_raw_split for {name} split '{split_label}' returned None")
                    logging.info(f"Raw data ready for {name} split '{split_label}'. Size: {len(raw_split_dataset)}")

                except Exception as e:
                    # Make failure to download/prepare raw data critical
                    logging.error(f"CRITICAL: Failed to prepare/slice raw data for {name} split '{split_label}'. Stopping. Error: {e}", exc_info=True)
                    raise # Re-raise to stop execution

                # --- Step 2: Generate Tokenized Path & Check/Tokenize ---
                tokenized_split_path = None # Define outside try for error message
                try:
                    tokenized_split_path = self._get_tokenized_split_path(dataset_cls, split_label, split_info)
                    logging.info(f"Target tokenized path: {tokenized_split_path}")

                    if tokenized_split_path.exists():
                        logging.info(f"Tokenized data already exists at {tokenized_split_path}. Skipping tokenization.")
                        continue # Skip tokenization, move to next split
                    else:
                        # raw_split_dataset should be valid here due to checks above
                        logging.info(f"Tokenized data not found. Starting tokenization for {tokenized_split_path.name}...")
                        start_time = time.time()
                        success = _tokenize_and_save_split(
                            raw_split_dataset=raw_split_dataset,
                            tokenized_cache_path=tokenized_split_path,
                            dataset_name=f"{name}_{split_label}",
                            split_label=split_label,
                            tokenizer=self.tokenizer,
                            max_length=self.max_length,
                            text_column=dataset_cls.text_column,
                            num_proc_to_use=self.num_workers # Pass num_workers for multiprocessing
                        )
                        tokenization_time = time.time() - start_time
                        if success:
                           logging.info(f"Tokenization for {tokenized_split_path.name} completed successfully in {tokenization_time:.2f}s.")
                        else:
                           # _tokenize_and_save_split logs errors internally
                           # Make tokenization failure critical
                           msg = f"Tokenization failed for path {tokenized_split_path}. Check logs from _tokenize_and_save_split."
                           logging.error(msg)
                           raise RuntimeError(msg) # Raise error to stop execution

                except Exception as e:
                     # Catch any other error during tokenization/saving step
                     path_info = f"at path '{tokenized_split_path}'" if tokenized_split_path else ""
                     logging.error(f"CRITICAL: Error during tokenization/saving for {name} split '{split_label}' {path_info}: {e}. Stopping.", exc_info=True)
                     raise # Re-raise to stop execution

        logging.info("--- Finished prepare_data ---")


    def setup(self, stage=None):
        """
        Loads pre-tokenized data from disk. Runs on all ranks.
        Failures during loading (e.g., FileNotFoundError) will raise an error.
        """
        logging.info(f"--- Setting up data for stage: {stage} (runs on all ranks) ---")
        # Clear lists at the beginning of setup for the current stage
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        # Keep track of failures per dataset to provide a summary error
        dataset_load_failures = {}

        for name in self.dataset_names:
            if name not in dataset_registry:
                # Should have been caught in prepare_data on rank 0, but check again
                msg = f"Dataset '{name}' not found in registry during setup. Available: {list(dataset_registry.keys())}."
                logging.error(msg)
                raise KeyError(msg) # Raise consistently

            dataset_cls = dataset_registry[name]
            specific_config = self.dataset_configs.get(name, {})
            splits_info = {}

            try:
                splits_info = dataset_cls.get_split_names(**specific_config)
            except Exception as e:
                 logging.error(f"CRITICAL: Failed to get split names for {name} during setup using config {specific_config}: {e}. Cannot load. Stopping.", exc_info=True)
                 raise # Re-raise to stop

            logging.info(f"Setting up dataset: {name} (tokenizer: {self.tokenizer_id}). Expecting splits based on: {splits_info}")

            # Track if any required split fails to load for *this* dataset
            current_dataset_load_failed = False
            failed_splits_for_dataset = []

            # --- Load based on stage ---
            stages_to_setup = []
            if stage in ("fit", None): stages_to_setup.extend(["train", "validation"])
            if stage in ("test", None): stages_to_setup.append("test")
            # Remove duplicates if stage is None
            stages_to_setup = list(dict.fromkeys(stages_to_setup))

            for split_label in stages_to_setup:
                 if split_label not in splits_info:
                      # Log if a split relevant to the stage isn't defined by get_split_names
                      # Only warn for validation/test if they are optional
                      log_level = logging.INFO
                      if split_label == 'train': log_level = logging.WARNING # Missing train is usually bad
                      logging.log(log_level, f"Split '{split_label}' not defined in get_split_names for dataset '{name}' (Stage: {stage}). Skipping load attempt.")
                      continue # Skip loading this optional/undefined split

                 split_detail = splits_info[split_label]
                 tokenized_path = None # Define for potential error message
                 try:
                    tokenized_path = self._get_tokenized_split_path(dataset_cls, split_label, split_detail)
                    logging.info(f"Attempting to load '{split_label}' split for {name} from: {tokenized_path}")

                    if not tokenized_path.exists():
                        # Raise FileNotFoundError immediately if path doesn't exist
                         raise FileNotFoundError(f"Required tokenized data not found at: {tokenized_path}")

                    # Load the dataset
                    dataset_instance = dataset_cls(
                        split=split_label, # Pass the logical split name
                        specific_tokenized_path=str(tokenized_path) # Pass the base tokenized dir
                        # The dataset __init__ should reconstruct the specific path or receive it directly
                        # Let's adjust based on your BaseHuggingFaceDataset __init__ expecting the *specific* dir
                        # tokenized_data_path=str(tokenized_path) # <---- Pass the specific path directly
                    )

                    # Append to the correct list
                    if split_label == "train": self.train_datasets.append(dataset_instance)
                    elif split_label == "validation": self.val_datasets.append(dataset_instance)
                    elif split_label == "test": self.test_datasets.append(dataset_instance)

                    logging.info(f"Successfully loaded '{split_label}' split for {name}.")

                 except FileNotFoundError as e:
                     # Make FileNotFoundError critical as prepare_data should have created it
                     logging.error(f"CRITICAL: Tokenized '{split_label}' split for {name} NOT FOUND. "
                                   f"Expected at '{tokenized_path}'. Ensure prepare_data ran successfully on rank 0 "
                                   f"with the *exact same* config (tokenizer, percentages, etc.). Error: {e}", exc_info=True)
                     current_dataset_load_failed = True
                     failed_splits_for_dataset.append(split_label)
                     # Continue checking other splits for this dataset to provide a full error summary
                 except Exception as e:
                     # Make any other loading error critical
                     logging.error(f"CRITICAL: Failed loading '{split_label}' split for {name} from '{tokenized_path}': {e}", exc_info=True)
                     current_dataset_load_failed = True
                     failed_splits_for_dataset.append(split_label)
                     # Continue checking other splits

            # Store failure info for this dataset if any split failed
            if current_dataset_load_failed:
                dataset_load_failures[name] = failed_splits_for_dataset

        # After checking all datasets, raise an error if any failed
        if dataset_load_failures:
             error_message = "CRITICAL: Failed to load required tokenized splits during setup:\n"
             for ds_name, failed_splits in dataset_load_failures.items():
                 error_message += f"  - Dataset '{ds_name}': Failed to load split(s) {failed_splits}\n"
             error_message += f"Ensure prepare_data (rank 0) completed successfully with matching configurations (tokenizer: {self.tokenizer_id})."
             logging.error(error_message)
             # Use a specific error type if desired, e.g., RuntimeError or a custom one
             raise RuntimeError(error_message)


        # Combine datasets from different sources
        # Handle possibility of empty lists gracefully
        self.train_dataset = ConcatDataset(self.train_datasets) if self.train_datasets else None
        self.val_dataset = ConcatDataset(self.val_datasets) if self.val_datasets else None
        self.test_dataset = ConcatDataset(self.test_datasets) if self.test_datasets else None

        logging.info(f"Setup complete for stage '{stage}'. Combined datasets:")
        logging.info(f"  Train sources: {len(self.train_datasets)}, Total examples: {len(self.train_dataset) if self.train_dataset else 0}")
        logging.info(f"  Val sources: {len(self.val_datasets)}, Total examples: {len(self.val_dataset) if self.val_dataset else 0}")
        logging.info(f"  Test sources: {len(self.test_datasets)}, Total examples: {len(self.test_dataset) if self.test_dataset else 0}")


    def _create_dataloader(self, dataset, shuffle=False):
        """Creates a DataLoader for the given dataset. Raises error if creation fails."""
        if dataset is None or (isinstance(dataset, ConcatDataset) and not dataset.datasets) or len(dataset) == 0:
             stage = "unknown"
             if shuffle: stage = "train"
             # Determine stage based on which dataloader method called this (implicit)
             # This logging is just informational, returning None is the main action
             logging.info(f"Dataset for {stage} stage is None or empty. Returning None for DataLoader.")
             return None

        # Determine persistent_workers
        # Ensure consistent behavior: if num_workers is 0, persistent_workers should be False.
        persistent_workers = self.num_workers > 0

        try:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True, # Usually beneficial with GPUs
                persistent_workers=persistent_workers,
                # collate_fn=None # Use default unless custom logic needed
            )
            # Check if loader is unexpectedly empty right after creation (can happen with workers)
            # This check might be overly cautious depending on PL version and worker init
            # try:
            #     _ = iter(loader)
            # except Exception as iter_e:
            #      logging.error(f"DataLoader created, but failed to create iterator (potential worker issue?): {iter_e}", exc_info=True)
            #      raise RuntimeError(f"DataLoader iterator creation failed") from iter_e
            return loader
        except Exception as e:
             # Make DataLoader creation failure critical
             logging.error(f"CRITICAL: Failed to create DataLoader: {e}", exc_info=True)
             raise # Re-raise the exception

    # --- Dataloader Methods ---
    # Keep these simple, relying on _create_dataloader and setup's error handling
    def train_dataloader(self):
        logging.info("Creating train dataloader...")
        loader = self._create_dataloader(self.train_dataset, shuffle=True)
        if loader is None:
             # This case is generally handled by setup failing if train data is missing/empty
             # But log it if _create_dataloader returns None for some reason
             logging.warning("Train dataloader is None (likely no training data loaded in setup).")
        # No need to check len(loader.dataset) here, _create_dataloader handles empty datasets
        return loader

    def val_dataloader(self):
        logging.info("Creating validation dataloader...")
        loader = self._create_dataloader(self.val_dataset, shuffle=False)
        if loader is None:
             logging.info("Validation dataloader is None (no validation data loaded in setup).") # Info is fine
        return loader

    def test_dataloader(self):
        logging.info("Creating test dataloader...")
        loader = self._create_dataloader(self.test_dataset, shuffle=False)
        if loader is None:
             logging.warning("Test dataloader is None (no test data loaded in setup).") # Warning might be appropriate
        return loader


    # --- Unified Path Generation Method (Unchanged) ---
    def _get_tokenized_split_path(self, dataset_cls, split_label: str, split_info: dict) -> Path:
        """
        Generates the unique path for a specific tokenized dataset split file/directory,
        considering dataset name, config, split label, and percentage slicing.
        Raises ValueError if dataset class doesn't have dataset_name.
        """
        if not hasattr(dataset_cls, 'dataset_name') or dataset_cls.dataset_name is None:
             raise ValueError(f"Dataset class {dataset_cls.__name__} must have a `dataset_name` attribute.")
        if not hasattr(dataset_cls, 'dataset_config_name'):
             # Add default attribute if missing to avoid errors later
             setattr(dataset_cls, 'dataset_config_name', None) # Or log a warning

        config_suffix = dataset_cls.dataset_config_name or 'default'
        dataset_id = f"{dataset_cls.dataset_name}_{config_suffix}"

        percentage_value = split_info.get("percentage") # Can be None or range tuple/list
        perc_suffix = get_percentage_suffix(percentage_value) # Use the helper

        tokenized_split_filename = f"{dataset_id}_{split_label}{perc_suffix}"

        return self.tokenized_cache_dir / tokenized_split_filename

    # --- Debugging Helpers (Unchanged, but ensure they handle errors cleanly) ---
    def debug_all_datasets(self, split_label="train", num_samples=3):
        """ Debug tokenizer behavior for all datasets using streaming. """
        # (Keep existing implementation, ensure internal errors are logged clearly)
        # Consider adding try-except around load_dataset and tokenization calls inside the loop
        # to prevent one dataset failure from stopping the debug of others, but log errors.
        from datasets import load_dataset # Local import is fine here

        print(f"\n🔍 [DEBUG MODE] Streaming and tokenizing {num_samples} sample(s) from each dataset (split='{split_label}')")

        for name in self.dataset_names:
            if name not in dataset_registry:
                logging.warning(f"[DEBUG] Dataset '{name}' not found in registry. Skipping.")
                continue

            dataset_cls = dataset_registry[name]
            # Use getattr for safety in case attributes are missing unexpectedly
            hf_name = getattr(dataset_cls, 'dataset_name', 'unknown')
            hf_config = getattr(dataset_cls, 'dataset_config_name', None)
            text_column = getattr(dataset_cls, 'text_column', 'text') # Default to 'text'

            print(f"\n=== 🧪 Dataset: {name} | HF: {hf_name} | Config: {hf_config} | Split: {split_label} ===")

            try:
                # Get split info specific to this dataset for the debug split
                # Use default config if none provided for this dataset
                specific_config = self.dataset_configs.get(name, {})
                all_splits_info = dataset_cls.get_split_names(**specific_config)
                debug_split_info = all_splits_info.get(split_label)

                if not debug_split_info:
                     print(f"⚠️ Split '{split_label}' not defined by get_split_names for {name}. Skipping.")
                     continue

                # Use the hf_split_name from the info dict
                hf_split_name_to_load = debug_split_info.get("hf_split_name", split_label)

                # Load in streaming mode
                dataset = load_dataset(
                    hf_name,
                    hf_config,
                    split=hf_split_name_to_load,
                    streaming=True
                )

                processed_count = 0
                for i, example in enumerate(dataset):
                    if processed_count >= num_samples:
                        break

                    # ---- replicate tokenize_function logic manually ----
                    raw_text = example.get(text_column)
                    actual_text_column = text_column
                    if raw_text is None:
                        possible_cols = [k for k, v in example.items() if isinstance(v, str) and v.strip()]
                        if not possible_cols:
                            print(f"⚠️ Sample {i} skipped: No valid text column found (tried '{text_column}').")
                            continue
                        actual_text_column = possible_cols[0]
                        raw_text = example[actual_text_column]
                        print(f"⚠️ Column '{text_column}' not found or empty, using fallback: '{actual_text_column}'")

                    if not isinstance(raw_text, str): raw_text = str(raw_text)
                    if not raw_text.strip():
                        print(f"\n--- Sample {i} (using '{actual_text_column}') --- (Empty text, skipping)")
                        continue

                    # Ensure pad token exists (redundant with __init__, but safe)
                    if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                    try:
                        tokenized = self.tokenizer(
                            raw_text,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors=None,
                            return_attention_mask=True # Ensure mask is returned
                        )
                        input_ids = tokenized["input_ids"]
                        attention_mask = tokenized.get("attention_mask") # Use .get for safety

                        print(f"\n--- Sample {i} (using '{actual_text_column}') ---")
                        # Truncate raw text for cleaner printing if very long
                        print_text = (raw_text[:100] + '...') if len(raw_text) > 103 else raw_text
                        print(f"📝 Raw Text:\n{repr(print_text)}")
                        print(f"🔢 input_ids[:20]: {input_ids[:20]}")
                        print(f"📏 Length: {len(input_ids)} (non-pad: {sum(attention_mask) if attention_mask else 'n/a'})")
                        # Handle potential decoding errors
                        try:
                             decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                             print(f"🧠 Decoded:\n{decoded_text}")
                        except Exception as decode_e:
                            print(f"🧠 Decoding failed: {decode_e}")

                        if attention_mask: print(f"🛡️ attention_mask[:20]: {attention_mask[:20]}")
                        else: print(f"🛡️ attention_mask: Not returned by tokenizer")

                        processed_count += 1

                    except Exception as e:
                        print(f"⚠️ Tokenization failed for sample {i}: {e}")
                        # Continue to next sample in debug mode

            except Exception as e:
                print(f"❌ Could not stream/process dataset '{name}' split '{split_label}': {e}")
                # Continue to next dataset in debug mode


# --- Helper Functions (Unchanged) ---
def get_percentage_suffix(percentage_info):
    """Generates a filename-safe suffix based on percentage slice."""
    if percentage_info is None:
        return "_full"
    elif isinstance(percentage_info, (float, int)):
        # Ensure percentage is treated as a fraction (e.g., 0.1 not 10.0)
        frac = float(percentage_info)
        if not (0.0 <= frac <= 1.0):
            logging.warning(f"Percentage value {frac} is outside [0.0, 1.0]. Interpreting as fraction.")
            # Decide handling: clamp, error, or assume it's already a fraction
            frac = max(0.0, min(1.0, frac)) # Clamp for safety
        percent_x_1000 = int(frac * 1000)
        return f"_p{percent_x_1000:03d}" # p000 to p999 (represents 0.0% to 99.9%)
    elif isinstance(percentage_info, (list, tuple)) and len(percentage_info) == 2:
        start_frac = float(percentage_info[0])
        end_frac = float(percentage_info[1])
        if not (0.0 <= start_frac <= 1.0 and 0.0 <= end_frac <= 1.0 and start_frac < end_frac):
             logging.error(f"Invalid percentage range: {percentage_info}. Must be 0.0 <= start < end <= 1.0.")
             # Raise error for invalid range as it affects slicing logic
             raise ValueError(f"Invalid percentage range for cache suffix: {percentage_info}")
        start_x_1000 = int(start_frac * 1000)
        end_x_1000 = int(end_frac * 1000)
        return f"_r{start_x_1000:03d}_{end_x_1000:03d}" # e.g., _r010_050 for [0.01, 0.05]
    else:
        logging.error(f"Unexpected percentage format: {percentage_info}. Cannot generate reliable suffix.")
        raise TypeError(f"Unsupported percentage format for cache suffix: {type(percentage_info)}")


def inspect_batch(batch, tokenizer=None, max_items=2, decode_labels=True):
    """ Inspect a batch of samples from a DataLoader. """
    # (Keep existing implementation)
    print("\n Inspecting Batch Sample:")
    if not isinstance(batch, dict):
        print(f"  Batch is not a dict, type: {type(batch)}")
        return
    for k, v in batch.items():
        # Check if value is tensor-like before accessing shape/dtype
        if hasattr(v, 'shape') and hasattr(v, 'dtype'):
             print(f"  {k}: shape={v.shape}, dtype={v.dtype}, type={type(v)}")
        else:
             print(f"  {k}: type={type(v)}, value (sample): {v[:max_items] if isinstance(v, (list, tuple)) else v}")


    if "input_ids" not in batch or not hasattr(batch["input_ids"], 'shape'):
        print("  'input_ids' not found or not a tensor in batch.")
        return

    print(f"\n Showing first {max_items} decoded item(s):")
    num_in_batch = batch["input_ids"].shape[0]

    for i in range(min(max_items, num_in_batch)):
        print(f"\n--- Sample {i} ---")

        try:
            input_ids = batch["input_ids"][i].tolist()
            # Use .get for optional keys like attention_mask and labels
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")

            print(f"input_ids[:20]: {input_ids[:20]} ...")
            if attention_mask is not None: print(f"attention_mask[:20]: {attention_mask[i].tolist()[:20]} ...")
            if labels is not None: print(f"labels[:20]: {labels[i].tolist()[:20]} ...")
            print(f"Full input_ids length: {len(input_ids)}")

            if tokenizer:
                # Decode input_ids
                try:
                    # Filter out pad tokens *before* decoding for cleaner output, if desired
                    # actual_input_ids = [t for t in input_ids if t != tokenizer.pad_token_id]
                    # decoded_input = tokenizer.decode(actual_input_ids, skip_special_tokens=True)
                    # Or decode directly:
                    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
                    print(f" Decoded input_ids:\n{decoded_input}")
                except Exception as e:
                    print(f" Could not decode input_ids for sample {i}: {e}")

                # Decode labels if present, different, and requested
                if decode_labels and labels is not None:
                    labels_list = labels[i].tolist()
                    # Causal LM often has labels == input_ids, avoid redundant decoding/printing
                    if labels_list != input_ids:
                        try:
                            # Filter out ignore_index (often -100) before decoding labels
                            actual_label_ids = [t for t in labels_list if t != -100]
                            decoded_labels = tokenizer.decode(actual_label_ids, skip_special_tokens=True)
                            print(f"  Decoded labels (ignore_index=-100 removed):\n{decoded_labels}")
                        except Exception as e:
                            print(f" Could not decode labels for sample {i}: {e}")
                    else:
                        print("  Labels are identical to input_ids.")


        except Exception as e:
            print(f" Error processing sample {i} in inspect_batch: {e}")


# --- Main Execution Example (Updated Cache Dir Structure & Exception Handling) ---
if __name__ == "__main__":

    # os.environ["HF_HUB_OFFLINE"] = "1" # Example: Force offline mode if needed

    # Use the setup_logging utility if available and configured
    try:
        setup_logging(level=logging.INFO,file_logging=True) # Or DEBUG for more verbosity
    except NameError:
        # Fallback if setup_logging is not defined or imported
        logging.warning("setup_logging utility not found. Using basicConfig.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Define cache directories relative to project root
    # Using more descriptive names potentially
    DATA_CACHE_ROOT = PROJECT_ROOT / "data" / "main_cache"
    MODEL_CACHE_ROOT = PROJECT_ROOT / "data" / "model_cache" # For HF downloads

    # Create directories if they don't exist
    DATA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    logging.info(f"Project Root: {PROJECT_ROOT}")
    logging.info(f"Data Cache Root: {DATA_CACHE_ROOT}") # Where tokenized data goes
    logging.info(f"Model Cache Dir: {MODEL_CACHE_ROOT}") # Where HF downloads models/tokenizers

    # --- Configuration ---
    tokenizer_name = "distilgpt2"
    # Use a smaller batch size and max_length for faster local testing/debugging
    batch_size_debug = 2
    max_length_debug = 64
    num_dataloader_workers_debug = 25 # Set to 0 for easier debugging (avoids multiprocessing issues)
    # num_dataloader_workers_debug = max(0, os.cpu_count() // 2 if os.cpu_count() else 0) # Use more cores

    # --- Load Tokenizer ---
    logging.info(f"\nLoading tokenizer '{tokenizer_name}'...")
    try:
        # Ensure download mode handles potential offline scenarios if needed
        # force_reuse_cache_if_exists might be problematic if cache is corrupt
        # Consider using default behavior or REUSE_CACHE_IF_EXISTS
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=str(MODEL_CACHE_ROOT),
            # download_mode="force_reuse_cache_if_exists" # Use with caution
        )
        logging.info(f"Tokenizer '{tokenizer_name}' loaded successfully.")
    except Exception as e:
         logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}", exc_info=True)
         raise # Stop script if tokenizer fails


    # --- Define Dataset Configurations ---
    # Use more descriptive variable name
    # Example: Slice Oscar differently for train/val if needed by defining separate entries
    # or handling within the OscarDataset class based on split label
    dataset_processing_configs = {
        "wikitext": {},
        "oscar": {
            # Specify slicing ONLY for the 'train' split via a specific key
            # The key name ('train_split_percentage') should match what OscarDataset.get_split_names expects
            "train_split_percentage": [0.0001, 0.0005] # Use a tiny slice: 0.01% to 0.05% of train data
        },
        "bookcorpus": {
            # No specific config, will use default 'train' split
        }
        # Add "openwebtext": {} if you have that class registered
    }
    logging.info(f"\nUsing Dataset Configurations: {dataset_processing_configs}")

    # --- Initialize DataModule ---
    logging.info("\nInitializing DataModule...")
    try:
        dm = LightningDataModule(
            tokenizer=tokenizer,
            dataset_configs=dataset_processing_configs, # Use the defined configs
            batch_size=batch_size_debug,
            max_length=max_length_debug,
            num_workers=num_dataloader_workers_debug,
            cache_dir=str(DATA_CACHE_ROOT) # Pass the specific data cache path
        )
        logging.info("DataModule initialized successfully.")
        logging.info(f"  Raw data cache expected in: {dm.raw_cache_dir}")
        logging.info(f"  Tokenized data cache expected in: {dm.tokenized_cache_dir}\n")
    except Exception as e:
        logging.error(f"Failed to initialize LightningDataModule: {e}", exc_info=True)
        raise # Stop script if DataModule init fails

    # --- Optional: Debug Tokenizer Behavior ---
    logging.info("----- Optional: Debugging Tokenizer on Raw Data -----")
    try:
        # Debug on 'train' split, check first 2 samples from each dataset
        dm.debug_all_datasets(split_label="train", num_samples=2)
    except Exception as e:
        logging.error(f"Error during debug_all_datasets: {e}", exc_info=True)
        # Decide if this failure is critical - maybe just warn and continue?
        logging.warning("Continuing after error in debug_all_datasets.")
    logging.info("----- End of Debug Tokenizer -----")


    # --- Run prepare_data (Rank 0 Only) ---
    # In a real multi-node setup, this is handled by Lightning automatically.
    # Here, we simulate it by just running it.
    logging.info("\n--- Running prepare_data() ---")
    try:
        # This call assumes we are effectively rank 0 in this script execution.
        dm.prepare_data()
        logging.info("--- prepare_data() finished successfully ---")
    except Exception as e:
        logging.error(f"--- CRITICAL ERROR during prepare_data(): {e} ---", exc_info=True)
        # <<< MODIFICATION: Changed sys.exit(1) to raise >>>
        raise # Re-raise the exception to stop execution and get traceback


    # --- Run setup (All Ranks) ---
    # This simulates what happens on each node/process.
    logging.info("\n--- Running setup('fit') ---")
    try:
        dm.setup("fit") # Setup for training and validation
        logging.info("--- setup('fit') finished successfully ---")
    except Exception as e:
        logging.error(f"--- CRITICAL ERROR during setup('fit'): {e} ---", exc_info=True)
        # <<< MODIFICATION: Changed sys.exit(1) to raise >>>
        raise # Re-raise the exception


    # --- Test Dataloaders ---
    logging.info("\n--- Testing train_dataloader ---")
    train_loader = dm.train_dataloader() # Should raise error internally if creation fails
    if train_loader and len(train_loader.dataset) > 0:
        try:
            logging.info(f"Train dataset size: {len(train_loader.dataset)}")
            start_batch_time = time.time()
            batch = next(iter(train_loader))
            end_batch_time = time.time()
            logging.info(f"Successfully fetched batch from train_dataloader in {end_batch_time - start_batch_time:.2f}s.")
            inspect_batch(batch, tokenizer=tokenizer, max_items=1) # Show only 1 item for brevity
            # print("Sample batch details:")
            # for k, v in batch.items(): print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        except StopIteration:
            logging.error("Train dataloader is unexpectedly empty after setup succeeded.")
            raise # Raise error if loader is empty when it shouldn't be
        except Exception as e:
            logging.error(f"Error getting batch from train_dataloader: {e}", exc_info=True)
            raise # Re-raise any other exception
    else:
        # If setup succeeded, loader should ideally not be None/empty unless dataset was truly empty.
        logging.warning("Train dataloader is None or empty. Check dataset sizes and setup logs.")
        # Decide if this is critical - for training, it usually is.
        # raise RuntimeError("Train dataloader is missing after setup, cannot proceed.")


    logging.info("\n--- Testing val_dataloader ---")
    val_loader = dm.val_dataloader()
    if val_loader and len(val_loader.dataset) > 0:
        try:
            logging.info(f"Validation dataset size: {len(val_loader.dataset)}")
            start_batch_time = time.time()
            val_batch = next(iter(val_loader))
            end_batch_time = time.time()
            logging.info(f"Successfully fetched batch from val_dataloader in {end_batch_time - start_batch_time:.2f}s.")
            inspect_batch(val_batch, tokenizer=tokenizer, max_items=1)
        except StopIteration:
            # This might be acceptable if a dataset genuinely has no validation split
            logging.warning("Validation dataloader is empty (might be expected).")
        except Exception as e:
            logging.error(f"Error getting batch from val_dataloader: {e}", exc_info=True)
            # Decide if validation loader failure is critical
            # raise e # Optionally raise
    else:
         logging.info("Validation dataloader is None or empty (Note: Oscar/Bookcorpus may have no default val split).")


    # --- Optional: Test setup and dataloader for 'test' stage ---
    logging.info("\n--- Running setup('test') ---")
    try:
        dm.setup("test") # Setup for test stage
        logging.info("--- setup('test') finished successfully ---")
    except Exception as e:
        logging.error(f"--- CRITICAL ERROR during setup('test'): {e} ---", exc_info=True)
        raise # Re-raise


    logging.info("\n--- Testing test_dataloader ---")
    test_loader = dm.test_dataloader()
    if test_loader and len(test_loader.dataset) > 0:
         try:
            logging.info(f"Test dataset size: {len(test_loader.dataset)}")
            start_batch_time = time.time()
            test_batch = next(iter(test_loader))
            end_batch_time = time.time()
            logging.info(f"Successfully fetched batch from test_dataloader in {end_batch_time - start_batch_time:.2f}s.")
            inspect_batch(test_batch, tokenizer=tokenizer, max_items=1)
         except StopIteration:
            logging.warning("Test dataloader is empty (might be expected).")
         except Exception as e:
            logging.error(f"Error getting batch from test_dataloader: {e}", exc_info=True)
            # Decide if test loader failure is critical
            # raise e # Optionally raise
    else:
         logging.info("Test dataloader is None or empty (Note: Oscar/Bookcorpus may have no default test split).")


    # --- Cleanup ---
    # Important if using persistent workers (num_workers > 0) in the script itself
    # In a full PL Trainer run, Trainer handles worker shutdown.
    if num_dataloader_workers_debug > 0:
        print("\n--- Cleaning up dataloaders to potentially terminate persistent workers ---")
        # Explicitly delete loaders to help Python's GC and potentially trigger worker shutdown
        try:
            del batch
            del val_batch
            del test_batch
        except NameError:
            pass # Variables might not exist if errors occurred earlier
        del train_loader
        del val_loader
        del test_loader
        import gc
        gc.collect()
        logging.info("DataLoader variables deleted; persistent workers should terminate if idle.")
        # Note: Sometimes workers might linger; ensuring the main script exits cleanly is key.

    print("\n--- Script finished ---")