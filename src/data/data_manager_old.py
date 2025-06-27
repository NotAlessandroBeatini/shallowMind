import logging
# Set root logger to DEBUG for your debug messages, INFO for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import sys
import re
from pathlib import Path
import time

# --- Robust Project Root Finding (Alternative to get_cwd) ---
# Assuming data_manager.py is somewhere within your project structure
try:
    # Assumes data_manager.py is in root or a direct subdir like 'src/data'
    # Adjust `parents[N]` based on the file's location relative to the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # If data_manager.py is in root
    # PROJECT_ROOT = Path(__file__).resolve().parent.parent # If in src/data
    # PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # If deeper
    logging.info(f"Determined Project Root: {PROJECT_ROOT}")
except NameError:
     # __file__ is not defined, likely running interactively or differently
     logging.warning("__file__ not defined, using current working directory as project root.")
     PROJECT_ROOT = Path(os.getcwd())
     CWD_PATH_STR = str(PROJECT_ROOT)
     # Apply the specific davinci-1 substitution if needed, but note its fragility
     if "/home" in CWD_PATH_STR:
          CWD_PATH_STR = re.sub(r"^.*(?=/home)", "/davinci-1", CWD_PATH_STR)
          PROJECT_ROOT = Path(CWD_PATH_STR)
          logging.warning(f"Applied davinci-1 specific path substitution. Root set to: {PROJECT_ROOT}")


sys.path.append(str(PROJECT_ROOT)) # Add project root to path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
import src.data.my_datasets # Important: Ensures registration happens
from src.data.my_datasets import dataset_registry, _tokenize_and_save_split # Import the helper
from src.utils.utils import setup_logging


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
        # Sanitize the tokenizer name for use in paths (replace slashes)
        # Handles potential None value gracefully, though tokenizer should always be provided
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', 'unknown_tokenizer')
        self.tokenizer_id = tokenizer_name.replace("/", "__")
        

        # Resolve cache_dir relative to PROJECT_ROOT if it's not absolute
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = PROJECT_ROOT / self.cache_dir
            logging.info(f"Relative cache_dir provided. Resolved to: {self.cache_dir}")

        self.raw_cache_dir = self.cache_dir / "raw"
        # Make the tokenized cache directory specific to the tokenizer
        self.tokenized_cache_dir = self.cache_dir / "tokenized" / self.tokenizer_id

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        # Create the tokenizer-specific directory
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
                logging.error("Tokenizer has no pad_token and no eos_token. Cannot set a default pad token!")
                exit()
                # Consider adding a default pad token manually if necessary, e.g.,
                # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # raise ValueError("Tokenizer needs a pad_token or eos_token.") # Or raise error


    # --- Updated prepare_data ---
# --- Updated prepare_data ---
    def prepare_data(self):
        logging.info(f"--- Starting prepare_data (runs only on rank 0) ---")
        logging.info(f"Using Raw cache directory: {self.raw_cache_dir}")
        logging.info(f"Using Tokenized cache directory (for {self.tokenizer_id}): {self.tokenized_cache_dir}")

        for name in self.dataset_names:
            if name not in dataset_registry:
                logging.error(f"Dataset '{name}' not found in registry. Available: {list(dataset_registry.keys())}. Skipping.")
                continue

            dataset_cls = dataset_registry[name]

            # <<< FIX: Get specific_config *before* logging it >>>
            specific_config = self.dataset_configs.get(name, {})
            # Now it's safe to log
            logging.info(f"Preparing dataset: {name} (Class: {dataset_cls.__name__}) with specific config: {specific_config}")

            try:
                # Now use the defined specific_config here
                splits_to_process = dataset_cls.get_split_names(**specific_config)
            except Exception as e:
                # Log the error and continue to the next dataset if getting split names fails
                logging.error(f"Failed to get split names for {name} using config {specific_config}: {e}. Skipping dataset.", exc_info=True)
                continue # <-- Allows processing other datasets if one fails here

            # Process each defined split ('train', 'validation', 'test', etc.)
            for split_label, split_info in splits_to_process.items():
                logging.info(f"Processing split '{split_label}' for {name} (Info: {split_info})")

                # --- Step 1: Ensure Raw Data Split Exists ---
                raw_split_dataset = None
                try:
                    # Pass the specific info for this split
                    raw_split_dataset = dataset_cls.download_raw_split(
                        split_info=split_info, # Contains hf_split_name and percentage
                        raw_cache_dir=str(self.raw_cache_dir),
                        # Pass other relevant kwargs if needed from specific_config
                        trust_remote_code=getattr(dataset_cls, 'trust_remote_code', False) # Ensure trust_remote_code is passed if needed
                    )
                    if raw_split_dataset is None:
                         raise ValueError("download_raw_split returned None")
                    logging.info(f"Raw data ready for {name} split '{split_label}' (slice: {split_info.get('percentage')}). Size: {len(raw_split_dataset)}")

                except Exception as e:
                    logging.error(f"Failed to prepare/slice raw data for {name} split '{split_label}'. Skipping tokenization for this split. Error: {e}", exc_info=True)
                    continue # Move to the next split for this dataset

                # --- Step 2: Generate Tokenized Path & Check/Tokenize ---
                try:
                    # Use the unified function, gives string with split and percentage
                    tokenized_split_path = self._get_tokenized_split_path(dataset_cls, split_label, split_info)
                    logging.info(f"Target tokenized path: {tokenized_split_path}")

                    if tokenized_split_path.exists():
                        logging.info(f"Tokenized data already exists. Skipping tokenization.")
                        continue
                    else:
                        # Ensure raw_split_dataset is valid before tokenizing
                        if raw_split_dataset is None:
                             logging.error(f"Cannot tokenize {name} split '{split_label}' because raw data is missing.")
                             raise Exception # Skip to next split if raw data failed

                        logging.info(f"Tokenized data not found. Starting tokenization...")
                        start_time = time.time()
                        success = _tokenize_and_save_split(
                            raw_split_dataset=raw_split_dataset,
                            tokenized_cache_path=tokenized_split_path, # Use the specific path
                            dataset_name=f"{name}_{split_label}", # More specific logging ID
                            split_label=split_label, # For logging inside _tokenize...
                            tokenizer=self.tokenizer,
                            max_length=self.max_length,
                            text_column=dataset_cls.text_column,
                            num_proc_to_use=self.num_workers
                        )
                        tokenization_time = time.time() - start_time
                        if success:
                           logging.info(f"Tokenization for {tokenized_split_path.name} completed successfully in {tokenization_time:.2f}s.")
                        else:
                           # Make error message more specific
                           logging.error(f"Tokenization failed for {tokenized_split_path}. Check logs from _tokenize_and_save_split.")
                           # Decide if failure should be fatal for the whole prepare_data step
                           raise RuntimeError(f"Tokenization failed for path {tokenized_split_path}")

                except Exception as e:
                     logging.error(f"Error during tokenization/saving for {name} split '{split_label}' at path '{tokenized_split_path}': {e}", exc_info=True)
                     # Stop the whole process if one tokenization fails critically after raw download
                     raise

        logging.info("--- Finished prepare_data ---")


    # --- Updated setup ---
    def setup(self, stage=None):
        logging.info(f"--- Setting up data for stage: {stage} (runs on all ranks) ---")
        # Clear lists at the beginning of setup for the current stage
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        for name in self.dataset_names:

            if name not in dataset_registry:
                logging.error(f"Dataset '{name}' not found in registry during setup. Skipping loading.")
                raise ValueError(f"Dataset '{name}' requested in config but not found in registry.")

            # Get the class from the registry
            dataset_cls = dataset_registry[name]
            # Get the specific config for this dataset
            specific_config = self.dataset_configs.get(name, {})

            try:
                # Get the split info *again* based on the current config
                splits_info = dataset_cls.get_split_names(**specific_config)
            except Exception as e:
                 logging.error(f"Failed to get split names for {name} during setup: {e}. Cannot load.", exc_info=True)
                 raise # Skip this dataset if config is bad

            logging.info(f"Setting up dataset: {name} (tokenizer: {self.tokenizer_id}). Expecting splits based on: {splits_info}")

            # --- Load based on stage ---
            loaded_successfully = True # Track per dataset

            if stage in ("fit", None):
                # Setup Train Split
                if "train" in splits_info:
                    split_label = "train"
                    split_detail = splits_info[split_label] # Get info for 'train'
                    try:
                        # *** Use the unified function ***
                        tokenized_path = self._get_tokenized_split_path(dataset_cls, split_label, split_detail)
                        logging.info(f"Attempting to load '{split_label}' split for {name} from: {tokenized_path}")
                        # Pass the specific final path to the Dataset constructor
                        self.train_datasets.append(dataset_cls(
                            split=split_label,
                            tokenized_data_path=str(tokenized_path)
                        ))
                        logging.info(f"Successfully loaded '{split_label}' split for {name}.")
                    except FileNotFoundError:
                         logging.error(f"CRITICAL: Tokenized '{split_label}' split for {name} NOT FOUND at {tokenized_path}. "
                                       f"Ensure prepare_data ran successfully with matching config (percentage: {split_detail.get('percentage')}).")
                         loaded_successfully = False
                         raise
                    except Exception as e:
                        logging.error(f"Failed loading '{split_label}' split for {name}: {e}", exc_info=True)
                        loaded_successfully = False
                        raise

                # Setup Validation Split (using its own split_info)
                if "validation" in splits_info:
                    split_label = "validation"
                    split_detail = splits_info[split_label] # Get info for 'validation'
                    try:
                        # *** Use the unified function ***
                        tokenized_path = self._get_tokenized_split_path(dataset_cls, split_label, split_detail)
                        logging.info(f"Attempting to load '{split_label}' split for {name} from: {tokenized_path}")
                        self.val_datasets.append(dataset_cls(
                            split=split_label,
                            tokenized_data_path=str(tokenized_path)
                        ))
                        logging.info(f"Successfully loaded '{split_label}' split for {name}.")
                    except FileNotFoundError:
                        logging.error(f"CRITICAL: Tokenized '{split_label}' split for {name} NOT FOUND at {tokenized_path}. "
                                      f"Ensure prepare_data ran successfully with matching config (percentage: {split_detail.get('percentage')}).")
                        # Decide if missing validation is critical
                        # loaded_successfully = False # Optional: Make missing validation fatal
                    except Exception as e:
                        logging.error(f"Failed loading '{split_label}' split for {name}: {e}", exc_info=True)
                        # loaded_successfully = False # Optional: Make failed validation fatal
                #else: # Log if validation expected but not found in config splits_info
                #     if "train" in splits_info: # Only warn if train exists but val doesn't
                #         logging.warning(f"No 'validation' split defined in get_split_names or config for dataset '{name}'.")


            # Setup Test Split (using its own split_info)
            if stage in ("test", None):
                 if "test" in splits_info:
                    split_label = "test"
                    split_detail = splits_info[split_label] # Get info for 'test'
                    try:
                        # *** Use the unified function ***
                        tokenized_path = self._get_tokenized_split_path(dataset_cls, split_label, split_detail)
                        logging.info(f"Attempting to load '{split_label}' split for {name} from: {tokenized_path}")
                        self.test_datasets.append(dataset_cls(
                             split=split_label,
                             tokenized_data_path=str(tokenized_path)
                        ))
                        logging.info(f"Successfully loaded '{split_label}' split for {name}.")
                    except FileNotFoundError:
                        logging.error(f"CRITICAL: Tokenized '{split_label}' split for {name} NOT FOUND at {tokenized_path}. "
                                      f"Ensure prepare_data ran successfully with matching config (percentage: {split_detail.get('percentage')}).")
                        loaded_successfully = False # Usually missing test data is critical if requested
                    except Exception as e:
                        logging.error(f"Failed loading '{split_label}' split for {name}: {e}", exc_info=True)
                        loaded_successfully = False

            # Check if loading failed critically for this dataset
            if not loaded_successfully:
                 # Make the error message more informative
                 missing_splits = []
                 if stage in ("fit", None) and "train" in splits_info and not any(ds.split == 'train' and ds.__class__ == dataset_cls for ds in self.train_datasets):
                     missing_splits.append("'train'")
                 if stage in ("fit", None) and "validation" in splits_info and not any(ds.split == 'validation' and ds.__class__ == dataset_cls for ds in self.val_datasets):
                     missing_splits.append("'validation'") # Or make optional based on your needs
                 if stage in ("test", None) and "test" in splits_info and not any(ds.split == 'test' and ds.__class__ == dataset_cls for ds in self.test_datasets):
                     missing_splits.append("'test'")

                 raise RuntimeError(f"Failed to load required tokenized split(s) {', '.join(missing_splits)} for dataset '{name}' (tokenizer: {self.tokenizer_id}). Check logs above for specific paths and errors.")


        # Combine datasets from different sources (remains the same)
        self.train_dataset = ConcatDataset(self.train_datasets) if self.train_datasets else None
        self.val_dataset = ConcatDataset(self.val_datasets) if self.val_datasets else None
        self.test_dataset = ConcatDataset(self.test_datasets) if self.test_datasets else None

        logging.info(f"Setup complete for stage '{stage}'. Combined datasets:")
        logging.info(f"  Train sources: {len(self.train_datasets)}, Total examples: {len(self.train_dataset) if self.train_dataset else 0}")
        logging.info(f"  Val sources: {len(self.val_datasets)}, Total examples: {len(self.val_dataset) if self.val_dataset else 0}")
        logging.info(f"  Test sources: {len(self.test_datasets)}, Total examples: {len(self.test_dataset) if self.test_dataset else 0}")
        



    def _create_dataloader(self, dataset, shuffle=False):
        """Creates a DataLoader for the given dataset."""
        if dataset is None or (isinstance(dataset, ConcatDataset) and not dataset.datasets) or len(dataset) == 0:
             stage = "unknown"
             # Try to determine stage based on context (best effort)
             if shuffle: stage = "train"
             # Accessing trainer directly might not always be safe, depends on PL version and context
             # elif self.trainer and self.trainer.validating: stage = "validation"
             # elif self.trainer and self.trainer.testing: stage = "test"

             # Log appropriately
             log_level = logging.WARNING # Warn if trying to create loader for empty dataset
             logging.log(log_level, f"Cannot create DataLoader for {stage} stage: dataset is None or empty.")
             # Return a dummy loader that yields nothing to prevent crashes downstream
             # return DataLoader([], batch_size=self.batch_size, num_workers=self.num_workers)
             # Returning None might be cleaner if subsequent code checks for None
             return None

        # Determine persistent_workers (good practice for num_workers > 0) 
        # BE CAREFUL THEY WON't Let the process end if not handled properly
        persistent_workers = self.num_workers > 0

        try:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True, # Usually beneficial with GPUs
                persistent_workers=persistent_workers,
                # collate_fn=None # Use default collate_fn
            )
        except Exception as e:
             logging.error(f"Failed to create DataLoader: {e}", exc_info=True)
             raise # Re-raise the exception

    def train_dataloader(self):
        logging.info("Creating train dataloader...")
        loader = self._create_dataloader(self.train_dataset, shuffle=True)
        if loader is None:
             logging.warning("Train dataloader is None (no training data).")
        return loader

    def val_dataloader(self):
        logging.info("Creating validation dataloader...")
        loader = self._create_dataloader(self.val_dataset, shuffle=False)
        if loader is None:
             logging.info("Validation dataloader is None (no validation data).") # Info level, as validation is optional
        return loader

    def test_dataloader(self):
        logging.info("Creating test dataloader...")
        loader = self._create_dataloader(self.test_dataset, shuffle=False)
        if loader is None:
             logging.warning("Test dataloader is None (no test data).") # Warning level, as test data might be expected
        return loader


    # --- Unified Path Generation Method ---
    def _get_tokenized_split_path(self, dataset_cls, split_label: str, split_info: dict) -> Path:
        """
        Generates the unique path for a specific tokenized dataset split,
        considering dataset name, config, split label, and percentage slicing.

        Args:
            dataset_cls: The dataset class (e.g., OscarDataset).
            split_label: The logical split name ('train', 'validation', 'test').
            split_info: The dictionary containing details for this split,
                        must include the 'percentage' key (can be None).

        Returns:
            A Path object for the specific tokenized split file/directory.
        """
        if dataset_cls.dataset_name is None:
             raise ValueError("Dataset class must have a `dataset_name` attribute.")
             
        config_suffix = dataset_cls.dataset_config_name or 'default'
        dataset_id = f"{dataset_cls.dataset_name}_{config_suffix}"

        # Get the percentage info for this specific split
        percentage_value = split_info.get("percentage") # Can be None
        perc_suffix = get_percentage_suffix(percentage_value) # Use the helper

        # Construct the unique filename/directory name for this split variant
        tokenized_split_filename = f"{dataset_id}_{split_label}{perc_suffix}"

        # Return the full path within the tokenizer-specific directory
        return self.tokenized_cache_dir / tokenized_split_filename

    def debug_all_datasets(self, split_label="train", num_samples=3):
        """
        Debug tokenizer behavior for all datasets using streaming mode.
        This re-implements the same logic as the tokenize_function inside _tokenize_and_save_split,
        but without depending on closure/context.
        """
        from datasets import load_dataset

        print(f"\n🔍 [DEBUG MODE] Streaming and tokenizing {num_samples} sample(s) from each dataset (split='{split_label}')")

        for name in self.dataset_names:
            if name not in dataset_registry:
                logging.warning(f"Dataset '{name}' not found in registry. Skipping.")
                continue

            dataset_cls = dataset_registry[name]
            hf_name = dataset_cls.dataset_name
            hf_config = dataset_cls.dataset_config_name
            text_column = dataset_cls.text_column

            print(f"\n=== 🧪 Dataset: {name} | HF: {hf_name} | Split: {split_label} ===")

            try:
                dataset = load_dataset(
                    hf_name,
                    hf_config,
                    split=split_label,
                    streaming=True
                )
            except Exception as e:
                print(f"❌ Could not stream dataset '{name}': {e}")
                continue

            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break

                # ---- replicate tokenize_function logic manually ----
                raw_text = example.get(text_column)
                if raw_text is None:
                    # Try auto-detecting the text column (like in your tokenize_function)
                    possible_cols = [
                        k for k, v in example.items()
                        if isinstance(v, str) and v.strip()
                    ]
                    if not possible_cols:
                        print(f"⚠️ Sample {i} skipped: No valid text column found.")
                        continue
                    text_column_fallback = possible_cols[0]
                    raw_text = example[text_column_fallback]
                    print(f"⚠️ Column '{text_column}' not found, using fallback: '{text_column_fallback}'")

                if not isinstance(raw_text, str):
                    raw_text = str(raw_text)

                if not raw_text.strip():
                    print(f"\n--- Sample {i} --- (Empty text, skipping)")
                    continue

                # Ensure pad token exists
                if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                try:
                    tokenized = self.tokenizer(
                        raw_text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors=None
                    )

                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized.get("attention_mask")

                    print(f"\n--- Sample {i} ---")
                    print(f"📝 Raw Text:\n{repr(raw_text)}")
                    print(f"🔢 input_ids[:10]: {input_ids[:10]}")
                    print(f"📏 Length: {len(input_ids)} (non-pad: {sum(attention_mask) if attention_mask else 'n/a'})")
                    print(f"🧠 Decoded:\n{self.tokenizer.decode(input_ids, skip_special_tokens=True)}")
                    if attention_mask:
                        print(f"🛡️  attention_mask[:10]: {attention_mask[:10]}")

                except Exception as e:
                    print(f"⚠️ Tokenization failed for sample {i}: {e}")




# In data_manager.py or a utils file
def get_percentage_suffix(percentage_info):
    """Generates a filename-safe suffix based on percentage slice."""
    if percentage_info is None:
        return "_full" # Indicates the full dataset split was used
    elif isinstance(percentage_info, (float, int)):
         # e.g., 0.1 -> _p100 (representing 10.0%)
         # e.g., 0.005 -> _p005 (representing 0.5%)
         # Using integer representation avoids float issues in filenames
        percent_x_1000 = int(float(percentage_info) * 1000)
        return f"_p{percent_x_1000:03d}"
    elif isinstance(percentage_info, (list, tuple)) and len(percentage_info) == 2:
        # e.g., [0.01, 0.05] -> _r010_050
        start_x_1000 = int(float(percentage_info[0]) * 1000)
        end_x_1000 = int(float(percentage_info[1]) * 1000)
        return f"_r{start_x_1000:03d}_{end_x_1000:03d}"
    else:
        # Fallback for unexpected types, maybe hash it or raise error
        logging.warning(f"Unexpected percentage format: {percentage_info}. Using generic suffix.")
        # Or raise ValueError("Invalid percentage format for cache suffix")
        return f"_customslice" # Or hash(str(percentage_info))


def inspect_batch(batch, tokenizer=None, max_items=2, decode_labels=True):
    """
    Inspect a batch of samples from a DataLoader.

    Args:
        batch (dict): A batch returned by the DataLoader.
        tokenizer (transformers.PreTrainedTokenizer, optional): If provided, decodes input_ids to text.
        max_items (int): How many samples to show.
        decode_labels (bool): Whether to also decode 'labels' if different from input_ids.
    """
    print("\n Inspecting Batch Sample:")
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, type={type(v)}")

    print(f"\n Showing first {max_items} decoded item(s):")
    for i in range(min(max_items, batch["input_ids"].shape[0])):
        print(f"\n--- Sample {i} ---")

        input_ids = batch["input_ids"][i].tolist()
        attention_mask = batch["attention_mask"][i].tolist()
        labels = batch["labels"][i].tolist()

        print(f"input_ids[:10]: {input_ids[:10]} ...")
        print(f"attention_mask[:10]: {attention_mask[:10]} ...")
        print(f"labels[:10]: {labels[:10]} ...")
        print(f"Full input_ids length: {len(input_ids)}")


        if tokenizer:
            try:
                decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
                print(f" Decoded input_ids:\n{decoded_input}")
                if decode_labels and labels != input_ids: #this for causal LM trying to predict the next token id
                    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
                    print(f"  Decoded labels:\n{decoded_labels}")
            except Exception as e:
                print(f" Could not decode sample {i}: {e}")


# --- Main Execution Example (Updated Cache Dir Structure) ---
if __name__ == "__main__":

    #os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    setup_logging(level=logging.INFO)
    # Define cache directories relative to project root
    MAIN_CACHE_DIR = PROJECT_ROOT / "data" / "main_cache"
    MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "model_cache"

    # Create directories if they don't exist (DataModule also does this, but good practice here too)
    MAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Main Cache Dir: {MAIN_CACHE_DIR}")
    print(f"Model Cache Dir: {MODEL_CACHE_DIR}")

    # Define tokenizer name
    # tokenizer_name = "gpt2"
    tokenizer_name = "distilgpt2" # Smaller model for faster testing

    print(f"\nLoading tokenizer '{tokenizer_name}'...")
    # Use trust_remote_code=True if using models requiring it
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(MODEL_CACHE_DIR),download_mode="force_reuse_cache_if_exists")

    # Define dataset configurations
    dataset_configs = {
         # Wikitext: Use default splits
        "wikitext": {},
         # Oscar: Use only a tiny fraction (0.01%) of the English train split for faster testing
        "oscar": {"train_split_percentage":[0.01,0.05]},
        # Bookcorpus: Use default (full train split)
        "bookcorpus": {}
    }

    print("\nInitializing DataModule...")
    # Use num_workers=0 for easier debugging, increase for performance
    num_dataloader_workers = 18 # os.cpu_count() // 2 if os.cpu_count() > 1 else 0

    dm = LightningDataModule(
        tokenizer=tokenizer,
        dataset_configs=dataset_configs,
        batch_size=4, # Keep small for testing
        max_length=128, # Smaller max_length for faster tokenization
        num_workers=num_dataloader_workers,
        cache_dir=str(MAIN_CACHE_DIR) # Pass the *main* cache directory path as string
    )

    print("-----"*90)
    print("----- DEBUG TOKENIZER -----")
    dm.debug_all_datasets(split_label="train", num_samples=3)
    print("----- END OF DEBUG TOKENIZER -----")

    # Output resolved cache paths from the DataModule instance
    print(f"\nRaw data will be cached/checked in: {dm.raw_cache_dir}")
    print(f"Tokenized data will be cached/checked in: {dm.tokenized_cache_dir}\n")

    # --- Run prepare_data ---
    # This will trigger download (via dataset class) and tokenization (via helper)
    print("--- Running prepare_data() ---")
    try:
        dm.prepare_data()
        print("--- prepare_data() finished successfully ---")
    except Exception as e:
        print(f"--- ERROR during prepare_data(): {e} ---", file=sys.stderr)
        # Decide whether to exit or continue
        sys.exit(1)


    # --- Run setup ---
    # This will load the tokenized datasets from the cache
    print("\n--- Running setup('fit') ---")
    try:
        dm.setup("fit") # Setup for training and validation
        print("--- setup('fit') finished successfully ---")
    except Exception as e:
        print(f"--- ERROR during setup('fit'): {e} ---", file=sys.stderr)
        sys.exit(1)

    # --- Test Dataloaders ---
    print("\n--- Testing train_dataloader ---")
    train_loader = dm.train_dataloader()
    if train_loader and len(train_loader.dataset) > 0:
        try:
            start_batch_time = time.time()
            batch = next(iter(train_loader))
            end_batch_time = time.time()
            inspect_batch(batch, tokenizer=tokenizer, max_items=2)
            print(f"Successfully fetched batch from train_dataloader in {end_batch_time - start_batch_time:.2f}s.")
            print("Sample batch details:")
            for k, v in batch.items(): print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            # Decode first sample
            # print("\nDecoded sample (input_ids[0]):")
            # print(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)) # Show special tokens
        except StopIteration:
            print("Train dataloader is empty.")
            raise
        except Exception as e:
            print(f"Error getting batch from train_dataloader: {e}", exc_info=True)
            raise
    else: print("Train dataloader is None or empty.")


    print("\n--- Testing val_dataloader ---")
    val_loader = dm.val_dataloader()
    if val_loader and len(val_loader.dataset) > 0:
        try:
            start_batch_time = time.time()
            val_batch = next(iter(val_loader))
            end_batch_time = time.time()
            inspect_batch(batch, tokenizer=tokenizer, max_items=2)
            print(f"Successfully fetched batch from val_dataloader in {end_batch_time - start_batch_time:.2f}s.")
            print("Sample batch details:")
            for k, v in val_batch.items(): print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        except StopIteration: 
            print("Validation dataloader is empty.") 
            raise
        except Exception as e: 
            print(f"Error getting batch from val_dataloader: {e}", exc_info=True) 
            raise
    else: print("Validation dataloader is None or empty (Note: Oscar/Bookcorpus have no default val split).")

    # Example of setting up for test stage
    print("\n--- Running setup('test') ---")
    try:
        dm.setup("test") # Setup for test stage
        print("--- setup('test') finished successfully ---")
    except Exception as e:
        print(f"--- ERROR during setup('test'): {e} ---", file=sys.stderr)
        raise

    print("\n--- Testing test_dataloader ---")
    test_loader = dm.test_dataloader()
    if test_loader and len(test_loader.dataset) > 0:
         try:
            start_batch_time = time.time()
            test_batch = next(iter(test_loader))
            inspect_batch(test_batch, tokenizer=tokenizer, max_items=2)

            end_batch_time = time.time()
            print(f"Successfully fetched batch from test_dataloader in {end_batch_time - start_batch_time:.2f}s.")
            print("Sample batch details:")
            for k, v in test_batch.items(): print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
         except StopIteration:
            print("Test dataloader is empty.")
            raise
         except Exception as e:
            print(f"Error getting batch from test_dataloader: {e}", exc_info=True)
            raise
    else: print("Test dataloader is None or empty (Note: Oscar/Bookcorpus have no default test split).")








    #if I dont gc the dataloaders, the persistent workers won't terminate!
    print("\n--- Killing lingering workers and processes---")
    del train_loader
    del val_loader
    del test_loader

    import gc
    gc.collect()

    print("\n--- Script finished ---")

