# src/data/my_datasets.py

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, DownloadMode, Dataset as HFDataset
import logging
from pathlib import Path
import time
import shutil
import math # <--- Import math for ceiling calculation
import os # <--- Import os for cpu count in tokenize
from typing import Union, Dict, Any # <--- For type hints
import itertools

from transformers import PreTrainedTokenizer

# Configure logging (using root logger setup elsewhere is also fine)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger for this module


# --- Central Registry ---
dataset_registry = {}

def register_dataset(name):
    logger.info(f"Registering dataset: {name}")
    def decorator(cls):
        if name in dataset_registry:
            logger.warning(f"Dataset '{name}' already registered. Overwriting.")
        dataset_registry[name] = cls
        return cls
    return decorator




def _tokenize_and_save_split(
    raw_split_dataset: HFDataset,
    tokenized_cache_path: Path,
    dataset_name: str,
    split_label: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int, # This is now the BLOCK size for packing
    text_column: str,
    num_proc_to_use: int = 1,
    add_eos_token: bool = True # Option to add EOS between documents
    ):
    """
    Tokenizes, concatenates, and chunks dataset splits for sequence packing,
    then saves the result to disk.
    """
    try:
        logger.info(f"Starting sequence packing for {dataset_name} split '{split_label}' (block size: {max_length})...")

        # --- Step 1: Initial Tokenization (NO padding/truncation) ---
        # Ensure EOS token exists if we plan to add it
        eos_token_id = tokenizer.eos_token_id
        if add_eos_token and eos_token_id is None:
            logger.warning(f"Requested add_eos_token=True but tokenizer has no eos_token_id. Disabling.")
            add_eos_token = False

        def tokenize_function(examples):
            texts = examples.get(text_column)
            # --- Add your text column finding/validation logic here ---
            if texts is None:
                 possible_cols = [k for k, v in examples.items() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str)]
                 if not possible_cols: raise ValueError(f"Cannot find text column '{text_column}' in {dataset_name}")
                 actual_text_column = possible_cols[0]
                 logger.warning(f"Text column '{text_column}' not found for {dataset_name}. Using '{actual_text_column}'.")
                 texts = examples[actual_text_column]
            if not isinstance(texts, list): raise TypeError(f"Expected text data to be a list, got {type(texts)}")
            if not all(isinstance(t, str) for t in texts if t is not None):
                 logger.warning(f"Non-string data found in {dataset_name}, converting to string.")
                 texts = [str(t) if t is not None else "" for t in texts]
            # --- End text column logic ---

            # Tokenize WITHOUT padding/truncation
            # Important: We only want the input_ids for concatenation
            output = tokenizer(texts) # Returns dict with 'input_ids', maybe 'attention_mask'

            # Add EOS token to the end of each document's tokens if requested
            if add_eos_token:
                for i in range(len(output["input_ids"])):
                    output["input_ids"][i].append(eos_token_id)
                    # Adjust attention mask if it exists (though we rebuild it later)
                    # if "attention_mask" in output:
                    #    output["attention_mask"][i].append(1)

            return {"input_ids": output["input_ids"]} # Only return input_ids

        logger.info(f"Step 1: Tokenizing individual examples using {num_proc_to_use} processes...")
        tokenized_dataset = raw_split_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc_to_use,
            remove_columns=raw_split_dataset.column_names # Remove all original columns
        )
        logger.info(f"Tokenization complete. Result has columns: {tokenized_dataset.column_names}")


        # --- Step 2: Grouping/Packing Function ---
        # This function processes the tokenized data (list of input_ids lists)
        def group_texts(examples):
            # Concatenate all lists of token IDs
            concatenated_ids = list(itertools.chain.from_iterable(examples['input_ids']))
            total_length = len(concatenated_ids)

            # Drop the small remainder chunk at the end.
            # Alternatively, you could pad the last chunk, but dropping is simpler.
            if total_length < max_length:
                 logger.warning(f"Total concatenated length ({total_length}) is less than max_length ({max_length}). Cannot create any blocks.")
                 return {"input_ids": [], "labels": [], "attention_mask": []} # Return empty lists

            total_length = (total_length // max_length) * max_length

            # Split by chunks of max_length.
            result_ids = [concatenated_ids[i : i + max_length] for i in range(0, total_length, max_length)]

            # Create labels (for Causal LM, usually shifted input_ids, handled by model/loss)
            # Here we just copy, the model's forward pass handles the shift typically
            result_labels = result_ids.copy()

            # Create attention mask - since we packed, it should be all 1s for full blocks
            result_attention_mask = [[1] * max_length for _ in result_ids]

            return {"input_ids": result_ids, "labels": result_labels, "attention_mask": result_attention_mask}

        logger.info(f"Step 2: Grouping texts into chunks of {max_length}...")
        # --- Step 3: Apply Grouping ---
        # NOTE: Grouping can be memory-intensive. Running without num_proc (num_proc=None or 1)
        # might be necessary if you encounter memory issues, although it will be slower.
        # Adjust num_proc based on your available RAM.
        packing_num_proc = num_proc_to_use # Start with this, reduce if needed
        logger.info(f"Applying grouping map using {packing_num_proc if packing_num_proc else 1} process(es). This may take time and memory.")
        packed_dataset = tokenized_dataset.map(
            group_texts,
            batched=True, # Process multiple tokenized examples at once in group_texts
            num_proc=packing_num_proc,
            # The output columns are determined entirely by group_texts
        )
        logger.info(f"Grouping complete. Packed dataset has {len(packed_dataset)} examples. Columns: {packed_dataset.column_names}")

        # --- Step 4: Save ---
        if not packed_dataset: # Handle empty dataset case
            logger.warning(f"Packed dataset for {dataset_name} split {split_label} is empty. Saving empty dataset structure.")
            # Create dummy structure if needed, or handle appropriately downstream
            # For now, let's just log and let save_to_disk handle empty if it can.

        # Check required columns
        required_cols = {'input_ids', 'labels', 'attention_mask'}
        if not required_cols.issubset(packed_dataset.column_names):
             logger.error(f"Packing failed to produce required columns for {dataset_name} split {split_label}. Got: {packed_dataset.column_names}")
             raise RuntimeError("Packing missing required columns.")

        logger.info(f"Saving packed dataset to {tokenized_cache_path}")
        tokenized_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove potentially incomplete cache if it exists from a failed previous run
        if tokenized_cache_path.exists():
            logger.warning(f"Removing existing cache directory before saving: {tokenized_cache_path}")
            shutil.rmtree(tokenized_cache_path)
        packed_dataset.save_to_disk(str(tokenized_cache_path))
        logger.info(f"Successfully saved packed data for {dataset_name} split {split_label}.")
        return True

    except Exception as e:
        logger.error(f"Failed during sequence packing for {dataset_name} split {split_label}: {e}", exc_info=True)
        # Clean up potentially corrupted cache directory
        if tokenized_cache_path.exists():
            try:
                logger.warning(f"Attempting removal of potentially corrupted {tokenized_cache_path} after error.")
                shutil.rmtree(tokenized_cache_path)
            except Exception as rm_e: logger.error(f"Failed removal of {tokenized_cache_path}: {rm_e}")
        return False # Indicate failure
# === END _tokenize_and_save_split ===


# --- Base Dataset Class ---
class BaseHuggingFaceDataset(Dataset):
    """Base class for handling Hugging Face datasets."""
    dataset_name = None
    dataset_config_name = None
    # Define the base splits available from the source dataset
    available_splits = ["train"] # Default, override in subclasses
    text_column = "text"
    trust_remote_code = False

    # === __init__ (loading tokenized data) - Unchanged ===
    def __init__(self, split: str, specific_tokenized_path: str):
        self.split = split
        self.tokenized_path = Path(specific_tokenized_path)
        self.dataset = None
        if not self.tokenized_path.exists():
            raise FileNotFoundError(
                f"Packed dataset for split '{self.split}' not found at {self.tokenized_path}"
            )
        try:
            logger.info(
                f"Loading packed dataset for split '{self.split}' from {self.tokenized_path}"
            )
            self.dataset = load_from_disk(str(self.tokenized_path), keep_in_memory=False)
            required_cols = {'input_ids', 'labels', 'attention_mask'}  # Check for packed columns
            if not required_cols.issubset(self.dataset.column_names):
                missing = required_cols - set(self.dataset.column_names)
                raise ValueError(
                    f"Loaded packed dataset {self.tokenized_path} missing required columns: {missing}"
                )
            logger.info(
                f"Packed dataset loaded from {self.tokenized_path} with columns {self.dataset.column_names}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load packed dataset from {self.tokenized_path} for split '{self.split}': {e}",
                exc_info=True,
            )
            raise e
    # === END __init__ ===

    # === CORRECTED download_raw_split ===
    @classmethod
    def download_raw_split(cls,
                           split_info: dict, # Expects dict from get_split_names
                           raw_cache_dir: str,
                           download_mode: DownloadMode = DownloadMode.REUSE_DATASET_IF_EXISTS,
                           **kwargs):
        """
        Downloads (or loads from cache) the raw data for a base split and
        applies percentage slicing afterwards if specified using .select().
        """
        if cls.dataset_name is None: raise ValueError("`dataset_name` must be set.")

        # split name is the key
        hf_split_name = split_info.get("hf_split_name") # e.g., "train", "validation"
        percentage_fraction = split_info.get("percentage") # e.g., 0.001 or None

        if hf_split_name is None: raise ValueError("split_info dict missing 'hf_split_name'.")

        load_args = [cls.dataset_name]
        if cls.dataset_config_name: load_args.append(cls.dataset_config_name)
        # Use trust_remote_code specified in the class definition by default
        effective_trust_remote_code = kwargs.get('trust_remote_code', cls.trust_remote_code)

        logger.info(f"Attempting load/download for {cls.dataset_name} (config: {cls.dataset_config_name or 'default'}) "
                    f"BASE split '{hf_split_name}' (trust_remote_code={effective_trust_remote_code}) using cache {raw_cache_dir}")
        start_time = time.time()
        try:
            # --- Step 1: Load the BASE split (e.g., "train") ---
            # Use the correct DownloadMode enum value
            #load_dataset is lazy by default, it will load when i ask for them
            base_split_dataset = load_dataset( 
                *load_args,
                split=hf_split_name, # <--- Pass the base split name ONLY
                cache_dir=str(raw_cache_dir),
                trust_remote_code=effective_trust_remote_code,
                download_mode=download_mode, # Use the passed argument
                
            )
            load_time = time.time() - start_time
            logger.info(f"Raw base split '{hf_split_name}' loaded/verified in {load_time:.2f}s. "
                        f"Base size: {len(base_split_dataset)} examples.")

            print("****"*90)
            print(percentage_fraction )
            # --- Step 2: Apply Percentage Slicing AFTER lazy loading (if requested) ---
            if percentage_fraction is not None:
                if len(base_split_dataset) == 0:
                    logger.warning(f"Base split '{hf_split_name}' is empty. Returning as-is.")
                    return base_split_dataset

                # Handle either (start, end) or just end
                if isinstance(percentage_fraction, tuple):
                    start_pct, end_pct = percentage_fraction
                    start_idx = int(len(base_split_dataset) * start_pct)
                    end_idx = int(len(base_split_dataset) * end_pct)
                    end_idx = min(end_idx, len(base_split_dataset)) # Avoid overflow
                    if start_idx >= end_idx:
                        raise ValueError(f"Invalid slicing range: start={start_idx}, end={end_idx}")
                    logger.info(f"Slicing from {start_pct*100:.1f}% to {end_pct*100:.1f}% -> indices {start_idx}:{end_idx}")
                    sliced_dataset = base_split_dataset.select(range(start_idx, end_idx))

                elif isinstance(percentage_fraction, float):
                    num_examples = max(1, math.ceil(len(base_split_dataset) * percentage_fraction))
                    sliced_dataset = base_split_dataset.select(range(num_examples))
                    logger.info(f"Selecting first {percentage_fraction*100:.1f}% ({num_examples} examples)")

                else:
                    raise ValueError(f"Unsupported percentage format: {percentage_fraction}")

                return sliced_dataset

            else:
                 # No slice requested or invalid percentage, return the full base split
                 logger.info(f"No percentage slice requested or applied for '{hf_split_name}'. Using full base split.")
                 return base_split_dataset

        except Exception as e:
            logger.error(f"Failed during load/download/select for raw split '{hf_split_name}' "
                         f"of {cls.dataset_name}: {e}", exc_info=True)
            raise # Re-raise the exception
    # === END CORRECTED download_raw_split ===


    # === CORRECTED BASE get_split_names ===
    @classmethod
    def get_split_names(cls, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary mapping split labels (e.g., 'train')
        to processing info (e.g., {'hf_split_name': 'train', 'percentage': None}).
        Subclasses override this to add percentage logic if needed.
        """
        split_info_dict = {}
        for split_label in cls.available_splits:
            split_info_dict[split_label] = {
                "hf_split_name": split_label, # Default: use the label as the HF name
                "percentage": None             # Default: no percentage slice
            }
        return split_info_dict
    # === END CORRECTED BASE get_split_names ===


    # === __len__ and __getitem__ (Unchanged) ===
    def __len__(self):
        if self.dataset is None:
            logger.error(f"Dataset is None in __len__ for {self.dataset_name} split {self.split}.")
            return 0
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset is None:
            raise RuntimeError(f"Dataset not loaded for {self.dataset_name} split {self.split}.")
        try:
            # The item loaded from disk already has the correct packed format
            item = self.dataset[idx]

            # Directly convert to tensors
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
            labels = torch.tensor(item["labels"], dtype=torch.long)

            # Optional: Verify length (should always be max_length now)
            # assert len(input_ids) == self.max_length # Requires storing max_length in __init__ if needed

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        except Exception as e:
             logger.error(f"Error retrieving packed item {idx} for {self.dataset_name} split {self.split}: {e}", exc_info=True)
             # Log the item keys if it's a KeyError
             if isinstance(e, KeyError): logger.error(f"Item keys: {item.keys()}")
             raise
    # === END __len__ and __getitem__ ===


# --- Specific Dataset Implementations ---

@register_dataset("wikitext")
class WikiTextDataset(BaseHuggingFaceDataset):
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    # Override available_splits from base class
    available_splits = ["train", "validation", "test"]
    text_column = "text"
    trust_remote_code = False


# === CORRECTED OscarDataset (inherits base get_split_names logic, then modifies) ===
@register_dataset("oscar")
class OscarDataset(BaseHuggingFaceDataset):
    dataset_name = "oscar"
    dataset_config_name = "unshuffled_deduplicated_en"
    available_splits = ["train"] # Only train is typically available
    text_column = "text"
    trust_remote_code = True # Needs True

    @classmethod
    def get_split_names(cls, **kwargs) -> Dict[str, Dict[str, Any]]:
        splits_info = super().get_split_names(**kwargs)

        raw = kwargs.get("train_split_percentage")
        train_split_percentage = ([float(x) for x in raw.split("-")] if isinstance(raw, str) else raw)

        if "train" in splits_info and train_split_percentage is not None:
            try:
                if isinstance(train_split_percentage, (float, int)):
                    percentage_range = (0.0, float(train_split_percentage))
                elif isinstance(train_split_percentage, list) and len(train_split_percentage) == 2:
                    percentage_range = (float(train_split_percentage[0]), float(train_split_percentage[1]))
                    if not (0.0 <= percentage_range[0] < percentage_range[1] <= 1.0):
                        raise ValueError("Percentage range must be within (0.0 <= start < end <= 1.0).")
                else:
                    raise ValueError(f"train_split_percentage must be a float or a list of two floats. received:{train_split_percentage}")
                    

                splits_info["train"]["percentage"] = percentage_range
                logger.info(f"OSCAR will use percentage slice: {percentage_range}")
            except Exception as e:
                logger.error(f"Invalid train_split_percentage: {e}")
                raise

        return splits_info

# === END CORRECTED OscarDataset ===


@register_dataset("bookcorpus")
class BookCorpusDataset(BaseHuggingFaceDataset):
    dataset_name = "bookcorpus"
    dataset_config_name = None
    available_splits = ["train"]
    text_column = "text"
    # Update trust_remote_code based on previous error logs
    trust_remote_code = True # <-- Set based on error messages


@register_dataset("openwebtext")
class OpenWebTextDataset(BaseHuggingFaceDataset):
    dataset_name = "openwebtext"
    dataset_config_name = None
    available_splits = ["train"]
    text_column = "text"
    # Update trust_remote_code based on previous error logs
    trust_remote_code = True # <-- Set based on error messages


# ------------------------------------------------------------------
# Large-scale web corpora (stream-friendly)
# ------------------------------------------------------------------

@register_dataset("cerebras-slim_pajama")
class SlimPajamaDataset(BaseHuggingFaceDataset):
    """
    cerebras/SlimPajama-627B  –  already deduped and doc-filtered
    """
    dataset_name = "cerebras/SlimPajama-627B"
    dataset_config_name = "default"        # <- must be a string, not None
    available_splits = ["train"]
    text_column = "text"
    trust_remote_code = False              # no custom code


@register_dataset("refinedweb")
class RefinedWebDataset(BaseHuggingFaceDataset):
    """
    RefinedWeb / CC-Net 2023-12 snapshot (English only by default)
    """
    dataset_name = "refinedweb"
    dataset_config_name = "default"
    available_splits = ["train"]
    text_column = "text"
    trust_remote_code = True               # HF script uses remote_code


@register_dataset("dolma")
class DolmaDataset(BaseHuggingFaceDataset):
    """
    Dolma v1 – 3 T tokens, mixture of web, books, code, papers.
    We keep the English split for now.
    """
    dataset_name = "allenai/dolma"
    dataset_config_name = "dolma-v1-en"    # other configs: -multi, -code, …
    available_splits = ["train"]
    text_column = "text"
    trust_remote_code = True


@register_dataset("redpajama")
class RedPajamaDataset(BaseHuggingFaceDataset):
    """
    RedPajama v1.2 – replica of LLaMA mix.
    """
    dataset_name = "togethercomputer/RedPajama-Data-1T"
    dataset_config_name = "default"
    available_splits = ["train"]
    text_column = "text"
    trust_remote_code = True

# Add other dataset classes following the same pattern...


