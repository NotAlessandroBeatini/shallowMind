import torch
from torch.utils.data import Dataset # Removed DataLoader, ConcatDataset imports - not used here
from datasets import load_dataset, load_from_disk,DownloadMode ,Dataset as HFDataset # Removed concatenate_datasets
import logging
from pathlib import Path
import time # For retries maybe (not currently implemented but useful import)
import shutil # For error cleanup
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Central Registry ---
dataset_registry = {}

def register_dataset(name):
    logging.info(f"Registering dataset: {name}")
    def decorator(cls):
        if name in dataset_registry:
            logging.warning(f"Dataset '{name}' already registered. Overwriting.")
        dataset_registry[name] = cls
        return cls
    return decorator


def _tokenize_and_save_split(
    raw_split_dataset: HFDataset,
    tokenized_cache_path: Path,
    dataset_name: str, # For logging
    split_label: str,   # For logging
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_column: str,
    num_proc_to_use: int = 1): #Processes to use 
    """
    Tokenizes a given raw dataset split and saves it to disk.
    Assumes the raw_split_dataset is already loaded. Uses multiprocessing.

    Args:
        raw_split_dataset (HFDataset): The loaded raw dataset split.
        tokenized_cache_path (Path): The path where the tokenized dataset should be saved.
        dataset_name (str): Name of the dataset for logging.
        split_label (str): Name of the split label for logging.
        tokenizer: The tokenizer instance.
        max_length (int): Max sequence length for padding/truncation.
        text_column (str): The name of the column containing the text data.

    Returns:
        bool: True if tokenization was successful, False otherwise.
    """
    try:
        logging.info(f"Tokenizing raw data for {dataset_name} split '{split_label}' using multiple processes...")

        def tokenize_function(examples):
            texts = examples.get(text_column)
            if texts is None:
                # Try to find another text column if the primary one isn't present
                possible_cols = [k for k, v in examples.items() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str)]
                if not possible_cols:
                    raise ValueError(f"Could not find text column '{text_column}' or any suitable string list column in dataset features: {list(examples.keys())}")
                else:
                    actual_text_column = possible_cols[0]
                    logging.warning(f"Text column '{text_column}' not found for {dataset_name}. Using '{actual_text_column}' instead.")
                    texts = examples[actual_text_column]

            if not isinstance(texts, list):
                 raise TypeError(f"Expected text data in column '{text_column}' (or fallback) to be a list, but got {type(texts)}")
            if not all(isinstance(t, str) for t in texts if t is not None):# Check if all elements are strings, ignoring None
                 logging.warning(f"Non-string data found in text column '{text_column}' for {dataset_name}. Attempting to convert to string.")
                 texts = [str(t) if t is not None else "" for t in texts]
                 #raise RuntimeError 

            # Ensure tokenizer has pad token ID (should be handled by DataModule init, but double check)
            if tokenizer.pad_token_id is None:
                 if tokenizer.eos_token_id is not None:
                      logging.warning("Tokenizer has no pad_token_id in tokenize_function, setting to eos_token_id")
                      tokenizer.pad_token_id = tokenizer.eos_token_id
                 else: # This case should ideally be prevented by DataModule init check
                      raise ValueError("Tokenizer has no pad_token_id and no eos_token_id. Cannot proceed.")


            return tokenizer(
                texts, padding="max_length", truncation=True, max_length=max_length
            )

        # Determine columns to remove - keep only tokenizer outputs ('input_ids', 'attention_mask')
        # Provide the original columns to `remove_columns`. `map` will automatically keep the new columns created by the function.
        cols_to_remove = list(raw_split_dataset.column_names)
        logging.debug(f"Columns in raw dataset: {cols_to_remove}")
        logging.debug(f"Will attempt to remove columns: {cols_to_remove} during tokenization map.")

        # Determine number of processes (use slightly less than all cores to leave resources)
        
        logging.info(f"Using num_proc={num_proc_to_use} for tokenization map") # (available CPUs: {num_cpus}).")


        tokenized = raw_split_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=cols_to_remove,
            num_proc=num_proc_to_use 
        )

        logging.info(f"Tokenization complete. Columns after tokenization: {tokenized.column_names}")
        if 'input_ids' not in tokenized.column_names or 'attention_mask' not in tokenized.column_names:
             logging.error(f"Tokenization failed to produce 'input_ids' or 'attention_mask' for {dataset_name} split {split_label}. Columns present: {tokenized.column_names}")
             raise RuntimeError("Tokenization did not produce expected columns.")


        logging.info(f"Saving tokenized dataset to {tokenized_cache_path}")
        tokenized_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(str(tokenized_cache_path))
        logging.info(f"Successfully saved tokenized data for {dataset_name} split {split_label}.")
        return True

    except Exception as e:
        logging.error(f"Failed to tokenize and save {dataset_name} split {split_label}: {e}", exc_info=True)
        # Clean up potentially corrupted partial save
        if tokenized_cache_path.exists():
            try:
                logging.warning(f"Attempting to remove potentially corrupted tokenized data at {tokenized_cache_path}")
                shutil.rmtree(tokenized_cache_path)
            except Exception as rm_e:
                logging.error(f"Failed to remove corrupted data at {tokenized_cache_path}: {rm_e}")
        return False





# --- Base Dataset Class ---
class BaseHuggingFaceDataset(Dataset):
    """Base class for handling Hugging Face datasets."""
    dataset_name = None           # e.g., "wikitext"
    dataset_config_name = None    # e.g., "wikitext-2-raw-v1"
    available_splits = []         # e.g., ["train", "validation", "test"]
    text_column = "text"          # Default column name for text data
    trust_remote_code = False     # Set to True for datasets like 'oscar'


    def __init__(self, split: str, tokenized_data_dir: str):
        """
        Loads the *pre-tokenized* data split from disk.
        Assumes prepare_data() in the DataModule has already run successfully
        for the correct tokenizer.

        Args:
            split (str): The split label (e.g., "train", "validation").
            tokenized_data_dir (str): The path to the directory containing
                                      tokenized splits for a SPECIFIC tokenizer.
                                      (e.g., .../cache/tokenized/gpt2__)
        """
        self.split = split
        # This is the tokenizer-specific base directory, e.g., .../tokenized/gpt2__
        self.tokenized_data_dir = Path(tokenized_data_dir)

        # Construct the unique ID for the dataset config
        config_suffix = self.dataset_config_name or 'default'
        self.dataset_id = f"{self.dataset_name}_{config_suffix}"

        # Construct the final path to the specific split folder WITHIN the tokenizer dir
        # e.g., .../tokenized/gpt2__/wikitext_wikitext-2-raw-v1_train
        self.tokenized_path = self.tokenized_data_dir / f"{self.dataset_id}_{split}"
        self.dataset = None

        if not self.tokenized_path.exists():
            # This error means prepare_data failed or didn't create this split for this tokenizer
            raise FileNotFoundError(
                f"Tokenized dataset for {self.dataset_name} (config: {config_suffix}) split '{split}' "
                f"not found at expected path: {self.tokenized_path}. "
                f"Ensure prepare_data() ran successfully FOR THE CORRECT TOKENIZER and completed tokenization for this split."
            )
        try:
            logging.info(f"Loading tokenized dataset from {self.tokenized_path}")
            # keep_in_memory=False can be useful for very large datasets if RAM is limited
            self.dataset = load_from_disk(str(self.tokenized_path), keep_in_memory=False)
            # Verify necessary columns exist after loading
            required_cols = ['input_ids', 'attention_mask']
            if not all(col in self.dataset.column_names for col in required_cols):
                 missing_cols = [col for col in required_cols if col not in self.dataset.column_names]
                 raise ValueError(f"Loaded dataset from {self.tokenized_path} is missing required columns: {missing_cols}. "
                                  f"Columns found: {self.dataset.column_names}. This might indicate a corrupted cache or an issue during tokenization.")
            logging.info(f"Successfully loaded tokenized dataset for {self.dataset_name} split '{split}'. Features: {self.dataset.features}")
        except Exception as e:
            logging.error(f"Failed to load dataset from disk at {self.tokenized_path}: {e}", exc_info=True)
            # You might want to attempt removing the potentially corrupted cache here
            # try:
            #     logging.warning(f"Attempting to remove potentially corrupted cache folder: {self.tokenized_path}")
            #     shutil.rmtree(self.tokenized_path)
            # except Exception as rm_e:
            #     logging.error(f"Failed to remove corrupted cache folder {self.tokenized_path}: {rm_e}")
            raise e # Re-raise the exception


    @classmethod
    def download_raw_split(cls, split_name_hf: str, raw_cache_dir: str, **kwargs):
        """
        Downloads (or loads from cache) the raw data for a specific Hugging Face split name.

        Args:
            split_name_hf (str): The split name recognized by Hugging Face (e.g., 'train', 'validation', 'train[:1%]').
            raw_cache_dir (str): The directory to cache the raw downloaded data.
            **kwargs: Additional arguments passed to load_dataset (potentially overriding class defaults).

        Returns:
            datasets.Dataset: The loaded raw dataset split.

        Raises:
            Exception: If loading/downloading fails.
        """
        if cls.dataset_name is None:
             raise ValueError("`dataset_name` must be set in the subclass.")

        load_args = [cls.dataset_name]
        if cls.dataset_config_name:
            load_args.append(cls.dataset_config_name)

        effective_trust_remote_code = kwargs.get('trust_remote_code', cls.trust_remote_code)

        logging.info(f"Attempting to load/download raw data for {cls.dataset_name} (config: {cls.dataset_config_name or 'default'}) "
                     f"split '{split_name_hf}' using cache {raw_cache_dir} "
                     f"(trust_remote_code={effective_trust_remote_code})")
        start_time = time.time()
        try:
            raw_split_dataset = load_dataset(
                *load_args,
                split=split_name_hf,
                cache_dir=str(raw_cache_dir), # Ensure cache_dir is a string
                trust_remote_code=effective_trust_remote_code,
                download_mode= "reuse_cache_if_exists" #DownloadMode.REUSE_DATASET_IF_EXISTS #"reuse_cache_if_exists"  # default behavior
            )
                # download_mode='force_redownload' # Optional: for debugging download issues
            
            load_time = time.time() - start_time
            logging.info(f"Raw data for {cls.dataset_name} split '{split_name_hf}' loaded/verified successfully in {load_time:.2f}s. "
                         f"Dataset size: {len(raw_split_dataset)} examples. Features: {raw_split_dataset.features}")
            return raw_split_dataset
        except Exception as e:
            logging.error(f"Failed to load/download raw split '{split_name_hf}' for {cls.dataset_name}: {e}", exc_info=True)
            raise # Re-raise the exception to be caught by prepare_data

    @classmethod
    def get_split_names(cls, **kwargs):
        """
        Returns a dictionary mapping internal split labels (e.g., 'train')
        to Hugging Face split strings (e.g., 'train[:1%]', 'validation').
        Allows customization via kwargs. Subclasses override this as needed.
        """
        # Default implementation: map available splits directly
        return {split: split for split in cls.available_splits}

    def __len__(self):
        if self.dataset is None:
            # This indicates a failure during __init__
            logging.error(f"Dataset is None in __len__ for {self.dataset_name} split {self.split}. Initialization failed.")
            return 0
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset is None:
            raise RuntimeError(f"Dataset not loaded for {self.dataset_name} split {self.split}. Check initialization logs.")

        try:
            item = self.dataset[idx]
            # Ensure keys exist before creating tensors
            if "input_ids" not in item or "attention_mask" not in item:
                 raise KeyError(f"Item at index {idx} in {self.dataset_name} split {self.split} is missing 'input_ids' or 'attention_mask'. Item keys: {item.keys()}")

            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
            # For causal LM, labels are usually the same as input_ids
            labels = input_ids.clone()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        except Exception as e:
             logging.error(f"Error retrieving item at index {idx} for {self.dataset_name} split {self.split}: {e}", exc_info=True)
             # Decide how to handle this: re-raise, return None, return dummy data?
             # Re-raising is often best during development.
             raise


# --- Specific Dataset Implementations ---

@register_dataset("wikitext")
class WikiTextDataset(BaseHuggingFaceDataset):
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    available_splits = ["train", "validation", "test"]
    text_column = "text"
    trust_remote_code = False # Wikitext doesn't require it


@register_dataset("oscar")
class OscarDataset(BaseHuggingFaceDataset):
    dataset_name = "oscar"
    # Example using a smaller subset for English
    dataset_config_name = "unshuffled_deduplicated_en"
    available_splits = ["train"] # OSCAR typically only has a train split
    text_column = "text"
    trust_remote_code = True # OSCAR requires trust_remote_code=True

    @classmethod
    def get_split_names(cls, train_split_percentage=None, **kwargs):
        """ Customizes the 'train' split name based on percentage. """
        splits = {}
        train_split_name = "train" # Default HF split name
        if train_split_percentage is not None:
            try:
                percentage = float(train_split_percentage)
                if not 0 < percentage <= 100:
                    raise ValueError("train_split_percentage must be between 0 (exclusive) and 100 (inclusive)")
                # Format for Hugging Face slicing API
                train_split_name = f"train[:{percentage:.2f}%]"
                logging.info(f"Using {percentage:.2f}% of OSCAR train split: '{train_split_name}'")
            except ValueError as e:
                 logging.error(f"Invalid train_split_percentage '{train_split_percentage}': {e}. Using full train split.")
                 train_split_name = "train" # Fallback to full split

        splits["train"] = train_split_name
        # Note: Oscar doesn't have standard validation/test splits.
        # You might need custom logic here if you want to *create* them from train.
        return splits


@register_dataset("bookcorpus")
class BookCorpusDataset(BaseHuggingFaceDataset):
    dataset_name = "bookcorpus"
    dataset_config_name = None # Bookcorpus doesn't have configs
    available_splits = ["train"]
    text_column = "text"
    trust_remote_code = False # Bookcorpus doesn't require it

