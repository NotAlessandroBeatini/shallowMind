# Project Workflow

This short guide outlines the typical steps to get a training run going.

## 1. Download data

The script `src/utils/download_datasets.py` fetches and caches the raw datasets defined in `DATASETS_TO_DOWNLOAD`. Example:

```bash
python src/utils/download_datasets.py --cache-dir data/raw_cache
```

Dataset locations are resolved relative to the project root.

### Dataset caching rules

`download_datasets.py` relies on the caching behaviour of `datasets.load_dataset`.
When called with the default `download_mode=REUSE_DATASET_IF_EXISTS`, the
library checks for a preprocessed dataset under
`<cache_dir>/raw/<builder_key>/<config>/<fingerprint>/`. If that folder exists
and contains valid Arrow files, they are loaded directly and no network access
occurs. If the folder is missing, `load_dataset` verifies that the original
source files are present in `<cache_dir>/raw/downloads`; any missing or
corrupted file triggers a fresh download and the dataset is rebuilt, producing a
new fingerprint directory.

Passing `--force-redownload` to the script sets
`download_mode=DownloadMode.FORCE_REDOWNLOAD`, which always re-downloads the
sources and regenerates the Arrow files, regardless of any existing cache.

## 2. Prepare and tokenize

`src/data/data_manager.py` implements `LightningDataModule`. It downloads raw splits (if needed) and creates tokenized chunks that live under `data/main_cache`. Calling `prepare_data()` will download and tokenize once on the rank‑0 process.

```python
from src.data.data_manager import LightningDataModule
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dm = LightningDataModule(tokenizer=tokenizer,
                         dataset_configs={"wikitext": {}},
                         batch_size=8,
                         max_length=128)
dm.prepare_data()
```

## 3. Train

Training is driven by `src/train/train.py` and the YAML config in `config/train_config.yaml`. Update that file as needed then run:

```bash
python src/train/train.py
```

If `ray.use_ray` is true, Ray will launch multiple workers with the strategy specified under `strategy:`.

## 4. Logs and caches

- **Training logs:** by default `utils.setup_logging` writes files in `logs/` under the project root. Each run gets a timestamped `app_<time>.log`. Pass `log_dir` to change this location.
- **Dataset download logs:** `src/utils/download_datasets.py` saves logs to `data/logs/download_datasets_<n>.log`. Use `--log-dir` to override.
- **Raw datasets:** downloaded splits reside in `data/main_cache/raw/`.
- **Tokenized datasets:** chunks live in `data/main_cache/tokenized/<tokenizer>/`.
- **Model cache:** Hugging Face models are stored in `data/model_cache`.

## 5. Notebook quick test

A simple sanity‑check notebook lives at `src/train/train.ipynb` and can be used to experiment with the data module or models.
