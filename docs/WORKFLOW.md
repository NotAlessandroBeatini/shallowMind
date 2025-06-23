# Project Workflow

This short guide outlines the typical steps to get a training run going.

## 1. Download data

The script `src/utils/download_datasets.py` fetches and caches the raw datasets defined in `DATASETS_TO_DOWNLOAD`. Example:

```bash
python src/utils/download_datasets.py --cache-dir data/raw_cache
```

Dataset locations are resolved relative to the project root.

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

Logs go to the `logs/` directory (configured via `utils.setup_logging`). Tokenized datasets are stored in `data/main_cache` and model downloads in `data/model_cache` by default.

## 5. Notebook quick test

A simple sanity‑check notebook lives at `src/train/train.ipynb` and can be used to experiment with the data module or models.

