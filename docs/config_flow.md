# Configuration Flow

This document traces how `src/train/train.py` consumes `config/train_config.yaml` and passes the settings to the various Lightning components.

## 1. Mapping YAML keys to `train.py`

The main entry point loads the YAML and extracts its top level keys:

```
215  run_settings = config.get("run", {})
216  model_settings = config.get("model", {})
217  trainer_params_from_config = config.get("trainer", {})
218  ray_settings = config.get("ray", {})
```

Each section is forwarded through the Ray worker configuration:

```
256  train_loop_config = {
257      "project_root_str": str(PROJECT_ROOT),
258      "global_seed": run_settings.get("seed", 42),
259      "run_name_base": run_settings.get("run_name", f"{model_settings['class_name']}-ray"),
260      "model_config": model_settings,
261      "datamodule_config": config.get("datamodule", {}),
262      "trainer_params": trainer_params_from_config,
263      "strategy_config": config.get("strategy", {}),
264      "run_settings": run_settings,
}
```

Inside `train_func_for_ray`, those entries are accessed directly:

```
108  cfg_model = train_loop_worker_config["model_config"]
109  cfg_datamodule = train_loop_worker_config["datamodule_config"]
110  cfg_trainer_params = train_loop_worker_config["trainer_params"]
111  cfg_strategy = train_loop_worker_config["strategy_config"]
112  cfg_run_settings = train_loop_worker_config["run_settings"]
```

## 2. Resolving relative paths

`train.py` provides a helper to resolve any path in the YAML relative to the project root:

```
83  def resolve_path_in_config(config_value: str, base_root: Path) -> Path:
84      path = Path(config_value)
85      return (base_root / path).resolve() if not path.is_absolute() else path.resolve()
```

It is used when instantiating the model and datamodule so that the `tokenizer_name_or_path` and dataset `cache_dir` can be specified relative to the repository root:

```
118  model_cls = get_model_class(cfg_model["module_path"], cfg_model["class_name"])
119  model_kwargs = cfg_model.get("kwargs", {}).copy()
120  if "tokenizer_name_or_path" in model_kwargs:
121       model_kwargs["tokenizer_name_or_path"] = str(resolve_path_in_config(model_kwargs["tokenizer_name_or_path"], cfg_project_root))
...
126  dm_kwargs = cfg_datamodule.get("kwargs", {}).copy()
127  if "cache_dir" in dm_kwargs:
128      dm_kwargs["cache_dir"] = str(resolve_path_in_config(dm_kwargs["cache_dir"], cfg_project_root))
```

The resolved paths are then passed to `LightningDataModule` and later to `RayDeepSpeedStrategy` for the DeepSpeed configuration file:

```
142      ds_path = resolve_path_in_config(
143          cfg_strategy.get("config_path", "config/ds_default.json"),
144          cfg_project_root
145      )
```

## 3. Passing settings to Lightning components

* **LightningDataModule** – created with `dm_kwargs` above. Within `LightningDataModule.__init__` the cache directory is again resolved relative to the project root if not absolute:

```
69      self.cache_dir = Path(cache_dir)
70      if not self.cache_dir.is_absolute():
71          self.cache_dir = PROJECT_ROOT / self.cache_dir
```

* **RayDeepSpeedStrategy** – constructed using the resolved DeepSpeed JSON or inline dict:

```
138      ds_stage = cfg_strategy.get("stage", 3)
140      deepspeed_cfg = cfg_strategy["config_dict"]   # if provided inline
142      ds_path = resolve_path_in_config(
143          cfg_strategy.get("config_path", "config/ds_default.json"),
144          cfg_project_root
145      )
150      pl_strategy_object = RayDeepSpeedStrategy(
151          stage=ds_stage,
152          config=deepspeed_cfg
      )
```

* **Lightning Trainer** – receives parameters from `cfg_trainer_params` when the trainer is created.

## 4. Dataset processing

`LightningDataModule` stores the `dataset_configs` from the YAML and uses them during `prepare_data` and `setup`:

```
47          dataset_configs: dict,
55          self.dataset_configs = dataset_configs
56          self.dataset_names = list(dataset_configs.keys())
```

During `prepare_data`, each dataset name is looked up in the registry and its config passed to `get_split_names`, `download_raw_split` and the tokenization helper. Example lines:

```
121              dataset_cls = dataset_registry[name]
122              specific_config = self.dataset_configs.get(name, {})
126                  splits_to_process = dataset_cls.get_split_names(**specific_config)
139                      raw_split_dataset = dataset_cls.download_raw_split(
140                          split_info=split_info,
141                          raw_cache_dir=str(self.raw_cache_dir),
```

`setup()` then loads the tokenized splits using the same configs:

```
219              dataset_cls = dataset_registry[name]
220              specific_config = self.dataset_configs.get(name, {})
224                  splits_info = dataset_cls.get_split_names(**specific_config)
254                      tokenized_path = self._get_tokenized_split_path(dataset_cls, split_label, split_detail)
262                      dataset_instance = dataset_cls(
263                          split=split_label,
264                          specific_tokenized_path=str(tokenized_path)
```

## 5. DeepSpeed configuration

The YAML `strategy.config_path` points to `config/ds_config.json`. `train_func_for_ray` resolves that path and passes it to `RayDeepSpeedStrategy`. The JSON itself defines optimizer and ZeRO settings used by DeepSpeed:

```
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,
  "gradient_clipping": 1.0,
  "fp16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}
```

When the strategy is instantiated, this file is read by DeepSpeed to configure the distributed optimizer and memory behaviour.
