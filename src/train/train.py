# train.py (Aligned with the provided Ray PDF documentation - TorchTrainer + train_func)

import os
import sys
import yaml
import logging
from pathlib import Path
from importlib import import_module
import time
import re

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Global PROJECT_ROOT ---
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

# --- PyTorch Lightning and Ray Imports ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor # Other PL Callbacks managed in train_func
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy # For DeepSpeed

try:
    import ray
    import ray.train.lightning # Import the module to access its contents
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    # Strategies and helpers are accessed via ray.train.lightning.<ClassName>
    HAS_RAY = True
    logging.info(f"Ray {ray.__version__} and Ray Train components successfully imported.")
except ImportError as e:
    HAS_RAY = False
    logging.warning(f"Ray components import failed: {e}. Ray-distributed training will be unavailable.")
    TorchTrainer, ScalingConfig, RunConfig, CheckpointConfig, ray_lightning = None, None, None, None, None


# --- Your Project-Specific Imports ---
try:
    from src.data.data_manager import LightningDataModule
    from src.utils.utils import setup_logging

except ImportError as e:
    logging.error(f"Failed to import LightningDataModule or setup_logging: {e}")
    raise


# --- Helper Functions ---
def load_config(config_path: Path) -> dict:
    logging.info(f"Attempting to load configuration from: {config_path}")
    if not config_path.exists():
        logging.error(f"CRITICAL: Configuration file not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing configuration file {config_path}: {e}")
        raise

def resolve_path_in_config(config_value: str, base_root: Path) -> Path:
    path = Path(config_value)
    return (base_root / path).resolve() if not path.is_absolute() else path.resolve()

def get_model_class(module_path: str, class_name: str):
    try:
        module = import_module(module_path)
        model_cls = getattr(module, class_name)
        logging.info(f"Successfully imported model class '{class_name}' from '{module_path}'")
        return model_cls
    except ImportError as e: logging.error(f"Could not import module '{module_path}': {e}"); raise
    except AttributeError: logging.error(f"Could not find class '{class_name}' in module '{module_path}'"); raise

# --- Training Function for Ray Workers (as per Ray Docs) ---
def train_func_for_ray(train_loop_worker_config: dict):
    """
    This function is executed by each Ray worker.
    It sets up the PyTorch Lightning model, data, and trainer, then calls .fit().
    """
    worker_context = ray.train.get_context()
    worker_rank = worker_context.get_world_rank()
    logging.info(f"[Worker {worker_rank}] Initializing training function.")

    # Extract configs
    cfg_project_root = Path(train_loop_worker_config["project_root_str"])
    cfg_model = train_loop_worker_config["model_config"]
    cfg_datamodule = train_loop_worker_config["datamodule_config"]
    cfg_trainer_params = train_loop_worker_config["trainer_params"] # Args for pl.Trainer
    cfg_strategy = train_loop_worker_config["strategy_config"]
    cfg_run_settings = train_loop_worker_config["run_settings"] # General run settings

    # Seed
    pl.seed_everything(cfg_run_settings.get("seed", 42) + worker_rank, workers=True)

    # 1. Instantiate Model (LightningModule)
    model_cls = get_model_class(cfg_model["module_path"], cfg_model["class_name"])
    model_kwargs = cfg_model.get("kwargs", {}).copy()
    if "tokenizer_name_or_path" in model_kwargs:
         model_kwargs["tokenizer_name_or_path"] = str(resolve_path_in_config(model_kwargs["tokenizer_name_or_path"], cfg_project_root))
    model = model_cls(**model_kwargs)
    logging.info(f"[Worker {worker_rank}] Model '{cfg_model['class_name']}' instantiated.")

    # 2. Instantiate DataModule
    dm_kwargs = cfg_datamodule.get("kwargs", {}).copy()
    if "cache_dir" in dm_kwargs:
        dm_kwargs["cache_dir"] = str(resolve_path_in_config(dm_kwargs["cache_dir"], cfg_project_root))
    datamodule = LightningDataModule(tokenizer=model.tokenizer, **dm_kwargs)
    logging.info(f"[Worker {worker_rank}] LightningDataModule instantiated.")

    # 3. Configure PyTorch Lightning Trainer for this worker
    # Strategy
    strategy_name = cfg_strategy.get("name", "deepspeed").lower()
    pl_strategy_object = None
    if strategy_name == "deepspeed":
        ds_config_path_str = cfg_strategy.get("config_path", "config/ds_default.json")
        ds_config_abs_path = resolve_path_in_config(ds_config_path_str, cfg_project_root)
        if not ds_config_abs_path.exists():
            raise FileNotFoundError(f"[Worker {worker_rank}] DeepSpeed config not found: {ds_config_abs_path}")
        # Use PL's DeepSpeedStrategy, Ray handles the distributed environment.
        pl_strategy_object = DeepSpeedStrategy(config=str(ds_config_abs_path))
        logging.info(f"[Worker {worker_rank}] Using PL DeepSpeedStrategy: {ds_config_abs_path}")
    elif strategy_name == "ray_ddp":
        pl_strategy_object = ray.train.lightning.RayDDPStrategy(find_unused_parameters=cfg_strategy.get("find_unused_parameters", False))
        logging.info(f"[Worker {worker_rank}] Using RayDDPStrategy.")
    elif strategy_name == "ray_fsdp":
        pl_strategy_object = ray.train.lightning.RayFSDPStrategy(find_unused_parameters=cfg_strategy.get("find_unused_parameters", False))
        logging.info(f"[Worker {worker_rank}] Using RayFSDPStrategy.")
    else:
        raise ValueError(f"[Worker {worker_rank}] Unsupported Ray strategy: {strategy_name}")

    # Callbacks
    # RayTrainReportCallback is crucial. Others are optional per worker.
    callbacks = [ray.train.lightning.RayTrainReportCallback()]
    if cfg_run_settings.get("lr_monitor_logging_interval"):
        callbacks.append(LearningRateMonitor(logging_interval=cfg_run_settings.get("lr_monitor_logging_interval")))

    # Logger - Only for global rank 0 worker
    pl_logger = None
    if cfg_run_settings.get("use_wandb", True) and worker_rank == 0:
        pl_logger = WandbLogger(
            project=cfg_run_settings.get("project", "ray-lightning-project"),
            name=cfg_run_settings.get("run_name", "ray_train_run"),
            group=cfg_run_settings.get("experiment_group", cfg_run_settings.get("run_name")),
            # id= # Optionally set a fixed run ID if resuming across Ray jobs
        )

    # PL Trainer arguments
    trainer_actual_args = {
        "accelerator": cfg_trainer_params.get("accelerator", "auto"), # e.g., "gpu"
        "devices": "auto",  # Ray sets CUDA_VISIBLE_DEVICES, PL "auto" uses 1 device per worker
        "strategy": pl_strategy_object,
        "plugins": [ray.train.lightning.RayLightningEnvironment()], # ESSENTIAL
        "callbacks": callbacks,
        "logger": pl_logger,
        "precision": cfg_trainer_params.get("precision", "16-mixed"),
        "max_epochs": cfg_trainer_params.get("max_epochs", -1),
        "max_steps": cfg_trainer_params.get("max_steps", -1),
        "val_check_interval": cfg_trainer_params.get("val_check_interval", 1.0),
        "log_every_n_steps": cfg_trainer_params.get("log_every_n_steps", 50),
        "gradient_clip_val": cfg_trainer_params.get("gradient_clip_val", None),
        "enable_checkpointing": False,  # Per Ray docs, RayTrainReportCallback handles checkpoints
    }
    if trainer_actual_args["max_epochs"] == -1 and trainer_actual_args["max_steps"] == -1:
        trainer_actual_args["max_epochs"] = 1 # Default
        logging.warning(f"[Worker {worker_rank}] max_epochs/max_steps not set. Defaulting to 1 epoch.")

    # Create and prepare the PyTorch Lightning Trainer
    trainer = pl.Trainer(**trainer_actual_args)
    trainer = ray.train.lightning.prepare_trainer(trainer) # As per Ray docs

    # 4. Start Training
    logging.info(f"[Worker {worker_rank}] Starting trainer.fit(model, datamodule)...")
    trainer.fit(model, datamodule=datamodule) # Use datamodule here
    logging.info(f"[Worker {worker_rank}] trainer.fit() completed.")


# --- Main Entry Point ---
def main(config: dict):
    global PROJECT_ROOT, HAS_RAY # Access globals

    # Extract main config sections
    run_settings = config.get("run", {})
    model_settings = config.get("model", {}) # For run_name default
    trainer_params_from_config = config.get("trainer", {}) # For accelerator type
    ray_settings = config.get("ray", {})

    if "class_name" not in model_settings:
        logging.error("CRITICAL: 'model.class_name' not specified in config.")
        sys.exit(1)

    use_ray_from_config = ray_settings.get("use_ray", False)

    if use_ray_from_config and HAS_RAY:
        logging.info("--- Configuring for Ray Distributed Training (using TorchTrainer + train_func) ---")

        if not ray.is_initialized():
            ray_address = ray_settings.get("address", "auto")
            num_cpus_driver = ray_settings.get("driver_cpus", None)
            try:
                # Try to make workers use the current conda env
                current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
                runtime_env = {}
                if current_conda_env:
                    runtime_env["conda"] = current_conda_env
                    logging.info(f"Attempting to use current Conda env for Ray workers: {current_conda_env}")
                else:
                    logging.warning("CONDA_DEFAULT_ENV not found. Ray workers might use a different environment.")

                ray.init(
                    address=ray_address,
                    num_cpus=num_cpus_driver,
                    ignore_reinit_error=True,
                    runtime_env=runtime_env if runtime_env else None # Pass it here
                )
                logging.info(f"Ray initialized/connected. Resources: {ray.cluster_resources()}")
            except Exception as e_ray_init:
                logging.error(f"CRITICAL: Failed to initialize Ray: {e_ray_init}. Exiting.")
                sys.exit(1)

        # Train Loop Config (passed to each worker)
        # This dictionary must only contain serializable items.
        # All paths within should be resolved by the worker using its received PROJECT_ROOT.
        train_loop_config = {
            "project_root_str": str(PROJECT_ROOT), # Pass PROJECT_ROOT as string
            "global_seed": run_settings.get("seed", 42),
            "run_name_base": run_settings.get("run_name", f"{model_settings['class_name']}-ray"),
            "model_config": model_settings, # Pass the whole sub-config
            "datamodule_config": config.get("datamodule", {}),
            "trainer_params": trainer_params_from_config,
            "strategy_config": config.get("strategy", {}),
            "run_settings": run_settings, # For WandB project name, LRMonitor interval etc.
        }

        # Scaling Config for Ray Train
        accelerator_is_gpu = (trainer_params_from_config.get("accelerator", "gpu") == "gpu")
        scaling_config = ScalingConfig(
            num_workers=ray_settings.get("num_workers", 1),
            use_gpu=accelerator_is_gpu,
            resources_per_worker={
                "CPU": ray_settings.get("cpus_per_worker", 1),
                "GPU": 1 if accelerator_is_gpu else 0
            },
            #placement_group_strategy=ray_settings.get("placement_group_strategy", "PACK")
        )

        # Run Config for Ray Train (storage, checkpointing)
        ray_checkpoint_metric = run_settings.get("ray_checkpoint_monitor", "val_loss") # Metric from PL
        run_config_for_ray = RunConfig(
            name=run_settings.get("ray_experiment_name", f"{model_settings['class_name']}-exp"),
            storage_path=str(resolve_path_in_config(ray_settings.get("storage_path", "ray_results"), PROJECT_ROOT)),
            checkpoint_config=CheckpointConfig(
                num_to_keep=run_settings.get("ray_checkpoint_num_to_keep", 1),
                checkpoint_score_attribute=ray_checkpoint_metric, # Check Ray Train logs for exact name
                checkpoint_score_order=run_settings.get("ray_checkpoint_mode", "min")
            ),
            failure_config=ray.train.FailureConfig(max_failures=ray_settings.get("max_failures", 0))
        )
        logging.info(f"Ray TorchTrainer: ScalingConfig={scaling_config}, RunConfig={run_config_for_ray}")
        logging.info(f"  Ray Train will monitor '{ray_checkpoint_metric}' for checkpointing.")


        # Instantiate and run the Ray TorchTrainer
        ray_torch_trainer = TorchTrainer(
            train_loop_per_worker=train_func_for_ray,
            train_loop_config=train_loop_config, # Passed to train_func_for_ray
            scaling_config=scaling_config,
            run_config=run_config_for_ray,
        )

        logging.info("--- Launching Ray TorchTrainer.fit() ---")
        result = ray_torch_trainer.fit() # This blocks until training is done

        # Process result
        # ... (same result handling as before) ...
        logging.info("--- Ray Training Finished ---")
        if result.checkpoint: logging.info(f"Best checkpoint by Ray Train: {result.checkpoint.path if result.checkpoint else 'N/A'}")
        if result.metrics: logging.info(f"Final metrics from Ray Train (best result): {result.metrics}")
        if result.error: logging.error(f"Ray training FAILED: {result.error}")

        if ray_settings.get("shutdown_ray_after_job", True) and ray.is_initialized():
            logging.info("Shutting down Ray.")
            ray.shutdown()

    elif use_ray_from_config and not HAS_RAY:
        logging.error("Config specifies `use_ray: true`, but Ray components are not available. Install Ray with train extras (`pip install ray[train]`) or set `use_ray: false` for local run.")
        sys.exit(1)
    else:
        # --- Local Training Fallback ---
        logging.info("--- Configuration for Local PyTorch Lightning Training ---")
        # (Your existing local training logic can be placed here)
        # For brevity, I'll put a placeholder. You'd instantiate model, dm, pl.Trainer
        # with local strategies (DeepSpeed, PTL_DDPStrategy) and call .fit().
        # This local path should be fully fleshed out similar to `train_func_for_ray`
        # but using pl.Trainer directly and standard PL strategies.
        logging.warning("Local training path not fully implemented in this example. Focus is Ray.")
        logging.info("To run locally: set `ray.use_ray: false` in config and implement local training logic.")
        # Example:
        # from src.train_local import run_local_training_from_config # You could move it to a separate file
        # run_local_training_from_config(config, PROJECT_ROOT)


if __name__ == "__main__":
    logging.info("--- Initializing Training Script ---")
    script_start_time = time.time()
    try:
        setup_logging(level=logging.INFO) # Or DEBUG for more verbosity
    except NameError:
        # Fallback if setup_logging is not defined or imported
        logging.warning("setup_logging utility not found. Using basicConfig.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        raise

    # --- Pre-flight System Checks ---
    # (These checks are good to keep)
    logging.info(f"Python version: {sys.version.split()[0]}")
    logging.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    # ... (PyTorch, CUDA, PL, Ray version checks from your previous version) ...
    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        logging.info(f"PyTorch CUDA available: {cuda_available}")
        if cuda_available:
            logging.info(f"  Num GPUs (PyTorch): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()): logging.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError: logging.critical("PyTorch NOT INSTALLED!"); raise
    logging.info(f"PyTorch Lightning version: {pl.__version__}")
    if HAS_RAY: logging.info(f"Ray version detected: {ray.__version__}")
    else: logging.info("Ray components NOT imported (HAS_RAY=False).")


    # --- Config Loading ---
    # (Your robust config loading from previous version)
    config_path_env = os.environ.get("TRAIN_CONFIG_PATH")
    default_config_rel_path = Path("config") / "train_config.yaml"
    final_config_path = None
    if config_path_env:
        path_from_env = Path(config_path_env)
        if path_from_env.is_absolute() and path_from_env.exists(): final_config_path = path_from_env
        elif (PROJECT_ROOT / path_from_env).exists(): final_config_path = (PROJECT_ROOT / path_from_env).resolve()
        elif path_from_env.exists(): final_config_path = path_from_env.resolve()
        else: logging.warning(f"Config from TRAIN_CONFIG_PATH '{config_path_env}' not found.")
    if not final_config_path or not final_config_path.exists():
        final_config_path = (PROJECT_ROOT / default_config_rel_path).resolve()
        logging.warning(f"Using default config path: {final_config_path}")
    loaded_config = load_config(final_config_path)


    # --- Basic Config Sanity Checks ---
    if "model" not in loaded_config or "class_name" not in loaded_config["model"]:
        logging.error("Config must have 'model.class_name'. Exiting."); sys.exit(1)
    if loaded_config.get("ray", {}).get("use_ray", False) and not HAS_RAY:
        logging.error("Config `ray.use_ray: true` but Ray components missing. Exiting."); sys.exit(1)
    logging.info("Basic config checks passed.")

    # --- Dummy DataModule and Model Instantiation Check (Optional Pre-flight) ---
    # This helps catch config errors related to model/data init BEFORE Ray starts.
    # Can be time-consuming if data loading is slow.
    if loaded_config.get("run",{}).get("pre_flight_instantiation_check", False): # Add this to your config
        logging.info("--- Performing Pre-flight Instantiation Check ---")
        try:
            temp_model_cfg = loaded_config["model"]
            temp_dm_cfg = loaded_config.get("datamodule", {})
            
            logging.info("[Pre-flight] Instantiating model for check...")
            temp_model_cls = get_model_class(temp_model_cfg["module_path"], temp_model_cfg["class_name"])
            temp_model_kwargs = temp_model_cfg.get("kwargs", {}).copy()
            if "tokenizer_name_or_path" in temp_model_kwargs:
                temp_model_kwargs["tokenizer_name_or_path"] = str(resolve_path_in_config(temp_model_kwargs["tokenizer_name_or_path"], PROJECT_ROOT))
            temp_model_instance = temp_model_cls(**temp_model_kwargs)
            logging.info("[Pre-flight] Model instantiated successfully.")

            logging.info("[Pre-flight] Instantiating DataModule for check...")
            temp_dm_kwargs = temp_dm_cfg.get("kwargs", {}).copy()
            if "cache_dir" in temp_dm_kwargs:
                temp_dm_kwargs["cache_dir"] = str(resolve_path_in_config(temp_dm_kwargs["cache_dir"], PROJECT_ROOT))
            temp_dm_instance = LightningDataModule(tokenizer=temp_model_instance.tokenizer, **temp_dm_kwargs)
            logging.info("[Pre-flight] DataModule instantiated successfully.")
            
            # Optional: Call prepare_data and setup on the dummy DM
            # logging.info("[Pre-flight] Calling dummy_dm.prepare_data()...")
            # temp_dm_instance.prepare_data()
            # logging.info("[Pre-flight] Calling dummy_dm.setup('fit')...")
            # temp_dm_instance.setup('fit')
            # train_loader = temp_dm_instance.train_dataloader()
            # if train_loader and len(train_loader) > 0:
            #     logging.info(f"[Pre-flight] Dummy train_dataloader has {len(train_loader.dataset)} samples.")
            # else:
            #     logging.warning("[Pre-flight] Dummy train_dataloader is empty or None.")
            logging.info("--- Pre-flight Instantiation Check Successful ---")
        except Exception as e_preflight:
            logging.error(f"CRITICAL: Pre-flight instantiation check FAILED: {e_preflight}", exc_info=True)
            raise


    logging.info("--- Starting Main Training Logic ---")
    main(loaded_config) # Pass the loaded config

    script_end_time = time.time()
    logging.info(f"--- Training Script Finished. Total duration: {(script_end_time - script_start_time):.2f} seconds ---")