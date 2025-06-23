# train.py
import os
import sys
import yaml
import logging
import re
from pathlib import Path
from importlib import import_module # For dynamic model loading
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup Project Root ---
try:
    # Infer project root assuming this script is in src/data/
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    logging.info(f"hello there. PROJECT ROOT: {str(PROJECT_ROOT)}")
    # Optional path correction: map /archive/SSD/... to /davinci-1/...
    root_str = str(PROJECT_ROOT)
    if root_str.startswith("/archive/SSD/"):
        corrected_root = re.sub(r"^/archive/SSD/", "/davinci-1/", root_str)
        PROJECT_ROOT = Path(corrected_root)
        logging.info(f"Rewritten PROJECT_ROOT path to: {PROJECT_ROOT}")

    logging.info(f"Determined Project Root: {PROJECT_ROOT}")
except NameError:
    # __file__ not defined (e.g., running interactively)
    logging.warning("__file__ is not defined, using current working directory as project root.")
    PROJECT_ROOT = Path(os.getcwd())

sys.path.append(str(PROJECT_ROOT))
# --- End Project Root Setup ---

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
# Check if Ray is installed before importing Ray-specific components
try:
    from ray.train.lightning import RayLightningTrainer, RayDDPStrategy, RayFSDPStrategy # Add other strategies if needed
    from ray.train import ScalingConfig
    import ray
    HAS_RAY = True
    logging.info("Ray detected. RayLightningTrainer will be used.")
except Exception as e:
    HAS_RAY = False
    logging.warning("Ray not detected. Falling back to standard PyTorch Lightning Trainer.")
    RayLightningTrainer = None # Define as None if not available
    ScalingConfig = None # Define as None if not available
    raise e


# Import your DataModule
logging.info("about to import lightning data module")
from src.data.data_manager import LightningDataModule
from src.utils.utils import setup_logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        raise

def resolve_path(base_root: Path, path_str: str) -> str:
    """Resolves a path relative to the project root if it's not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        resolved_path = base_root / path
        logging.info(f"Resolved relative path '{path_str}' to '{resolved_path}'")
        return str(resolved_path)
    return path_str

def get_model_class(module_path: str, class_name: str):
    """Dynamically imports and returns a model class."""
    try:
        module = import_module(module_path)
        model_cls = getattr(module, class_name)
        logging.info(f"Successfully imported model class '{class_name}' from '{module_path}'")
        return model_cls
    except ImportError as e:
        logging.error(f"Could not import module '{module_path}': {e}")
        raise
    except AttributeError:
        logging.error(f"Could not find class '{class_name}' in module '{module_path}'")
        raise


def train_model(config: dict):
    """
    Sets up and runs the training process based on the configuration.
    """
    run_config = config.get("run", {})
    model_config = config.get("model", {})
    datamodule_config = config.get("datamodule", {})
    trainer_config = config.get("trainer", {})
    strategy_config = config.get("strategy", {})
    ray_config = config.get("ray", {})

    seed = run_config.get("seed", 42)
    seed_everything(seed, workers=True) # Ensure workers are also seeded
    logging.info(f"Set random seed to {seed}")

    # === Instantiate model === #
    model_module_path = model_config.get("module_path", "src.models.default_models") # Default path
    model_class_name = model_config.get("class_name")
    if not model_class_name:
        raise ValueError("Missing 'class_name' in model configuration section.")
    model_class = get_model_class(model_module_path, model_class_name)
    model_kwargs = model_config.get("kwargs", {})
    logging.info(f"Instantiating model '{model_class_name}' with kwargs: {model_kwargs}")
    # Note: Tokenizer is loaded *inside* the model's __init__
    model = model_class(**model_kwargs)

    # === Instantiate DataModule === #
    # Pass the tokenizer from the instantiated model to the DataModule
    datamodule_kwargs = datamodule_config.get("kwargs", {})
    # Resolve cache_dir relative to project root
    if "cache_dir" in datamodule_kwargs:
         datamodule_kwargs["cache_dir"] = resolve_path(PROJECT_ROOT, datamodule_kwargs["cache_dir"])

    logging.info(f"Instantiating LightningDataModule with kwargs: {datamodule_kwargs}")
    dm = LightningDataModule(
        tokenizer=model.tokenizer, # Use the tokenizer from the model
        **datamodule_kwargs
    )
    # Important: prepare_data and setup will be called by the Trainer/Ray Trainer automatically

    # === Logging === #
    logger = None
    if run_config.get("use_wandb", True):
        logger = WandbLogger(
            project=run_config.get("project", "llm-training"),
            name=run_config.get("run_name", f"{model_class_name}-run"),
            log_model=run_config.get("wandb_log_model", False), # Often False for large models
            # Consider adding entity, offline mode etc. from config
        )
        logging.info(f"Using WandbLogger for project '{logger.project}', run '{logger.name}'")
    else:
        logging.info("WandbLogger disabled.")

    # === Callbacks === #
    callbacks = []
    checkpoint_dir = resolve_path(PROJECT_ROOT, run_config.get("checkpoint_dir", "checkpoints/"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=run_config.get("checkpoint_monitor", "val_loss"),
        mode=run_config.get("checkpoint_mode", "min"),
        save_top_k=run_config.get("save_top_k", 1),
        dirpath=checkpoint_dir,
        filename=run_config.get("checkpoint_filename", "{epoch}-{val_loss:.2f}-best"),
        save_last=run_config.get("save_last_checkpoint", True) # Good for resuming
    )
    callbacks.append(checkpoint_callback)
    logging.info(f"Using ModelCheckpoint: monitor='{checkpoint_callback.monitor}', mode='{checkpoint_callback.mode}', dir='{checkpoint_callback.dirpath}'")

    if run_config.get("use_early_stopping", True):
        early_stop_callback = EarlyStopping(
            monitor=run_config.get("early_stop_monitor", "val_loss"),
            patience=run_config.get("early_stop_patience", 3),
            mode=run_config.get("early_stop_mode", "min"),
            verbose=True
        )
        callbacks.append(early_stop_callback)
        logging.info(f"Using EarlyStopping: monitor='{early_stop_callback.monitor}', patience={early_stop_callback.patience}")

    lr_monitor = LearningRateMonitor(logging_interval=run_config.get("lr_monitor_logging_interval", "step"))
    callbacks.append(lr_monitor)
    logging.info("Using LearningRateMonitor.")

    # === Strategy === #
    strategy_name = strategy_config.get("name", "deepspeed").lower()
    strategy = None
    if strategy_name == "deepspeed":
        ds_config_path = resolve_path(PROJECT_ROOT, strategy_config.get("config_path", "configs/ds_config_zero2.json"))
        if not Path(ds_config_path).exists():
             logging.warning(f"DeepSpeed config path '{ds_config_path}' does not exist. Check path.")
        # find_unused_parameters can be helpful for certain models/setups but has overhead
        find_unused = strategy_config.get("find_unused_parameters", False)
        strategy = DeepSpeedStrategy(
            config=ds_config_path,
            # remote_device=None, # Usually let DeepSpeed handle this
            # offload_optimizer=... # Controlled by JSON config
            # offload_parameters=... # Controlled by JSON config
        )
        logging.info(f"Using DeepSpeedStrategy with config: {ds_config_path}")
    elif strategy_name == "ddp" and HAS_RAY:
        strategy = RayDDPStrategy(find_unused_parameters=strategy_config.get("find_unused_parameters", False))
        logging.info("Using RayDDPStrategy.")
    elif strategy_name == "fsdp" and HAS_RAY:
         # FSDP requires specific configuration (policy, activation checkpointing)
         # Add FSDP specific kwargs from strategy_config if needed
        strategy = RayFSDPStrategy(find_unused_parameters=strategy_config.get("find_unused_parameters", False))
        logging.info("Using RayFSDPStrategy.")
    elif strategy_name == "auto":
        strategy = "auto" # Let Lightning choose
        logging.info("Using 'auto' strategy (Lightning chooses).")
    # Add more strategies (FSDP without Ray, etc.) if needed
    else:
        logging.warning(f"Unsupported or non-distributed strategy '{strategy_name}'. Using default.")
        strategy = None # Or "auto"

    # === Trainer Arguments === #
    # Arguments common to both standard Trainer and Ray Trainer init config
    trainer_args = {
        "max_epochs": trainer_config.get("max_epochs", 5),
        "max_steps": trainer_config.get("max_steps", -1), # -1 means use epochs
        "accelerator": trainer_config.get("accelerator", "gpu"),
        "devices": trainer_config.get("devices", "auto"), # 'auto' for local, 1 for Ray worker
        "logger": logger,
        "callbacks": callbacks,
        "precision": trainer_config.get("precision", "16-mixed"), # More flexible than just 16
        "gradient_clip_val": trainer_config.get("gradient_clip_val", None),
        "val_check_interval": trainer_config.get("val_check_interval", 1.0), # Check every epoch
        "log_every_n_steps": trainer_config.get("log_every_n_steps", 50),
        # Add other trainer args like accumulate_grad_batches, detect_anomaly etc. from config
    }

    # --- Choose Execution Mode: Ray or Local ---
    use_ray = ray_config.get("use_ray", False) and HAS_RAY

    if use_ray:
        logging.info("--- Starting Ray Training ---")
        if not ray.is_initialized():
             # Consider making Ray address configurable
             ray.init(address=ray_config.get("address", "auto")) # Connect to existing cluster or start new one

        # === Ray Trainer config === #
        # Ensure 'devices' for the trainer init is 1 when using Ray workers
        trainer_args["devices"] = 1
        trainer_args["strategy"] = strategy # Strategy defined above

        scaling_config = ScalingConfig(
            num_workers=ray_config.get("num_workers", 2), # Number of Ray workers
            use_gpu=trainer_args["accelerator"] == "gpu",
            resources_per_worker={
                "CPU": ray_config.get("cpus_per_worker", 4),
                 # Ensure GPU is requested only if accelerator is gpu
                "GPU": 1 if trainer_args["accelerator"] == "gpu" else 0
            },
            # Add placement group strategy etc. if needed
        )
        logging.info(f"Ray ScalingConfig: {scaling_config}")

        # === Run training on Ray Cluster === #
        # `run_config` in Ray Tune terminology, here just plain training args
        ray_run_config = ray.train.RunConfig(
             storage_path=resolve_path(PROJECT_ROOT, ray_config.get("storage_path", "/tmp/ray_results")), # For Ray logs/checkpoints
             # name=... # Ray experiment name if needed
             # failure_config=... # For retries
        )

        # Pass trainer args inside the trainer_init_config
        ray_trainer = RayLightningTrainer(
            trainer_init_config=trainer_args,
            scaling_config=scaling_config,
            run_config=ray_run_config
        )

        # `fit` now takes the datamodule directly
        result = ray_trainer.fit(model, datamodule=dm)
        logging.info(f"Ray training finished. Result: {result}")
        # Consider saving the best checkpoint path from result

    else:
        logging.info("--- Starting Local PyTorch Lightning Training ---")
        # Use the strategy defined earlier, adjust devices if needed for local multi-GPU
        trainer_args["strategy"] = strategy if strategy else "auto"
        if trainer_args["devices"] == "auto" and trainer_args["accelerator"] == "gpu":
             import torch
             trainer_args["devices"] = torch.cuda.device_count() if torch.cuda.is_available() else 1
             logging.info(f"Auto-detected {trainer_args['devices']} GPUs for local training.")

        local_trainer = Trainer(**trainer_args)
        local_trainer.fit(model, datamodule=dm)
        logging.info("Local training finished.")


if __name__ == "__main__":

    try:
        setup_logging(level=logging.INFO) # Or DEBUG for more verbosity
    except NameError:
        # Fallback if setup_logging is not defined or imported
        logging.warning("setup_logging utility not found. Using basicConfig.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config_path = os.environ.get("TRAIN_CONFIG_PATH", "config/train_config.yaml") # Allow setting via env var
    config = load_config(resolve_path(PROJECT_ROOT, config_path))
    train_model(config)

