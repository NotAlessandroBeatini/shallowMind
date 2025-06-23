# train.py
import os
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from ray.train.lightning import RayLightningTrainer
from ray.train import ScalingConfig

# === Import your model and dataloader dynamically === #
from models.default_models import HuggingFaceLLM  # or use from models.encoder_model import HuggingFaceEncoderModel
from data.data_manager import get_dataloaders


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(model_class, model_kwargs, dataloader_kwargs, run_config):
    seed_everything(run_config.get("seed", 42))

    # === Instantiate model === #
    model = model_class(**model_kwargs)

    # === Instantiate dataloaders === #
    train_loader, val_loader = get_dataloaders(model.tokenizer, **dataloader_kwargs)

    # === Logging === #
    wandb_logger = WandbLogger(
        project=run_config.get("project", "shallowMind"),
        name=run_config.get("run_name", "test-run"),
    )

    # === Callbacks === #
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=run_config.get("checkpoint_dir", "checkpoints/"),
        filename="best-checkpoint"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=run_config.get("early_stop_patience", 3),
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # === DeepSpeed Strategy === #
    strategy = DeepSpeedStrategy(
        config=run_config.get("deepspeed_config", "configs/ds_config.json")
    )

    # === Ray Trainer config === #
    scaling_config = ScalingConfig(
        num_workers=run_config.get("num_workers", 8),
        use_gpu=True,
        resources_per_worker={"CPU": run_config.get("cpus_per_worker", 4), "GPU": 1},
    )

    # === Run training on Ray Cluster === #
    ray_trainer = RayLightningTrainer(
        scaling_config=scaling_config,
        trainer_init_config={
            "max_epochs": run_config.get("max_epochs", 5),
            "accelerator": "gpu",
            "devices": 1,  # Always 1 per Ray worker
            "strategy": strategy,
            "logger": wandb_logger,
            "callbacks": [checkpoint_callback, early_stop_callback, lr_monitor],
            "precision": run_config.get("precision", 16),
        }
    )

    ray_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    config = load_config("configs/train_config.yaml")

    model_class = HuggingFaceLLM  # Or dynamically assign via config
    model_kwargs = config["model"]
    dataloader_kwargs = config["dataloader"]
    run_config = config["run"]

    train_model(model_class, model_kwargs, dataloader_kwargs, run_config)
