# In src/callbacks/deepspeed_monitor.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import logging
import torch # Import torch

logger = logging.getLogger(__name__)

class DeepSpeedMonitor(Callback):
    # __init__ can be removed if it only calls super()

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer):
        if not hasattr(trainer.strategy, "model") or trainer.strategy.model is None:
            return

        deepspeed_engine = trainer.strategy.model
        grad_norm = None # Default to None

        if hasattr(deepspeed_engine, 'get_global_grad_norm'):
            # This is a method call
            grad_norm = deepspeed_engine.get_global_grad_norm()
        elif hasattr(deepspeed_engine, 'global_grad_norm'):
            # This is accessing an attribute
            grad_norm = deepspeed_engine.global_grad_norm
        
        # --- THIS IS THE FIX ---
        # Only log if grad_norm is a valid number (not None).
        # We also check if it's a tensor and get its item() value if so.
        if grad_norm is not None:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
                
            pl_module.log("deepspeed/grad_norm", grad_norm, on_step=True, on_epoch=False)