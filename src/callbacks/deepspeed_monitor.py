# In src/callbacks/deepspeed_monitor.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import logging

logger = logging.getLogger(__name__)

class DeepSpeedMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.grad_norm_dict = {}

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer):
        """
        This hook is called by Lightning just before the optimizer step.
        It's the perfect place to get the grad norm from the DeepSpeed engine.
        """
        if not hasattr(trainer.strategy, "model") or trainer.strategy.model is None:
            # DeepSpeed engine not yet initialized
            return

        # Access the DeepSpeedEngine instance
        deepspeed_engine = trainer.strategy.model

        # DeepSpeed has a `get_global_grad_norm` method which is exactly what we need.
        # It might be called different things in different versions, so we check.
        if hasattr(deepspeed_engine, 'get_global_grad_norm'):
            grad_norm = deepspeed_engine.get_global_grad_norm()
        elif hasattr(deepspeed_engine, 'global_grad_norm'):
            grad_norm = deepspeed_engine.global_grad_norm
        else:
            grad_norm = -1.0 # Indicate that we couldn't find it

        if grad_norm != -1.0:
            # We use self.log here, which will be sent to W&B on the next logging step
            pl_module.log("deepspeed/grad_norm", grad_norm, on_step=True, on_epoch=False)