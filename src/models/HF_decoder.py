# src/models/default_models.py

import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Optional, Dict, Any

# Try to import DeepSpeed optimizers
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    DeepSpeedCPUAdam = None # Define for type hints
    FusedAdam = None        # Define for type hints

class HuggingFaceLLM(pl.LightningModule):
    """
    A PyTorch LightningModule that wraps a Hugging Face causal language model
    for training and inference.
    """
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8, # Added epsilon
        warmup_steps: int = 0, # Added for potential scheduler
        weight_decay: float = 0.0, # Added for optimizer
        use_deepspeed_adam: bool = True,
        prefer_cpu_adam: bool = False,
        predict_max_length: int = 64,
        tokenizer_cache_dir: Optional[str] = None, # Optional cache dir
        model_cache_dir: Optional[str] = None, # Optional cache dir
        # Add other relevant hyperparameters like gradient checkpointing config
    ):
        """
        Args:
            model_name_or_path (str): Name or path of the Hugging Face model.
            learning_rate (float): Peak learning rate for optimizer.
            adam_epsilon (float): Epsilon for AdamW optimizer.
            warmup_steps (int): Number of warmup steps for learning rate scheduler (if used).
            weight_decay (float): Weight decay for optimizer.
            use_deepspeed_adam (bool): Whether to attempt using DeepSpeed-optimized Adam if available.
            prefer_cpu_adam (bool): If True and use_deepspeed_adam is True, prefers DeepSpeedCPUAdam.
            predict_max_length (int): Max length for generation in predict_step.
            tokenizer_cache_dir (Optional[str]): Path to cache directory for tokenizer.
            model_cache_dir (Optional[str]): Path to cache directory for model.
        """
        super().__init__()
        # Use save_hyperparameters() to automatically save arguments to self.hparams
        # and handle loading from checkpoints
        self.save_hyperparameters()

        # Load tokenizer and model from Hugging Face
        # Pass cache_dir if provided
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path, cache_dir=self.hparams.tokenizer_cache_dir
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
             self.hparams.model_name_or_path, cache_dir=self.hparams.model_cache_dir
        )

        # Ensure padding token exists for tokenizer and model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.print(f"Tokenizer missing pad token, setting to EOS token: {self.tokenizer.eos_token}")
            # Also update the model config if necessary
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through the underlying Hugging Face model."""
        # Filter out kwargs not expected by the model if necessary, or let the model handle it
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            # Pass other relevant args like output_hidden_states=..., output_attentions=... if needed
            **kwargs
        )

    def _common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Common logic for training, validation, and test steps."""
        # Assuming labels are correctly prepared in the dataloader (e.g., input_ids shifted)
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"), # Use .get for safety
            labels=batch["labels"]
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        loss = self._common_step(batch, batch_idx)
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single validation step."""
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) # Sync across GPUs
        # Optionally calculate perplexity: torch.exp(loss)
        # self.log("val_perplexity", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single test step."""
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True) # Sync across GPUs
        # self.log("test_perplexity", torch.exp(loss), on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """Performs prediction (generation) for a batch."""
        # Generate uses model.forward internally, doesn't need labels
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_length=self.hparams.predict_max_length,
            # Add other generation parameters (do_sample, num_beams, temperature, etc.)
            # potentially from hparams
            pad_token_id=self.tokenizer.pad_token_id, # Ensure pad token ID is set
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Return generated sequences and potentially input IDs for reference
        return {"generated_ids": outputs, "input_ids": batch["input_ids"]}


    def configure_optimizers(self):
        """
        This method is required by PyTorch Lightning, but we are letting the
        DeepSpeed strategy handle the optimizer and scheduler creation based
        on the 'strategy.config_dict' in the main YAML config file.
        
        Therefore, we simply do nothing here.
        """
        pass 
    # def configure_optimizers(self):
    #     """Configures the optimizer and learning rate scheduler."""
    #     # Filter out parameters that don't require gradients (if any)
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #             "weight_decay": 0.0,
    #         },
    #     ]

    #     # --- Select Optimizer Class ---
    #     OptimizerClass = torch.optim.AdamW # Default fallback
    #     optimizer_kwargs = {
    #         "lr": self.hparams.learning_rate,
    #         "eps": self.hparams.adam_epsilon,
    #         # weight_decay is handled by parameter groups
    #     }

    #     if self.hparams.use_deepspeed_adam and HAS_DEEPSPEED:
    #         if self.hparams.prefer_cpu_adam:
    #             if DeepSpeedCPUAdam:
    #                 OptimizerClass = DeepSpeedCPUAdam
    #                 # DeepSpeedCPUAdam might have different args, adjust kwargs if needed
    #                 optimizer_kwargs["adamw_mode"] = True # Usually needed
    #                 self.print("Using DeepSpeedCPUAdam optimizer (CPU Offload).")
    #             else:
    #                 self.print("DeepSpeedCPUAdam requested but not available, falling back to AdamW.")
    #         else:
    #             if FusedAdam:
    #                 OptimizerClass = FusedAdam
    #                  # FusedAdam might have different args, adjust kwargs if needed
    #                 optimizer_kwargs["adam_w_mode"] = True # Usually needed
    #                 self.print("Using FusedAdam optimizer (GPU).")
    #             else:
    #                 self.print("FusedAdam requested but not available, falling back to AdamW.")
    #     else:
    #          self.print(f"Using standard torch.optim.AdamW optimizer. use_deepspeed_adam={self.hparams.use_deepspeed_adam}, HAS_DEEPSPEED={HAS_DEEPSPEED}")


    #     optimizer = OptimizerClass(optimizer_grouped_parameters, **optimizer_kwargs)

    #     # --- Optional: Learning Rate Scheduler ---
    #     # Example: Linear warmup and decay
    #     # scheduler = get_linear_schedule_with_warmup(
    #     #     optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches
    #     # )
    #     # scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     # return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    #     # Return only optimizer if no scheduler is used
    #     return optimizer


# For quick local testing
if __name__ == "__main__":
    print("Testing HuggingFaceLLM...")
    # Create an instance for testing
    model = HuggingFaceLLM(model_name_or_path="distilgpt2", learning_rate=1e-4)
    tokenizer = model.tokenizer

    # Create a dummy input
    text = ["Hello, this is a test.", "Another short sentence."]
    # Ensure batch processing works with padding
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Create labels (usually input_ids shifted, but for loss calculation, just copy)
    tokens["labels"] = tokens["input_ids"].clone()
    # Mask padding token labels
    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = -100


    # Run a forward pass
    try:
        output = model(**tokens)
        print(f"Test forward pass successful. Loss: {output.loss.item():.4f}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Test optimizer configuration
    try:
        opt = model.configure_optimizers()
        print(f"Optimizer configuration successful: {type(opt)}")
    except Exception as e:
        print(f"Error during optimizer configuration: {e}")

    # Test prediction step
    try:
         # Simulate a batch for prediction
        pred_batch = tokenizer("Generate text from this:", return_tensors="pt", padding=True, truncation=True)
        pred_output = model.predict_step(pred_batch, 0)
        print(f"Prediction step successful. Generated IDs shape: {pred_output['generated_ids'].shape}")
        print("Decoded generation:")
        decoded = tokenizer.batch_decode(pred_output['generated_ids'], skip_special_tokens=True)
        for i, txt in enumerate(decoded):
            print(f" Sample {i}: {txt}")
    except Exception as e:
         print(f"Error during prediction step: {e}")

