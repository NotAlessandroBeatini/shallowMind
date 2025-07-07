# src/models/encoder_model.py

import pytorch_lightning as pl
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    # AdamW, # Use torch.optim.AdamW
    # get_linear_schedule_with_warmup # Optional: If scheduler needed
)
from typing import Optional, Dict, Any

# Try to import DeepSpeed optimizers
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    DeepSpeedCPUAdam = None # Define for type hints
    FusedAdam = None        # Define for type hints


class HuggingFaceEncoderModel(pl.LightningModule):
    """
    A LightningModule for encoder-only models (e.g., BERT, RoBERTa)
    supporting sequence classification or Masked Language Modeling (MLM).
    """
    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        task: str = "classification",  # "classification" or "mlm"
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        num_labels: int = 2, # Only used for classification task
        use_deepspeed_adam: bool = False, # Default False for encoders unless needed
        prefer_cpu_adam: bool = False,
        tokenizer_cache_dir: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name_or_path (str): HuggingFace model name or path.
            task (str): Task type: "classification" or "mlm".
            learning_rate (float): Peak learning rate.
            adam_epsilon (float): Epsilon for AdamW optimizer.
            warmup_steps (int): Warmup steps for LR scheduler (if used).
            weight_decay (float): Weight decay for optimizer.
            num_labels (int): Number of output labels for classification task.
            use_deepspeed_adam (bool): Attempt to use DeepSpeed Adam optimizers.
            prefer_cpu_adam (bool): Prefer DeepSpeedCPUAdam if use_deepspeed_adam=True.
            tokenizer_cache_dir (Optional[str]): Path to cache directory for tokenizer.
            model_cache_dir (Optional[str]): Path to cache directory for model.
        """
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path, cache_dir=self.hparams.tokenizer_cache_dir
        )

        if self.hparams.task == "classification":
            self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.model_name_or_path,
                num_labels=self.hparams.num_labels,
                cache_dir=self.hparams.model_cache_dir
            )
        elif self.hparams.task == "mlm":
            self.model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
                 self.hparams.model_name_or_path,
                 cache_dir=self.hparams.model_cache_dir
            )
        else:
            raise ValueError(f"Unsupported task '{self.hparams.task}'. Choose 'classification' or 'mlm'.")

        # Ensure pad token is consistent if model uses it (less common for BERT classification)
        # but good practice for MLM or if using padding during classification.
        if self.tokenizer.pad_token is None and self.model.config.pad_token_id is None:
             # BERT often uses token ID 0 for padding by default if not explicitly set
             # Avoid setting eos_token as pad_token for BERT unless absolutely necessary
             # Check if a pad_token_id exists in the config even if pad_token is None
             if self.tokenizer.pad_token_id is not None:
                  self.print(f"Tokenizer has pad_token_id {self.tokenizer.pad_token_id} but pad_token is None.")
             else:
                  self.print("Warning: Tokenizer and model have no default pad token/ID. Ensure padding strategy is handled correctly.")


    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass specific to the task."""
        # For MLM, labels are the input_ids with non-masked tokens set to -100
        # For Classification, labels are class indices
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids, # Pass token_type_ids if present in batch
            return_dict=True, # Ensure output is a dictionary-like object
            **kwargs
        )

    def _common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Common logic for training, validation, and test steps."""
        # Pass the whole batch, letting the model pick what it needs
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        loss = self._common_step(batch, batch_idx)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single validation step."""
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Optionally log task-specific metrics (Accuracy, F1 for classification)
        # You would need to compute predictions from logits (outputs.logits) and compare to labels

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single test step."""
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # Optionally log task-specific metrics

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """Performs prediction, returning logits."""
        # Don't pass labels during prediction
        predict_batch = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self.model(**predict_batch, return_dict=True)
        return {"logits": outputs.logits, "input_ids": batch["input_ids"]} # Return logits and inputs

    def configure_optimizers(self):
        """
        This method is required by PyTorch Lightning, but we are letting the
        DeepSpeed strategy handle the optimizer and scheduler creation based
        on the 'strategy.config_dict' in the main YAML config file.
        
        Therefore, we simply do nothing here.
        """
        pass 
    
    # def configure_optimizers(self):
    #     """Configures the optimizer, adding DeepSpeed Adam support."""
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
    #     }

    #     if self.hparams.use_deepspeed_adam and HAS_DEEPSPEED:
    #         if self.hparams.prefer_cpu_adam:
    #             if DeepSpeedCPUAdam:
    #                 OptimizerClass = DeepSpeedCPUAdam
    #                 optimizer_kwargs["adamw_mode"] = True
    #                 self.print("Using DeepSpeedCPUAdam optimizer (CPU Offload).")
    #             else:
    #                 self.print("DeepSpeedCPUAdam requested but not available, falling back to AdamW.")
    #         else:
    #             if FusedAdam:
    #                 OptimizerClass = FusedAdam
    #                 optimizer_kwargs["adam_w_mode"] = True
    #                 self.print("Using FusedAdam optimizer (GPU).")
    #             else:
    #                 self.print("FusedAdam requested but not available, falling back to AdamW.")
    #     else:
    #          self.print(f"Using standard torch.optim.AdamW optimizer. use_deepspeed_adam={self.hparams.use_deepspeed_adam}, HAS_DEEPSPEED={HAS_DEEPSPEED}")


    #     optimizer = OptimizerClass(optimizer_grouped_parameters, **optimizer_kwargs)

    #     # --- Optional: Learning Rate Scheduler ---
    #     # scheduler = ...
    #     # return {"optimizer": optimizer, "lr_scheduler": ...}

    #     return optimizer

# For quick testing
if __name__ == "__main__":
    print("\nTesting HuggingFaceEncoderModel (Classification)...")
    try:
        model_cls = HuggingFaceEncoderModel(model_name_or_path="prajjwal1/bert-tiny", task="classification", num_labels=3, learning_rate=1e-4)
        tokenizer_cls = model_cls.tokenizer
        inputs_cls = tokenizer_cls("This is a classification test.", return_tensors="pt")
        inputs_cls["labels"] = torch.tensor([1]) # Dummy label
        outputs_cls = model_cls(**inputs_cls)
        print(f"Classification Test Loss: {outputs_cls.loss.item():.4f}")
        opt_cls = model_cls.configure_optimizers()
        print(f"Classification Optimizer: {type(opt_cls)}")
        pred_cls = model_cls.predict_step(inputs_cls, 0)
        print(f"Classification Predict Logits Shape: {pred_cls['logits'].shape}")

    except Exception as e:
        print(f"Error during classification test: {e}")


    print("\nTesting HuggingFaceEncoderModel (MLM)...")
    try:
        model_mlm = HuggingFaceEncoderModel(model_name_or_path="prajjwal1/bert-tiny", task="mlm", learning_rate=1e-4)
        tokenizer_mlm = model_mlm.tokenizer
        text_mlm = "Let's test the [MASK] language model."
        inputs_mlm = tokenizer_mlm(text_mlm, return_tensors="pt")
        # Create MLM labels (copy input_ids, mask non-[MASK] tokens)
        labels_mlm = inputs_mlm["input_ids"].clone()
        labels_mlm[labels_mlm != tokenizer_mlm.mask_token_id] = -100 # Ignore non-masked tokens in loss
        inputs_mlm["labels"] = labels_mlm

        outputs_mlm = model_mlm(**inputs_mlm)
        print(f"MLM Test Loss: {outputs_mlm.loss.item():.4f}")
        opt_mlm = model_mlm.configure_optimizers()
        print(f"MLM Optimizer: {type(opt_mlm)}")
        pred_mlm = model_mlm.predict_step(inputs_mlm, 0)
        print(f"MLM Predict Logits Shape: {pred_mlm['logits'].shape}")

    except Exception as e:
        print(f"Error during MLM test: {e}")

