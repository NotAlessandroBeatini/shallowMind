# custom_model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer # Optional: If using text
from typing import Optional, Dict, Any

# --- Optional: Import DeepSpeed Optimizers (consistent with other models) ---
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    DeepSpeedCPUAdam = None
    FusedAdam = None
# --- End Optional Imports ---


class CustomModelTemplate(pl.LightningModule):
    """
    A template for a custom PyTorch Lightning model integrated with the training workflow.
    Replace placeholders with your specific architecture and logic.
    """
    def __init__(
        self,
        # --- Essential Hyperparameters for Training ---
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0, # For optional scheduler
        use_deepspeed_adam: bool = False, # Reuse optimizer logic
        prefer_cpu_adam: bool = False,

        # --- Custom Architecture Hyperparameters (MODIFY/ADD AS NEEDED) ---
        vocab_size: Optional[int] = None, # Example: Needed if using embeddings
        embedding_dim: int = 256,       # Example: Dimension for embeddings/features
        hidden_dim: int = 512,          # Example: Dimension for RNN/Transformer layers
        num_layers: int = 2,            # Example: Number of layers in RNN/Transformer
        num_classes: Optional[int] = None, # Example: For classification tasks
        dropout_prob: float = 0.1,      # Example: Dropout rate

        # --- Tokenizer (Optional - only if your model processes text) ---
        tokenizer_name_or_path: Optional[str] = None,
        tokenizer_cache_dir: Optional[str] = None,
        # --- End Tokenizer ---

        **kwargs # Allow passing extra unused args without error
    ):
        """
        Args:
            learning_rate, weight_decay, ...: Standard training hyperparameters.
            use_deepspeed_adam, prefer_cpu_adam: Optimizer selection flags.
            vocab_size, embedding_dim, ...: YOUR custom model hyperparameters.
            num_classes: Number of output classes if doing classification.
            tokenizer_name_or_path: Path/name if using a HuggingFace tokenizer.
        """
        super().__init__()
        # Save hyperparameters (makes them accessible via self.hparams and handles checkpointing)
        # Make sure ALL arguments you want logged and accessible are defined above
        self.save_hyperparameters()

        # --- Optional: Load Tokenizer ---
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        if self.hparams.tokenizer_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path,
                cache_dir=self.hparams.tokenizer_cache_dir
            )
            # Crucial if using embeddings based on tokenizer vocab size
            if self.hparams.vocab_size is None:
                 self.hparams.vocab_size = self.tokenizer.vocab_size
                 self.print(f"Inferred vocab_size from tokenizer: {self.hparams.vocab_size}")
            elif self.hparams.vocab_size != self.tokenizer.vocab_size:
                 self.print(f"Warning: Provided vocab_size ({self.hparams.vocab_size}) differs "
                            f"from tokenizer vocab_size ({self.tokenizer.vocab_size}). Using provided value.")

            # Handle padding token consistency if needed (like in decoder model)
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token # Common practice
                 self.print(f"Set tokenizer pad_token to eos_token: {self.tokenizer.eos_token}")
        # --- End Tokenizer ---


        # ====================================================================
        # --- 1. DEFINE YOUR CUSTOM MODEL LAYERS HERE ---
        # ====================================================================
        # Example Layers (replace with your actual architecture):

        # a) Embedding Layer (if processing sequences of indices, e.g., text)
        if self.hparams.vocab_size:
             self.embedding = nn.Embedding(
                 self.hparams.vocab_size,
                 self.hparams.embedding_dim,
                 padding_idx=self.tokenizer.pad_token_id if self.tokenizer else None # Handle padding if tokenizer used
            )
        else:
             # Maybe your input is already features, no embedding needed
             self.embedding = nn.Identity() # Placeholder if no embedding needed

        # b) Encoder (Example: LSTM or a stack of Linear layers)
        # Example LSTM:
        self.encoder = nn.LSTM(
            input_size=self.hparams.embedding_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            batch_first=True, # Important: Lightning batches are typically (batch, seq, feature)
            dropout=self.hparams.dropout_prob if self.hparams.num_layers > 1 else 0,
            bidirectional=False # Example choice
        )
        # Example Linear Encoder:
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.hparams.dropout_prob),
        #     # Add more layers as needed
        # )


        # c) Output Head (Example: Classification or Language Modeling)
        if self.hparams.num_classes:
            # Classification Head
            self.output_head = nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        elif self.hparams.vocab_size:
             # Language Model Head (predict next token)
             self.output_head = nn.Linear(self.hparams.hidden_dim, self.hparams.vocab_size)
        else:
             # Regression Head or other task
             # self.output_head = nn.Linear(self.hparams.hidden_dim, 1) # Example regression
             raise ValueError("Define an output head based on your task (e.g., set num_classes or vocab_size).")


        # d) Dropout Layer
        self.dropout = nn.Dropout(self.hparams.dropout_prob)

        # ====================================================================
        # --- End Layer Definition ---
        # ====================================================================


    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, features: Optional[torch.Tensor] = None, **kwargs):
        """
        Defines the forward pass of your custom model.

        Args:
            input_ids (Optional[torch.Tensor]): Batch of input IDs (e.g., from tokenizer).
            attention_mask (Optional[torch.Tensor]): Mask for padding tokens.
            features (Optional[torch.Tensor]): Batch of pre-computed features (if not using input_ids).
            **kwargs: Additional arguments from the batch.

        Returns:
            torch.Tensor: The final output of the model (e.g., logits).
        """
        # ====================================================================
        # --- 2. IMPLEMENT YOUR FORWARD PASS LOGIC HERE ---
        # ====================================================================

        # --- a) Handle Input ---
        if input_ids is not None:
            # Assumes input_ids -> embedding -> encoder
            if not hasattr(self, 'embedding') or not isinstance(self.embedding, nn.Embedding):
                 raise RuntimeError("Model received input_ids but has no nn.Embedding layer or vocab_size wasn't set.")
            x = self.embedding(input_ids) # (batch, seq_len, embedding_dim)
        elif features is not None:
            # Assumes features -> encoder directly
            x = features # (batch, seq_len, feature_dim) - ensure feature_dim matches encoder input size
        else:
            raise ValueError("Model needs either 'input_ids' or 'features' in the batch.")

        x = self.dropout(x) # Apply dropout after embedding/input features

        # --- b) Pass through Encoder ---
        # Example for LSTM:
        # packed_output, (hidden, cell) = self.encoder(x) # Pass hidden state if stateful
        encoder_output, _ = self.encoder(x) # Get output sequence from LSTM/Transformer etc.
                                         # Output shape depends on encoder type (e.g., LSTM: (batch, seq_len, hidden_dim))

        # Example for Sequential Linear Encoder:
        # encoder_output = self.encoder(x)

        # --- c) Select Relevant Output for Head ---
        # For classification with RNN/Transformer, often use the output of the first token ([CLS]) or the last hidden state.
        # For sequence-to-sequence or LM, use the whole sequence output.
        # *ADJUST THIS BASED ON YOUR TASK AND ENCODER*
        # Example: Use the last time step output from LSTM for classification
        if self.hparams.num_classes and isinstance(self.encoder, nn.LSTM):
             # Assuming batch_first=True for LSTM
             final_encoder_output = encoder_output[:, -1, :] # (batch, hidden_dim)
        else:
             # Example: Use the entire sequence for LM head
             final_encoder_output = encoder_output # (batch, seq_len, hidden_dim)


        final_encoder_output = self.dropout(final_encoder_output)

        # --- d) Pass through Output Head ---
        logits = self.output_head(final_encoder_output) # (batch, num_classes) or (batch, seq_len, vocab_size)

        return logits
        # ====================================================================
        # --- End Forward Pass ---
        # ====================================================================


    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss based on model output and labels.
        MODIFY THIS based on your specific task (classification, LM, regression, etc.).
        """
        # --- Example: Cross-Entropy Loss for Classification ---
        if self.hparams.num_classes:
             # Input logits: (batch, num_classes)
             # Input labels: (batch,) with class indices
             loss = F.cross_entropy(logits, labels)

        # --- Example: Cross-Entropy Loss for Language Modeling ---
        elif self.hparams.vocab_size:
             # Input logits: (batch, seq_len, vocab_size)
             # Input labels: (batch, seq_len) with token indices (often shifted input_ids)
             # Reshape for cross_entropy: expects (N, C) and (N,)
             loss = F.cross_entropy(
                 logits.view(-1, self.hparams.vocab_size), # (batch * seq_len, vocab_size)
                 labels.view(-1), # (batch * seq_len,)
                 ignore_index=self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else -100
                 # Important: Ignore padding tokens in loss calculation if applicable
             )
        # --- Add other loss types (e.g., MSE for regression) ---
        # elif task == 'regression':
        #    loss = F.mse_loss(logits.squeeze(), labels.float())
        else:
            raise NotImplementedError("Loss calculation not implemented for the current task configuration. Set num_classes or vocab_size, or add custom loss logic.")

        return loss

    def _common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Common logic for training, validation, and test steps."""
        # Prepare inputs for the forward pass
        # The batch contents depend on your LightningDataModule's __getitem__
        forward_kwargs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch.get("labels")

        if labels is None:
            raise ValueError("Batch must contain 'labels' for loss calculation during training/validation/testing.")

        # Get model output (logits)
        logits = self.forward(**forward_kwargs)

        # Calculate loss
        loss = self._calculate_loss(logits, labels)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        loss = self._common_step(batch, batch_idx)
        lr = self.trainer.optimizers[0].param_groups[0]['lr'] # Get current LR
        self.log("lr", lr, on_step=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log other custom training metrics if needed
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single validation step."""
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Log other custom validation metrics (e.g., accuracy)
        # Example Accuracy (for classification):
        # if self.hparams.num_classes:
        #     logits = self.forward(**{k: v for k, v in batch.items() if k != 'labels'})
        #     preds = torch.argmax(logits, dim=-1)
        #     acc = (preds == batch['labels']).float().mean()
        #     self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single test step."""
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # Log other custom test metrics

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Performs prediction. MODIFY this based on what you want to output.
        By default, returns the raw model output (logits).
        """
        forward_kwargs = {k: v for k, v in batch.items() if k != 'labels'} # Don't need labels for prediction
        output = self.forward(**forward_kwargs)
        # Could return probabilities: F.softmax(output, dim=-1)
        # Could return class predictions: torch.argmax(output, dim=-1)
        # Could implement generation logic if it's a generative model
        return output

    def configure_optimizers(self):
            """
            This method is required by PyTorch Lightning, but we are letting the
            DeepSpeed strategy handle the optimizer and scheduler creation based
            on the 'strategy.config_dict' in the main YAML config file.
            
            Therefore, we simply do nothing here.
            """
            pass 

    # def configure_optimizers(self):
    #     """
    #     Configures the optimizer and optional learning rate scheduler.
    #     Reuses the robust logic from previous examples.
    #     """
    #     # Filter out parameters that don't require gradients
    #     # Apply weight decay only to certain parameters (e.g., not biases and LayerNorms)
    #     no_decay = ["bias", "LayerNorm.weight", "embedding.weight"] # Often exclude embedding weight too
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
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
    #          self.print(f"Using standard torch.optim.AdamW optimizer.")


    #     optimizer = OptimizerClass(optimizer_grouped_parameters, **optimizer_kwargs)

    #     # --- Optional: Learning Rate Scheduler ---
    #     # Example: Linear warmup and decay (requires installing transformers)
    #     # from transformers import get_linear_schedule_with_warmup
    #     # try:
    #     #    num_training_steps = self.trainer.estimated_stepping_batches
    #     #    self.print(f"Estimated training steps: {num_training_steps}")
    #     # except Exception:
    #     #     # Might fail if trainer setup isn't complete yet, provide fallback or estimate
    #     #     num_training_steps = 100000 # Placeholder, adjust
    #     #     self.print(f"Could not estimate training steps, using placeholder: {num_training_steps}")

    #     # scheduler = get_linear_schedule_with_warmup(
    #     #     optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps
    #     # )
    #     # scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     # return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    #     # Return only optimizer if no scheduler is used
    #     return optimizer

# --- Example for local testing (Optional) ---
if __name__ == "__main__":
    print("Testing CustomModelTemplate...")

    # Example config (adjust to your needs)
    config = {
        "learning_rate": 1e-3,
        "vocab_size": 1000, # Example vocab size
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 1,
        "num_classes": 10, # Example classification task
        "dropout_prob": 0.1,
        # "tokenizer_name_or_path": "gpt2" # Uncomment if using a tokenizer
    }
    model = CustomModelTemplate(**config)
    print("\nModel Hyperparameters:")
    print(model.hparams)
    print("\nModel Architecture:")
    print(model)

    # Create dummy batch (adjust shapes based on your model/task)
    batch_size = 4
    seq_len = 20
    if model.hparams.vocab_size:
        dummy_input_ids = torch.randint(0, model.hparams.vocab_size, (batch_size, seq_len))
    else:
        dummy_input_ids = None # Or create dummy features

    if model.hparams.num_classes:
        dummy_labels = torch.randint(0, model.hparams.num_classes, (batch_size,)) # Classification labels
    elif model.hparams.vocab_size:
         dummy_labels = torch.randint(0, model.hparams.vocab_size, (batch_size, seq_len)) # LM labels
    else:
         dummy_labels = torch.rand(batch_size, 1) # Regression labels example


    dummy_batch = {"input_ids": dummy_input_ids, "labels": dummy_labels}
    # Add "attention_mask" if your model uses it
    # dummy_batch["attention_mask"] = torch.ones_like(dummy_input_ids)

    # Test forward pass
    try:
        print("\nTesting forward pass...")
        logits = model.forward(**{k: v for k, v in dummy_batch.items() if k != 'labels'})
        print("Logits shape:", logits.shape) # Check if shape matches expectation
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    # Test training step
    try:
        print("\nTesting training step...")
        loss = model.training_step(dummy_batch, 0)
        print("Training loss:", loss.item())
        loss.backward() # Test backward pass
        print("Backward pass successful.")
    except Exception as e:
        print(f"Training step failed: {e}")
        raise

    # Test optimizer configuration
    try:
        print("\nTesting optimizer configuration...")
        opt = model.configure_optimizers()
        print("Optimizer:", opt)
    except Exception as e:
        print(f"Optimizer configuration failed: {e}")
        raise

    print("\nCustomModelTemplate basic tests passed!")

