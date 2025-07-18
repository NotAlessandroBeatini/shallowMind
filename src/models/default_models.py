# src/models/default_models.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from typing import Optional, Dict, Any, List
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

# --- Optional: Import DeepSpeed Optimizers ---
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    DeepSpeedCPUAdam = None
    FusedAdam = None
# --- End Optional Imports ---

class HuggingFaceLLM(pl.LightningModule):
    """
    A PyTorch LightningModule for training/fine-tuning Hugging Face AutoModels
    for Causal Language Modeling, including text generation with sampling.
    """
    def _safe_print(self, *objects, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
        """
        Behaves like `LightningModule.print` **after** the model is attached to a Trainer,
        but falls back to the built-in `print` **before** that (e.g. during __init__).

        If `DEBUG_MODE` is on, always logs through the module-level logger instead.
        """
        # Debug mode: send everything to the Python logger
        if getattr(self, "_debug_mode", False):
            logger.info(sep.join(str(o) for o in objects))
            return

        trainer = getattr(self, "_trainer", None)   
        if trainer is not None and getattr(trainer, "is_global_zero", True):
            # The Lightning helper is now safe to call
            super().print(*objects, sep=sep, end=end, file=file, flush=flush)
        else:
            # Trainer not yet attached → plain old print
            print(*objects, sep=sep, end=end, file=file, flush=flush)


    def __init__(
        self,
        # --- Model and Tokenizer ---
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        attn_implementation: Optional[str] = "sdpa",
        tokenizer_cache_dir: Optional[str] = None,

        # --- Training Hyperparameters ---
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        warmup_steps_ratio: float = 0.0,
        use_deepspeed_adam: bool = False,
        prefer_cpu_adam: bool = False,

        # --- Generation Specific ---
        predict_max_length: int = 128,
        predict_num_beams: int = 1, # Set to 1 if using sampling primarily
        predict_do_sample: bool = True,
        predict_top_k: int = 50,
        predict_top_p: float = 0.95,
        predict_temperature: float = 0.7,
        predict_repetition_penalty: float = 1.0,

        use_torch_compile: bool = False,

        # --- Debugging Control ---
        DEBUG_MODE: bool = False,


        **kwargs # Allow passing extra unused args from config
    ):
        super().__init__()
        self._debug_mode = DEBUG_MODE
        self._printer = self._safe_print
        #self._printer = logger.info if self._debug_mode else self.print

        # Consolidate all potential hparams from __init__ args and kwargs
        #locals() returns a dict of all local variables, right after entering the function, is essentially the full list
        #of constructor arguments plus self.
        current_locals = locals().copy()
        hparams_to_save = {
            key: current_locals[key] for key in [
                "model_name_or_path", "attn_implementation", "tokenizer_name_or_path", "learning_rate",
                "weight_decay", "adam_epsilon", "warmup_steps_ratio",
                "use_deepspeed_adam", "prefer_cpu_adam", "predict_max_length",
                "predict_num_beams", "predict_do_sample", "predict_top_k",
                "predict_top_p", "predict_temperature", "predict_repetition_penalty", "use_torch_compile" ,
            ]
        }
        # Add any other predict_* params from kwargs if they were passed
        for k_arg in kwargs:
            if k_arg.startswith("predict_") and k_arg not in hparams_to_save:
                hparams_to_save[k_arg] = kwargs[k_arg]

        self.save_hyperparameters(hparams_to_save)

        self.model_cache_dir = model_cache_dir
        self.tokenizer_cache_dir = tokenizer_cache_dir

        _tokenizer_path = self.hparams.tokenizer_name_or_path or self.hparams.model_name_or_path
        self._printer(f"Loading tokenizer from: {_tokenizer_path}")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            _tokenizer_path,
            cache_dir=self.tokenizer_cache_dir,
            use_fast=True,
            padding_side='left',
            
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self._printer(f"Tokenizer pad_token/pad_token_id was None. Set to eos_token: {self.tokenizer.eos_token} (ID: {self.tokenizer.pad_token_id})")
            else:
                self._printer(f"Warning: Tokenizer has no eos_token. Adding a [PAD] token.")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self._printer(f"Loading model config for: {self.hparams.model_name_or_path}")
        model_config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            cache_dir=self.model_cache_dir,
        )
        model_config.pad_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            model_config.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.bos_token_id is not None:
             model_config.bos_token_id = self.tokenizer.bos_token_id

        self._printer(f"Loading AutoModelForCausalLM: {self.hparams.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hparams.model_name_or_path,
            config=model_config,
            cache_dir=self.model_cache_dir,
            attn_implementation=self.hparams.attn_implementation,
        )
        
        current_model_vocab_size = self.model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(self.tokenizer)
        if tokenizer_vocab_size != current_model_vocab_size:
            self._printer(f"Resizing model token embeddings from {current_model_vocab_size} to {tokenizer_vocab_size}")
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            self.model.config.vocab_size = tokenizer_vocab_size

       # Optionally compile only the forward call
        if use_torch_compile and hasattr(torch, "compile"):
            compiled_fwd = torch.compile(self.model.forward)
            # Bind the compiled function as this module’s forward
            # We wrap it so that self is correctly passed through
            def _forward_compiled(*args, **kwargs):
                return compiled_fwd(*args, **kwargs)
            self.forward = _forward_compiled

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def _common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, step_name: str) -> torch.Tensor:
        model_outputs = self.forward(**batch)
        if model_outputs.loss is None:
            print_func = logger.warning if self._debug_mode else lambda msg: self.print(msg, rank_zero_only=True)
            print_func(f"Warning: Loss is None in {step_name}_step (batch_idx: {batch_idx}). Ensure 'labels' are in batch and model computes loss.")
            if hasattr(model_outputs, 'logits') and "labels" in batch:
                logits = model_outputs.logits
                labels = batch["labels"]
                ignore_idx = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
                loss = F.cross_entropy(logits.view(-1, self.model.config.vocab_size), labels.view(-1), ignore_index=ignore_idx)
            else:
                print_func(f"Error: Cannot compute loss in {step_name}_step (batch_idx: {batch_idx}) as logits or labels are missing.")
                return torch.tensor(0.0, device=self.device, requires_grad=True if step_name == "train" else False)
        else:
            loss = model_outputs.loss
        return loss

    def _do_logging(self, name: str, value: Any, **kwargs):
        """
        A wrapper for self.log() that handles different logging backends
        based on the execution context (real training vs. local debug).
        """
        # --- Main Logging Path for W&B ---
        # We expect to enter this block during a real training run.
        if not self._debug_mode and self.trainer and self.trainer.loggers:
            self.log(name, value, **kwargs)

        # --- Fallback for Local Debugging (e.g., if __name__ == "__main__") ---
        elif self._debug_mode:
            log_str = f"[Local Debug] {name}: {value}"
            if kwargs.get("prog_bar"): log_str += " (prog_bar)"
            if kwargs.get("on_step"): log_str += " (on_step)"
            if kwargs.get("on_epoch"): log_str += " (on_epoch)"
            logger.info(log_str)
            
        # --- NEW: Diagnostic block if logging is skipped during a real run ---
        # This 'else' will only be reached if 'not self._debug_mode' is True,
        # but the trainer or logger is not available.
        else:
            # Use a flag to ensure this detailed warning is printed only once.
            if not hasattr(self, '_logging_warning_printed'):
                self._printer("\n" + "="*80)
                self._printer("!!! LOGGING SKIPPED (NON-DEBUG MODE) !!!")
                self._printer(f"W&B logging for '{name}' was skipped because the condition was not met.")
                self._printer(f"  - self._debug_mode: {self._debug_mode}")
                self._printer(f"  - self.trainer is attached: {self.trainer is not None}")
                if self.trainer:
                    self._printer(f"  - self.trainer.loggers is available: {self.trainer.loggers}")
                self._printer("This warning will not be shown again for this run.")
                self._printer("="*80 + "\n")
                
                # Set the flag to prevent spamming the log.
                self._logging_warning_printed = True

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx, "train")
        lr = -1.0
        if self.trainer and hasattr(self.trainer, 'optimizers') and self.trainer.optimizers and \
           len(self.trainer.optimizers) > 0 and self.trainer.optimizers[0].param_groups and \
           len(self.trainer.optimizers[0].param_groups) > 0:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self._do_logging("lr", lr, on_step=True, logger=True, sync_dist=False)
        self._do_logging("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self._common_step(batch, batch_idx, "val")
        perplexity = torch.exp(loss) # Calculate perplexity
        self._do_logging("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._do_logging("val_perplexity", perplexity, on_epoch=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self._common_step(batch, batch_idx, "test")
        self._do_logging("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> List[str]:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if input_ids is None:
            self._printer("Warning: 'input_ids' not found in batch for predict_step. Skipping generation.")
            first_key = next(iter(batch), None)
            batch_size = batch[first_key].size(0) if first_key else 1
            return [""] * batch_size

        gen_kwargs = {
            "max_length": self.hparams.get("predict_max_length", 128),
            "num_beams": self.hparams.get("predict_num_beams", 1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.hparams.get("predict_do_sample", True), # Default to True for more diverse output
            "top_k": self.hparams.get("predict_top_k", 50),
            "top_p": self.hparams.get("predict_top_p", 0.95),
            "temperature": self.hparams.get("predict_temperature", 0.7),
            "repetition_penalty": self.hparams.get("predict_repetition_penalty", 1.0)
        }
        # If using beam search, sampling parameters might not be used or behave differently
        if gen_kwargs["num_beams"] > 1:
            gen_kwargs["do_sample"] = False # Typically False when num_beams > 1
            # top_k, top_p, temperature might still have some effect or be ignored with beam search
            # depending on the transformers version and exact generate logic.
            # Often, for beam search, you primarily control num_beams and length_penalty.
            self._printer(f"Beam search active (num_beams={gen_kwargs['num_beams']}). 'do_sample' forced to False.")
        
        self._printer(f"Generating text with kwargs: {gen_kwargs}")
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        
        prompt_lengths = [torch.sum(mask).item() for mask in attention_mask] if attention_mask is not None else [ids.shape[-1] for ids in input_ids]
        generated_texts = []
        for i, gen_ids_sample in enumerate(generated_ids):
            start_index = min(prompt_lengths[i], len(gen_ids_sample))
            generated_part_ids = gen_ids_sample[start_index:]
            decoded_text = self.tokenizer.decode(generated_part_ids, skip_special_tokens=True)
            generated_texts.append(decoded_text.strip())
            if self._debug_mode:
                 full_decoded_with_special = self.tokenizer.decode(gen_ids_sample, skip_special_tokens=False)
                 logger.info(f"  Full Gen (Sample {i}, Special Tokens): {full_decoded_with_special}")
                 logger.info(f"  Generated Part (Sample {i}): {decoded_text.strip()}")
        return generated_texts

    def configure_optimizers(self):
        """
        This method is required by PyTorch Lightning, but we are letting the
        DeepSpeed strategy handle the optimizer and scheduler creation based
        on the 'strategy.config_dict' in the main YAML config file.
        
        Therefore, we simply do nothing here.
        """
        pass 
    # def configure_optimizers(self):
    #     no_decay = ["bias", "LayerNorm.weight", "embedding.weight"]
    #     optimizer_grouped_parameters = [
    #         {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.hparams.weight_decay,},
    #         {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0,},
    #     ]
    #     OptimizerClass = torch.optim.AdamW
    #     optimizer_kwargs = {"lr": self.hparams.learning_rate, "eps": self.hparams.adam_epsilon}
    #     if self.hparams.use_deepspeed_adam and HAS_DEEPSPEED:
    #         if self.hparams.prefer_cpu_adam and DeepSpeedCPUAdam:
    #             OptimizerClass, optimizer_kwargs["adamw_mode"] = DeepSpeedCPUAdam, True
    #             self._printer("Using DeepSpeedCPUAdam optimizer (CPU Offload).")
    #         elif not self.hparams.prefer_cpu_adam and FusedAdam:
    #             OptimizerClass, optimizer_kwargs["adam_w_mode"] = FusedAdam, True
    #             self._printer("Using FusedAdam optimizer (GPU).")
    #         else: self._printer(f"DeepSpeed Adam requested but specific type not available/preferred. Falling back to AdamW.")
    #     else: self._printer(f"Using standard torch.optim.AdamW optimizer.")
    #     optimizer = OptimizerClass(optimizer_grouped_parameters, **optimizer_kwargs)

    #     if self.hparams.warmup_steps_ratio > 0:
    #         num_training_steps = -1
    #         if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches is not None and self.trainer.estimated_stepping_batches > 0 :
    #             num_training_steps = self.trainer.estimated_stepping_batches
    #             self._printer(f"Scheduler: Estimated total training steps from estimated_stepping_batches: {num_training_steps}")
    #         elif self.trainer and hasattr(self.trainer, 'max_steps') and self.trainer.max_steps is not None and self.trainer.max_steps > 0 :
    #             num_training_steps = self.trainer.max_steps
    #             self._printer(f"Scheduler: Using max_steps for total training steps: {num_training_steps}")
    #         if num_training_steps <=0 :
    #              self._printer(f"Warning: num_training_steps ({num_training_steps}) is not positive or could not be determined. Scheduler disabled.")
    #              return optimizer
    #         num_warmup_steps = int(self.hparams.warmup_steps_ratio * num_training_steps)
    #         self._printer(f"Scheduler: Warmup steps: {num_warmup_steps}, Total steps: {num_training_steps}")
    #         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    #         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
    #     else:
    #         self._printer("No warmup configured for LR scheduler.")
    #         return optimizer

# --- Example for local testing (Optional) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Testing HuggingFaceLLM with Generation...")

    model_name = "gpt2" # Using gpt2
    config_params = {
        "model_name_or_path": model_name, "tokenizer_name_or_path": model_name,
        "learning_rate": 5e-5, "weight_decay": 0.01, "adam_epsilon": 1e-8,
        "warmup_steps_ratio": 0.0, "use_deepspeed_adam": False, "prefer_cpu_adam": False,
        "DEBUG_MODE": True,

        # --- Generation Params for Testing ---
        "predict_max_length": 70,         # Max total length (prompt + new tokens)
        "predict_num_beams": 1,           # Set > 1 for beam search
        "predict_do_sample": True,        # Activate sampling
        "predict_top_k": 50,
        "predict_top_p": 0.95,
        "predict_temperature": 0.7,
        "predict_repetition_penalty": 1.2
    }
    model = HuggingFaceLLM(**config_params)
    print("\nModel Hyperparameters:"); print(model.hparams)
    print(f"Tokenizer final padding side: {model.tokenizer.padding_side}")
    print(f"Tokenizer final pad_token: '{model.tokenizer.pad_token}' (ID: {model.tokenizer.pad_token_id})")
    print(f"Model config pad_token_id: {model.model.config.pad_token_id}")
    print(f"Model config eos_token_id: {model.model.config.eos_token_id}")
    print(f"Model config bos_token_id: {model.model.config.bos_token_id}")
    print(f"Model vocab size: {model.model.config.vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device for local test: {device} ---")
    model.to(device)

    prompts = ["The capital of France is", "Once upon a time, in a galaxy"]
    # Keep prompt tokenization max_length relatively short to allow for more generation
    inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=10)
    
    dummy_batch = {"input_ids": inputs["input_ids"].to(device), "attention_mask": inputs["attention_mask"].to(device)}
    print(f"\nDummy batch for prediction (prompts tokenized for device {device}):\n{dummy_batch}"); print(f"  Input IDs shape: {dummy_batch['input_ids'].shape}")

    try:
        print("\nTesting predict_step (generation)...")
        generated_texts = model.predict_step(dummy_batch, 0)
        print("\nGenerated Texts (only the generated part):")
        for i, text in enumerate(generated_texts): print(f"  Prompt '{prompts[i][:30]}...': '{text}'")
        assert len(generated_texts) == len(prompts)
    except Exception as e: print(f"Predict step (generation) failed: {e}"); raise

    print("\n--- Testing Training Step (Quick Check) ---")
    train_seq_len, train_batch_size = 32, 2
    dummy_input_ids_train = torch.randint(0, model.tokenizer.vocab_size, (train_batch_size, train_seq_len), device=device)
    dummy_attention_mask_train = torch.ones_like(dummy_input_ids_train, device=device) 
    dummy_labels_train = dummy_input_ids_train.clone().to(device)
    dummy_train_batch = {"input_ids": dummy_input_ids_train, "attention_mask": dummy_attention_mask_train, "labels": dummy_labels_train}
    try:
        class DummyTrainer:
            def __init__(self, module):
                self.global_rank = 0; self.is_global_zero = True
                self.logger = None; self.loggers = []
                self.lightning_module = module
                self.estimated_stepping_batches = 100; self.max_steps = -1 
                self.accelerator_connector = type('obj', (object,), {'strategy': 'foo'})()
                
                optimizers_output = module.configure_optimizers()
                if isinstance(optimizers_output, dict): self.optimizers = [optimizers_output["optimizer"]]
                else: self.optimizers = [optimizers_output]
        
        model.trainer = DummyTrainer(model)
        loss = model.training_step(dummy_train_batch, 0)
        # Loss will be printed by the _do_logging helper using logger.info
        loss.backward()
        print("Backward pass successful for training step.")
    except Exception as e: print(f"Training step check failed: {e}"); raise
    print("\nHuggingFaceLLM with generation basic tests passed!")