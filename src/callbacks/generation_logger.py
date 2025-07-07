# In src/callbacks/generation_logger.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import logging

# Use the module's logger for cleaner output
logger = logging.getLogger(__name__)

# --- CENTRALIZED PROMPT REPOSITORY ---
# Define different sets of prompts for various evaluation purposes.
PROMPT_SETS = {
    "default": [
        "Once upon a time, in a land of code,",
        "The best way to train a language model is",
        "Question: What is the capital of Italy? Answer:",
    ],
    "creative": [
        "Write a short poem about a lonely robot.",
        "The spaceship landed on the alien planet. The first thing the astronaut saw was",
        "A recipe for a dish that doesn't exist:",
    ],
    "qa": [
        "Question: What is the powerhouse of the cell? Answer:",
        "Question: Explain the theory of relativity in one sentence. Answer:",
        "Question: Who wrote the novel '1984'? Answer:",
    ],
    "code": [
        "// Python function to calculate the factorial of a number\ndef factorial(n):",
        "/* Javascript code to make a GET request using fetch */\nfetch('https://api.example.com/data')",
    ]
}


class GenerationLogger(Callback):
    def __init__(self, prompt_set_name: str = "default", log_every_n_steps: int = -1):
        """
        A callback to log generated text samples to a wandb.Table.

        Args:
            prompt_set_name (str): The name of the prompt set to use from PROMPT_SETS.
            log_every_n_steps (int): Log every N validation steps (batches). Default -1 logs on epoch end.
        """
        super().__init__()

        if prompt_set_name not in PROMPT_SETS:
            raise ValueError(
                f"Prompt set '{prompt_set_name}' not found. "
                f"Available sets are: {list(PROMPT_SETS.keys())}"
            )
        
        self.prompts = PROMPT_SETS[prompt_set_name]
        self.log_every_n_steps = log_every_n_steps
        logger.info(f"GenerationLogger initialized with prompt set '{prompt_set_name}'.")

    def _log_generations(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Only run on global rank 0 to avoid multiple workers generating and logging
        if trainer.global_rank != 0:
            return

        pl_module.eval() # Put the model in evaluation mode

        columns = ["global_step", "prompt", "generated_text"]
        data = []

        for prompt in self.prompts:
            try:
                inputs = pl_module.tokenizer(prompt, return_tensors="pt", padding=True).to(pl_module.device)
                generated_ids = pl_module.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=pl_module.tokenizer.eos_token_id # Crucial for open-ended generation
                )
                prompt_len = inputs["input_ids"].shape[1]
                decoded_text = pl_module.tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
                data.append([trainer.global_step, prompt, decoded_text])
            except Exception as e:
                logger.error(f"Error during text generation for prompt '{prompt}': {e}")
                data.append([trainer.global_step, prompt, f"GENERATION FAILED: {e}"])

        # Log the data to a W&B Table
        if trainer.logger:
            trainer.logger.log_table(key=f"generations_{pl_module.current_epoch}", columns=columns, data=data)
            logger.info(f"Logged {len(data)} generated text samples to W&B table.")

        pl_module.train() # Put the model back in training mode

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Log at a specific step interval during the validation epoch."""
        if self.log_every_n_steps > 0 and (batch_idx + 1) % self.log_every_n_steps == 0:
            self._log_generations(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log at the end of the validation epoch if step-based logging is disabled."""
        if self.log_every_n_steps <= 0:
            self._log_generations(trainer, pl_module)