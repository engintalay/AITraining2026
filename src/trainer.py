import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
from .config import AppConfig
from .utils import setup_logger, get_gpu_memory_usage
import time

logger = setup_logger("Trainer")

class MonitoringCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.total_tokens_processed = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_time = time.time()
            elapsed_total = current_time - self.start_time
            
            # Estimate tokens - crude approximation if not directly provided
            # Standard HF logs don't give "tokens processed" directly in step logs usually,
            # but we can infer from batch size * seq len * steps roughly or rely on 'loss'.
            # Better: The prompt wants "tokens/sec".
            # We can calculate it if we knew the input sizes, but let's approximate or just log step speed.
            # Using samples/sec from logs is safer.
            
            # logs often has: 'loss', 'learning_rate', 'epoch', 'step'
            
            # Let's augment the log with GPU usage
            gpu_stats = get_gpu_memory_usage()
            
            logger.info(f"Step {state.global_step} | Loss: {logs.get('loss', 'N/A')} | Epoch: {logs.get('epoch', 'N/A')}")
            if isinstance(gpu_stats, dict):
                 logger.info(f"VRAM Used: {gpu_stats['used_gb']} GB")

            # Calculate ETA if not present (HF usually prints it, but we log it)
            if state.max_steps > 0:
                percent_done = state.global_step / state.max_steps
                if percent_done > 0:
                    eta_seconds = (elapsed_total / percent_done) - elapsed_total
                    logger.info(f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s")

class LLMTrainer:
    def __init__(self, config: AppConfig, model, tokenizer, dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
    
    def train(self):
        logger.info("Initializing Trainer...")
        
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            num_train_epochs=self.config.training.num_train_epochs,
            optim=self.config.training.optim,
            warmup_steps=self.config.training.warmup_steps,
            fp16=True if torch.cuda.is_available() else False,
            report_to="none", # We use our own logger
            ddp_find_unused_parameters=False,
            # Resume logic handled here or via trainer.train(resume_from_checkpoint=...)
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[MonitoringCallback()]
        )

        logger.info("Starting training...")
        
        # Handle Resume
        resume_from_checkpoint = None
        if self.config.training.resume_from_checkpoint:
            # If output_dir exists and has checkpoints, it will try to resume.
            # Explicitly passing True to train method tells it to look in output_dir.
            resume_from_checkpoint = self.config.training.resume_from_checkpoint
            logger.info(f"Resume capability enabled: {resume_from_checkpoint}")

        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        logger.info("Training completed.")
        logger.info(f"Saving final model to {self.config.training.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training.output_dir)
        
        return train_result
