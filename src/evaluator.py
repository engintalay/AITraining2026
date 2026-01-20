import torch
import time
from typing import Dict
from .config import AppConfig
from .model_loader import load_model, load_tokenizer
from .utils import setup_logger
from peft import PeftModel

logger = setup_logger("Evaluator")

class Evaluator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tokenizer = load_tokenizer(config.model)
        
        # Load base model
        # We load it in inference mode
        self.base_model = load_model(config.model, inference_mode=True)
        
        # Try to load adapter if available
        self.model = self.base_model
        adapter_path = f"{config.training.output_dir}"
        import os
        if os.path.exists(adapter_path):
            try:
                logger.info(f"Loading LoRA adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            except Exception as e:
                logger.warning(f"Could not load adapter: {e}. Running in base model mode only.")
        else:
            logger.warning(f"No adapter found at {adapter_path}. Running purely with base model.")

    def generate_response(self, question: str, use_adapter: bool = False) -> Dict:
        # Prompt formatting
        input_text = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Enable/Disable adapter
        if isinstance(self.model, PeftModel):
            if use_adapter:
                self.model.enable_adapter_layers()
            else:
                 # Standard PeftModel doesn't have a global "disable" that works instantly for inference 
                 # without `disable_adapter_layers()` or context manager `disable_adapter()`.
                 # The context manager is safest.
                 pass 
        
        start_time = time.time()
        
        # Generation Logic with Context Manager for Base Model
        if isinstance(self.model, PeftModel) and not use_adapter:
             with self.model.disable_adapter():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
        else:
            # Either it's not a PeftModel, or we WANT to use the adapter
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
        end_time = time.time()
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response part logic if needed, but returning full text is fine for 1st pass.
        # Let's try to remove the prompt.
        response_text = generated_text.replace(input_text, "").strip()

        tokens_used = len(outputs[0])
        duration_ms = (end_time - start_time) * 1000
        
        return {
            "answer": response_text,
            "tokens_used": tokens_used,
            "response_time_ms": round(duration_ms, 2)
        }

    def compare(self, question: str):
        # 1. Base Model
        logger.info("Generating with Base Model...")
        base_stats = self.generate_response(question, use_adapter=False)
        
        # 2. Finetuned Model
        logger.info("Generating with Finetuned Model...")
        finetuned_stats = self.generate_response(question, use_adapter=True)
        
        return {
            "question": question,
            "base_model": base_stats,
            "finetuned_model": finetuned_stats
        }
