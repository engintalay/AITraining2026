import yaml
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
import os

class ModelConfig(BaseModel):
    name_or_path: str = Field(..., description="Path to local model or HuggingFace ID")
    quantization_bit: Optional[int] = Field(4, description="4 or 8 bit quantization")
    use_gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    
    @validator("quantization_bit")
    def validate_quantization(cls, v):
        if v not in [4, 8, None]:
            raise ValueError("Quantization must be 4, 8, or None")
        return v

class PeftConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = Field(default=["q_proj", "v_proj"], description="Modules to apply LoRA to")

class TrainingConfig(BaseModel):
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_steps: int = 50
    output_dir: str = "outputs"
    optim: str = "paged_adamw_32bit"
    max_steps: int = -1
    warmup_steps: int = 0
    resume_from_checkpoint: bool = True
    
class DataConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to JSON dataset")
    max_seq_length: int = 2048
    validation_split_percentage: int = 10

class AppConfig(BaseModel):
    model: ModelConfig
    peft: PeftConfig
    training: TrainingConfig
    data: DataConfig
    
    @classmethod
    def load_from_yaml(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found a {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
