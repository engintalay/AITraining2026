import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .config import DataConfig
from .utils import setup_logger

logger = setup_logger("DataHandler")

class InstructDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_seq_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # Format prompt (Simplified alpaca-like format)
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output_text + self.tokenizer.eos_token

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Labels: Mask the prompt part for loss calculation (optional but standard)
        # For simplicity in this v1, we train on the whole sequence or just masking? 
        # Making labels = input_ids for CausalLM. 
        # ideally we should mask the user prompt, but simple CausalLM fine-tuning works without valid masking too, just slightly less efficient.
        # Let's do simple clone for now.
        labels = input_ids.clone()
        
        # Simple masking of padding tokens
        # -100 is the ignore index for CrossEntropyLoss
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_dataset(data_config: DataConfig, tokenizer):
    logger.info(f"Loading dataset from {data_config.dataset_path}")
    with open(data_config.dataset_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Found {len(data)} samples")
    return InstructDataset(data, tokenizer, data_config.max_seq_length)
