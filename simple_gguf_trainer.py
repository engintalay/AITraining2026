#!/usr/bin/env python3
"""
Simple GGUF model trainer using llama-cpp-python
For low VRAM systems like GTX 1650
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load training dataset from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_training_data(dataset: List[Dict]) -> List[str]:
    """Format dataset for training"""
    formatted_data = []
    for item in dataset:
        if 'input' in item and 'output' in item:
            text = f"Soru: {item['input']}\nCevap: {item['output']}"
            formatted_data.append(text)
    return formatted_data

def main():
    try:
        # Load config
        config = load_config('config_gtx1650.yaml')
        logger.info("Configuration loaded successfully")
        
        # Load dataset
        dataset_path = config['data']['dataset_path']
        dataset = load_dataset(dataset_path)
        logger.info(f"Loaded {len(dataset)} training examples")
        
        # Format data
        training_texts = format_training_data(dataset)
        logger.info(f"Formatted {len(training_texts)} training texts")
        
        # Print sample
        if training_texts:
            logger.info(f"Sample training text:\n{training_texts[0][:200]}...")
        
        # For GGUF files, you would typically need to:
        # 1. Convert to a format suitable for fine-tuning (like safetensors)
        # 2. Use a different training framework
        # 3. Or use the model for inference only
        
        logger.info("GGUF files are typically used for inference, not training.")
        logger.info("For fine-tuning, you need the original model in HuggingFace format.")
        logger.info("Consider downloading a smaller model like 'microsoft/DialoGPT-small' when internet is available.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
