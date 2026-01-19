import logging
import torch
import os
import psutil
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logger(name: str = "LLM_Trainer", log_file: str = "training.log", level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True, console=console),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(name)

def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return "CUDA Not Available"
    
    free_mem, total_mem = torch.cuda.mem_get_info()
    used_mem = total_mem - free_mem
    return {
        "total_gb": round(total_mem / (1024**3), 2),
        "used_gb": round(used_mem / (1024**3), 2),
        "free_gb": round(free_mem / (1024**3), 2)
    }

def print_system_info():
    logger = logging.getLogger("LLM_Trainer")
    logger.info("System Info:")
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    logger.info(f"RAM Total: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_info = get_gpu_memory_usage()
        logger.info(f"GPU Memory: {mem_info}")
    else:
        logger.warning("No GPU detected. Training might be extremely slow or fail.")

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
