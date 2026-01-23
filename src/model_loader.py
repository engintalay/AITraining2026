import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from .config import ModelConfig, PeftConfig
from .utils import setup_logger

logger = setup_logger("ModelLoader")

def load_tokenizer(model_config: ModelConfig):
    logger.info(f"Loading tokenizer for {model_config.name_or_path}")
    
    # Yerel model için tokenizer sorunu varsa GPT-2 tokenizer kullan
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=True
        )
    except:
        logger.warning("Yerel tokenizer yüklenemedi, GPT-2 tokenizer kullanılıyor")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token")
    return tokenizer

def load_model(model_config: ModelConfig, peft_config: PeftConfig = None, inference_mode: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! GPU required for this model.")
    
    logger.info(f"Loading model {model_config.name_or_path} on GPU. Inference Mode: {inference_mode}")
    
    bnb_config = None
    if model_config.quantization_bit in [4, 8]:
        logger.info(f"Using {model_config.quantization_bit}-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(model_config.quantization_bit == 4),
            load_in_8bit=(model_config.quantization_bit == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4" if model_config.quantization_bit == 4 else "fp4"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False if not inference_mode else True,
        local_files_only=True  # Offline mod
    )

    if not inference_mode:
        if model_config.quantization_bit is not None:
             model = prepare_model_for_kbit_training(model)
        
        if peft_config:
            logger.info("Applying PEFT/LoRA configuration")
            lora_config = LoraConfig(
                r=peft_config.r,
                lora_alpha=peft_config.lora_alpha,
                target_modules=peft_config.target_modules,
                lora_dropout=peft_config.lora_dropout,
                bias=peft_config.bias,
                task_type=peft_config.task_type
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    return model
