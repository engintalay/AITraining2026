import argparse
from src.config import AppConfig
from src.utils import setup_logger, print_system_info, set_seed
from src.model_loader import load_model, load_tokenizer
from src.data_handler import load_dataset
from src.trainer import LLMTrainer

logger = setup_logger("Main")

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Platform")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    print_system_info()
    set_seed(42)

    logger.info(f"Loading configuration from {args.config}")
    try:
        config = AppConfig.load_from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Load Tokenizer
    tokenizer = load_tokenizer(config.model)

    # Load Dataset
    dataset = load_dataset(config.data, tokenizer)

    # Load Model
    model = load_model(config.model, config.peft)

    # Initialize Trainer
    trainer = LLMTrainer(config, model, tokenizer, dataset)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()
