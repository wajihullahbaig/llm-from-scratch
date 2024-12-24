# Script for running a saved model
import logging
from configs.configuration_manager import ConfigurationManager
from text_generation.text_generator import TextGenerator
from utils.model_persistance import load_model
from utils.misc import set_device

import tiktoken

from utils.seeding import set_seed

# Change the string below for the saved model's name
MODEL_PATH = f"gpt_model_1epochs-wiki-1-500min.pt"
TEMPERATURE = 1.2

def main():
    model = load_model(MODEL_PATH)

    tokenizer = tiktoken.get_encoding("gpt2")
    device = set_device()
    model.to(device)

    start_context = "Water is essential for all forms of life because "

    print(f"\n\n{50*'='}")
    set_seed(42)
    # Setup logging 
    logger_name = "GPT2-Trainer"
    logger = logging.getLogger("GPT2-Trainer")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - {logger_name} - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    llm_configs = ConfigurationManager("config.yaml",logger=logger)     
    tg = TextGenerator(model, tokenizer, llm_configs, logger=logger)
    text = tg.generate_text(start_context)
    logger.info(text)

    
if __name__ == "__main__":
    main()

