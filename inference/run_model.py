# Script for running a saved model
import logging
from configs.configuration_manager import ConfigurationManager
from text_generation.text_generator import TextGenerator
from utils.model_persistance import load_model
from utils.misc import set_device

import tiktoken

from utils.seeding import set_seed


def main():
    MODEL_PATH = f"gpt_model_1epochs-wiki-1-500min.pt"
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
    generator = TextGenerator(model, tokenizer, llm_configs, logger=logger)
    text = generator.generate_text(start_context)
    logger.info(text)

    # Basic usage with temperature
    text = generator.generate_text(
        prompt="Once upon a time",
        max_new_tokens=50,
        temperature=0.7
    )

    # With top-k and top-p sampling
    text = generator.generate_text(
        prompt="Once upon a time",
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )

    # Using beam search
    text = generator.generate_text(
        prompt="Once upon a time",
        max_new_tokens=50,
        num_beams=5,
        temperature=0.7,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    # Using greedy decoding (no sampling)
    text = generator.generate_text(
        prompt="Once upon a time",
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0
    )
    
if __name__ == "__main__":
    main()

