# Script for running a saved model
from utils.models import load_model
from utils.misc import set_device
from utils.eval import generate_and_print_sample
import tiktoken

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
    generate_and_print_sample(model, tokenizer, device, start_context, TEMPERATURE)
    
if __name__ == "__main__":
    main()

