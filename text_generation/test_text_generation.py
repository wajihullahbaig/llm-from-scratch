import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer
from text_generator import TextGenerator
from utils.misc import set_device
from utils.model_persistance import load_model
from utils.seeding import set_seed 

def test_text_generation():
    # Initialize model and tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    device = set_device()
    MODEL_PATH = f"outputs/07012025_171018/GPT_CONFIG_162M/gpt-model-wiki-162M.pt"
    model,config = load_model(MODEL_PATH)
    model.to(device)

    print(f"\n\n{50*'='}")
    set_seed(42)
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        context_size=config["context_length"],
    )

    # Test Case 1: Basic Generation with different temperatures
    prompt = "Water is essential for all forms of life because"
    print("\n=== Temperature Testing ===")
    for temp in [0.2, 0.7,1.2, 1.5]:
        output = generator.generate_text(
            prompt=prompt,
            max_new_tokens=30,
            temperature=temp,
            do_sample=True
        )
        print(f"\nTemperature {temp}:", output)

    # Test Case 2: Top-k and Top-p Sampling
    print("\n=== Top-k and Top-p Testing ===")
    sampling_configs = [
        {"top_k": 10, "top_p": None},
        {"top_k": None, "top_p": 0.5},
        {"top_k": 50, "top_p": 0.9}
    ]
    
    for config in sampling_configs:
        output = generator.generate_text(
            prompt=prompt,
            max_new_tokens=30,
            temperature=1.2,
            **config
        )
        print(f"\nConfig {config}:", output)

    # Test Case 3: Beam Search
    print("\n=== Beam Search Testing ===")
    beam_sizes = [1, 3, 5,7,9]
    for num_beams in beam_sizes:
        output = generator.generate_text(
            prompt=prompt,
            max_new_tokens=30,
            num_beams=num_beams,
            temperature=1.2,
            early_stopping=True
        )
        print(f"\nBeam Size {num_beams}:", output)

    # Test Case 4: Repetition Prevention
    print("\n=== Repetition Prevention Testing ===")
    repetitive_prompt = "The cat and the dog. The cat and the dog. The"
    
    # Without repetition penalty
    output = generator.generate_text(
        prompt=repetitive_prompt,
        max_new_tokens=50,
        repetition_penalty=2.0
    )
    print("\nNo Repetition Penalty:", output)
    
    # With repetition penalty
    output = generator.generate_text(
        prompt=repetitive_prompt,
        max_new_tokens=50,
        repetition_penalty=1.5
    )
    print("\nWith Repetition Penalty 1.5:", output)
    
    # With n-gram prevention
    output = generator.generate_text(
        prompt=repetitive_prompt,
        max_new_tokens=50,
        no_repeat_ngram_size=3
    )
    print("\nWith No-Repeat N-gram (size 3):", output)

    # Test Case 5: Batch Generation
    print("\n=== Batch Generation Testing ===")
    prompts = [
        "The secret to happiness is",
        "In the year 2050, humans will",
        "The most important invention ever is"
    ]
    
    outputs = generator.generate_text(
        prompt=prompts,
        max_new_tokens=30,
        temperature=1.2,
        top_k=50,
        top_p=0.9
    )
    
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

if __name__ == "__main__":
    test_text_generation()