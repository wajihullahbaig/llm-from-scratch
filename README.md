# GPT-Style Language Model from Scratch

This project trains a GPT-2-level large language model (LLM) from scratch, allowing you to explore how modern LLMs are built and trained. By experimenting with smaller or larger models, as well as small (public-domain text) or large (English Wikipedia) datasets, you can better understand the inner workings and trade-offs involved in training such models.

## Key Features

- **Modular training scripts:**  
  - `train_gpt_txt.py`: Trains on a small text dataset (e.g., "Alice in Wonderland"), suitable for quick local experiments.
  - `train_gpt_wiki.py`: Trains on a large English Wikipedia dataset (~11 GB, ~6.4M articles), requiring a Hugging Face account and API token.

- **Parameter Tuning:**  
  Easily adjust training parameters (batch size, context length, subset ratios, advanced training techniques) directly in the scripts.

- **Model Configurations:**  
  Two sample configurations (124M and 355M parameters) are provided. These configurations mimic GPT-2 scale models and can be adapted depending on your compute resources.

- **Device Flexibility:**  
  The code can detect and run on the best available device (`cpu`, `mps`, `cuda`) for efficient training.

## Setup Instructions

1. **Install Miniconda (Recommended):**  
   [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

2. **Create and Activate the Environment:**
   ```
   conda create -n my_LLM python=3.10.15
   conda activate my_LLM
   pip install -r requirements.txt
   ```

3.	**Hugging Face Login (For Wikipedia Data):**
    ```
    huggingface-cli login
    ```
    You need a Hugging Face account and API token to access large datasets.

## Running the Training

- **Small Dataset (e.g., “Alice in Wonderland”):**

    Ideal for quick local tests and understanding the training loop:

    ```
    python train_gpt_txt.py
    ```

- **Large Dataset (English Wikipedia):**

    For more realistic large-scale training (GPU recommended):
    ```
    python train_gpt_wiki.py
    ```

## Important Parameters
Modify these in the scripts (`train_gpt_txt.py` / `train_gpt_wiki.py`) as needed:
```
DEVICE_NAME = None       # 'cpu', 'mps', 'cuda'; if None, best device is auto-selected
TRAIN_RATIO = 0.9        # Fraction of data for training; remainder for validation
NUM_EPOCHS = 1           # Typically 1 for large datasets; can do more for small sets
DATA_BATCH_SIZE = 8      # Batch size; small for CPU, larger for GPU if memory allows
SUBSET_RATIO = 0.0001    # Use a fraction of the full dataset (e.g., 0.01 = 1%)
ADVANCED_TRAINING = True # Enables LR warmup, cosine decay, grad clipping
TEST_BEFORE_TRAINING = False
START_CONTEXT = "Water is essential for all forms of life because "
TEMPERATURE = 1.2         # Controls randomness in text generation
```

## Model Configurations
- **124M Model (Smaller):**

    More manageable on local machines or smaller GPUs.
    ```
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    ```
- **355M Model (Larger):**

    Requires stronger hardware and smaller batch sizes, but can yield richer results.
    ```
    GPT_CONFIG_355M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    ```
    Set `MODEL_TO_USE` in the script to one of these configurations. If your compute is limited, start with the smaller model and more data for better training outcomes.

## Additional Scripts
- **Use a Trained Model:**
    ```
    python run_model.py
    ```
    Make sure you specify the correct model path and name in run_model.py.

- **GPU vs CPU Testing (If GPU available):**
    ```
    python gpu_debug_test.py
    ```

## Results and Performance (124M model, CPU and MPS on Macbook Pro M2, 32GB RAM; GPU on single A100)
### “Alice in Wonderland” Training:

- CPU: ~3.96 min
- MPS (Apple Silicon): ~1.68 min
- GPU: ~0.27 min (batch=8), ~0.22 min (batch=32)

### English Wikipedia Training (0.01% subset):

- CPU: ~6.94 min
- MPS: ~2.41 min
- GPU: ~0.46 min (batch=8), ~0.4 min (batch=32)

These results show clear performance gains when using GPU acceleration and larger batch sizes. Even a small fraction of the Wikipedia dataset produces coherent text. Using the 124M model on a fraction of Wikipedia (1% of data, ~30 million tokens) for ~37 minutes on a GPU yields significantly more logical and coherent text than training solely on a small dataset.

## Tips
- **Smaller Model + More Data:** Often produces more coherent results than a larger model trained on very limited data.
- **Advanced Training Techniques:** Try enabling `ADVANCED_TRAINING` for learning rate warmup, cosine decay, and gradient clipping to improve stability and results.
- **Experimentation is Key:** Adjust batch sizes, subsets of data, and training time to see what fits your compute resources and desired outcome.

## Further Reading
For additional insights and inspiration, check out the [LLMs-from-scratch GitHub repository](https://github.com/rasbt/LLMs-from-scratch).