# This file trains a GPT model on a large dataset (English Wikipedia)
import torch
import time
import tiktoken

from utils.misc import print_lib_versions, set_device
from utils.models import GPTModel, save_model
from utils.plot import plot_losses
from utils.eval import calc_loss_loader, generate_and_print_sample
from training.trainer import train_model_simple, train_model_advanced
from utils.data import create_dataloader_HFD, get_wiki_dataset

# Constants
MODEL_SAVE_PATH = f"gpt-model-wiki-2.pt"

DEVICE_NAME=None # cpu, mps or cuda. If None, will select the best available one
TRAIN_RATIO = 0.9 # Train vs validation dataset ration
NUM_EPOCHS = 1 # Number of epochs (1, since the dataset is already huge)
DATA_BATCH_SIZE = 8 # Batch size. Suggested 8 or less for local training, and 16, 32, 64 or even 128 for training on GPU
SUBSET_RATIO=0.0001 # If you want to train on a smaller subset instead the original dataset. E.g. 0.1 would mean 10% from the original dataset
ADVANCED_TRAINING = True # If True, will use learning rate warmup, cosine decay and gradient clipping.
TEST_BEFORE_TRAINING = False # If True, it will test the model before it is trained (may take some time)
START_CONTEXT = "Water is essential for all forms of life because "
TEMPERATURE = 1.2

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 1024, # Shortened context length (orig: 1024)
    "emb_dim": 1024,        # Embedding dimension
    "n_heads": 16,         # Number of attention heads
    "n_layers": 24,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
} # For this model you will need to make batches at most of size 4, and train on GPU. Locally with batch size 1-2, but will take a long time

MODEL_TO_USE = GPT_CONFIG_124M

TEST_RATIO = 0.01 # For final testing (independent of hyperparameters and model size/architecture)

def main():
    # Print libraries versions
    print_lib_versions()

    print(f"\n{50 * '='}")
    print("\t\tDATA PROCESSING AND PREVIEW")
    print(f"{50 * '='}")

    # Importing and viewing the data
    ds = get_wiki_dataset()
    ds = ds.shuffle(seed=42)  # Shuffle the dataset with a fixed seed for reproducibility

    ds_split = ds.train_test_split(test_size=TEST_RATIO, shuffle=False)
    test_data = ds_split["test"]  # For final testing (independent of hyperparameters and model size/architecture). Not used in the code for noe, can be implemented later
    ds = ds_split["train"]  # Remaining for train/val split later

    print(f"\nTesting dataset size (not used for now): {test_data.num_rows:,}")

    if SUBSET_RATIO and SUBSET_RATIO < 1:
        print(f"Taking {SUBSET_RATIO * 100}% from the original dataset.\n")
        ds = ds.select(range(int(SUBSET_RATIO * ds.num_rows)))

    # Splitting the data
    ds_split = ds.train_test_split(train_size=TRAIN_RATIO, shuffle=False)
    train_data = ds_split["train"]
    val_data = ds_split["test"]

    print(f"Training dataset size: {train_data.num_rows:,}")
    print(f"Validation dataset size: {val_data.num_rows:,}")

    # Initializing tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # For reproducibility due to the shuffling in the data loader
    torch.manual_seed(123)

    # Data loaders (training and validation)
    train_loader = create_dataloader_HFD(
        tokenizer,
        train_data,
        batch_size=DATA_BATCH_SIZE,
        max_length=MODEL_TO_USE["context_length"],
        stride=MODEL_TO_USE["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_HFD(
        tokenizer,
        val_data,
        batch_size=DATA_BATCH_SIZE,
        max_length=MODEL_TO_USE["context_length"],
        stride=MODEL_TO_USE["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print("\nTrain loader:")
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)

    print("\nValidation loader:")
    x, y = next(iter(val_loader))
    print(x.shape, y.shape)

    # Calculating train and validation tokens
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print(f"\nTraining tokens:    {train_tokens:,}")
    print(f"Validation tokens:  {val_tokens:,}")
    print(f"All tokens:         {train_tokens + val_tokens:,}")

    # Selecting device
    print(f"\n{50 * '='}")
    print("\t\tSELECTING DEVICE AND MODEL")
    print(f"{50 * '='}")
    
    device = set_device(DEVICE_NAME)
    print(f"Device: {device}")
    print(f"Model used: {MODEL_TO_USE}")

    # Testing the initial model (before it is trained). It may take some time.
    if TEST_BEFORE_TRAINING:
        print(f"\n{50 * '='}")
        print("\t\tINITIAL MODEL TESTING")
        print(f"{50 * '='}")

        torch.manual_seed(123) # For reproducibility
        start_time_model = time.time()

        model = GPTModel(MODEL_TO_USE)
        model.eval();  # Disable dropout during inference
        model.to(device)
        print(f"Model to device completed in {(time.time() - start_time_model) / 60:.2f} minutes.")

        # Calculating training and validation loss
        start_time_model = time.time()
        with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
            train_loss = calc_loss_loader(train_loader, model, device)
            val_loss = calc_loss_loader(val_loader, model, device)
        print(f"Model eval completed in {(time.time() - start_time_model) / 60:.2f} minutes.")
        print("\nTraining loss:  ", train_loss)
        print("Validation loss:", val_loss)

        # Checking initial model output
        print("\nModel output before training:")
        generate_and_print_sample(
            model, tokenizer, device, START_CONTEXT, TEMPERATURE
        )

    print(f"\n{50 * '='}")
    print("\t\tSTART TRAINING")
    print(f"{50 * '='}")

    # Initializing an empty GPT model (with random weights)
    torch.manual_seed(123) # For reproducibility
    model = GPTModel(MODEL_TO_USE)
    model.to(device)

    # Creating optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    # Double checking that the model is on the right device
    print(f"Model located on device: {next(model.parameters()).device}")

    # Starting the time counter
    start_time = time.time()

    if ADVANCED_TRAINING:
        # Training with advanced techniques like learning rate warmup, cosine decay and gradient clipping.
        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(0.2 * total_steps) # 20% warmup
        
        train_losses, val_losses, tokens_seen, lrs = train_model_advanced(
            model, train_loader, val_loader, optimizer, device, n_epochs=NUM_EPOCHS,
            eval_freq=5, eval_iter=1, start_context=START_CONTEXT,
            tokenizer=tokenizer, warmup_steps=warmup_steps, 
            initial_lr=1e-5, min_lr=1e-5, temperature=TEMPERATURE
        )
    else:
        # Training in a more simple way (without learning rate warmup, cosine decay and gradient clipping).
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=NUM_EPOCHS, eval_freq=5, eval_iter=5,
            start_context=START_CONTEXT, tokenizer=tokenizer, temperature=TEMPERATURE
        )

    # Printing time it took to train
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n\nTraining completed in {execution_time_minutes:.2f} minutes.")

    # Plottong train and val losses per epoch
    plot_losses(NUM_EPOCHS, tokens_seen, train_losses, val_losses)

    # Save the trained model (for furture reuse)
    save_model(MODEL_SAVE_PATH, model, optimizer, train_losses, val_losses, MODEL_TO_USE, NUM_EPOCHS)

if __name__ == "__main__":
    main()
