# This file trains a GPT model on a small txt dataset ("Alice in Wonderland" novel). Good fors trying out locally.
import os
import torch
import time
import tiktoken
import logging 
import sys

from data.dataset_processors import DataLoaderConfig, DatasetProcessor
from utils.misc import print_lib_versions, set_device
from utils.model_summary import print_model_summary
from utils.models import GPTModel, save_model
from utils.plot import plot_losses
from utils.eval import calc_loss_loader, generate_and_print_sample
from training.trainer import train_model_simple, train_model_advanced
from utils.seeding import set_seed
from configs.configuration_manager import ConfigurationManager
from data.downloaders import CorpusDownloader
from models.gpt2_model import GPTModel

def main():
    set_seed(42)
    # Setup logging 
    logger_name = "GPT2-Trainer"
    logger = logging.getLogger("GPT2-Trainer")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - {logger_name} - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Let's pick the yaml config file
    script_name = sys.argv[0]  
    yaml_config_file = sys.argv[2]        
    logger.info(f"Running: {script_name}")
    logger.info(f"YAML config is: {yaml_config_file}")

    # Print libraries versions
    print_lib_versions(logger)

    logger.info(f"{50 * '='}")
    logger.info("\t\tDATA PROCESSING AND PREVIEW")
    logger.info(f"{50 * '='}")

    llm_configs = ConfigurationManager(yaml_config_file)     
    cdl = CorpusDownloader()
    text_data = cdl.download_alice_wonderland()
    

    if llm_configs.get_setting('training.subset_ratio') and llm_configs.get_setting('training.subset_ratio') < 1:
        logger.info(f"Taking {llm_configs.get_setting('training.subset_ratio') * 100}% from the original dataset.\n")
        text_data = text_data[:int(len(text_data) * llm_configs.get_setting('training.subset_ratio'))]

    logger.info("First 100 characters:")
    logger.info(text_data[:99])

    logger.info("Last 100 characters:")
    logger.info(text_data[-99:])

    # Initializing tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Calculating total tokens and characters
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    logger.info(f"Total characters: {total_characters:,}")
    logger.info(f"Total tokens:     {total_tokens:,}")

    # Splitting the data
    split_idx = int(llm_configs.get_setting('training.test_train_ratio') * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    logger.info(f"Train data length: {len(train_data) }")
    logger.info(f"Validation data length: {len(val_data) }")

    # Sanity check
    if total_tokens * (llm_configs.get_setting("training.test_train_ratio")) < llm_configs.get_setting("model_configs.context_length"):
        logger.info("\nNot enough tokens for the training loader. "
            "Try to lower the 'context_length' or "
            "increase the `training_ratio`")
        return

    if total_tokens * (1-llm_configs.get_setting("training.test_train_ratio")) <  llm_configs.get_setting("model_configs.context_length"):
        logger.info("Not enough tokens for the validation loader. "
            "Try to lower the 'context_length' or "
            "decrease the `training_ratio`")
        return

    # Load data via configurations
    # Create configuration object
    dataloader_config = DataLoaderConfig(
        batch_size=llm_configs.get_setting('training.batch_size'),
        max_length=llm_configs.get_setting("model_configs.context_length"),
        stride=llm_configs.get_setting("model_configs.context_length"),
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # Create data loader for training
    train_loader = DatasetProcessor.create_text_dataloader(
        tokenizer=tokenizer,
        text=train_data,
        config=dataloader_config
    )

    dataloader_config.drop_last = False
    dataloader_config.shuffle = False
     # Create data loader for validation
    val_loader = DatasetProcessor.create_text_dataloader(
        tokenizer=tokenizer,
        text=val_data,
        config=dataloader_config
    )
    
    # Printing out shapes of data in data loaders for transparency
    logger.info("Train loader:")
    for x, y in train_loader:
        logger.info(f"{x.shape}, {y.shape}") # Features([batch_size, context_length]), Target([batch_size, context_length])

    logger.info("Validation loader:")
    for x, y in val_loader:
        logger.info(f"{x.shape}, {y.shape}") # Features([batch_size, context_length]), Target([batch_size, context_length])


    # Calculating train and validation tokens
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    logger.info(f"Training tokens:    {train_tokens:,}")
    logger.info(f"Validation tokens:  {val_tokens:,}")
    logger.info(f"All tokens:         {train_tokens + val_tokens:,}")

    # Selecting device
    logger.info(f"{50 * '='}")
    logger.info("\t\tSELECTING DEVICE AND MODEL")
    logger.info(f"{50 * '='}")

    device = set_device(llm_configs.get_setting('model.device'))
    logger.info(f"Device: {device}")
    logger.info(f"Model used: {llm_configs.get_setting('model.model_type')}")

    # Testing the initial model (before it is trained). It may take some time.
    if llm_configs.get_setting('training.test_before_training'):
        logger.info(f"{50 * '='}")
        logger.info("\t\tINITIAL MODEL TESTING")
        logger.info(f"{50 * '='}")

        start_time_model = time.time()

        model = GPTModel(llm_configs)
        print_model_summary(llm_configs, logger, model)
        model.eval();  # Disable dropout during inference
        model.to(device)
        logger.info(f"Model to device completed in {(time.time() - start_time_model) / 60:.2f} minutes.")

        # Calculating training and validation loss
        start_time_model = time.time()
        with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
            train_loss = calc_loss_loader(train_loader, model, device)
            val_loss = calc_loss_loader(val_loader, model, device)
        logger.info(f"Model eval completed in {(time.time() - start_time_model) / 60:.2f} minutes.")
        logger.info(f"Training loss: {train_loss}")
        logger.info(f"Validation loss: { val_loss}")

        # Checking initial model output
        logger.info("Model output before training:")
        generate_and_print_sample(
            model, tokenizer, device, llm_configs.get_setting('inference.start_context'), llm_configs.get_setting('inference.temperature')
        )

    
    logger.info(f"{50 * '='}")
    logger.info("\t\tSTART TRAINING")
    logger.info(f"{50 * '='}")

    # Initializing an empty GPT model (with random weights)
    model = GPTModel(llm_configs)
    print_model_summary(llm_configs, logger, model)        
    model.to(device)

    # Creating optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    # Double checking that the model is on the right device
    logger.info(f"Model located on device: {next(model.parameters()).device}\n")

    # Starting the time counter
    start_time = time.time()

    if llm_configs.get_setting('training.advanced_training'):
        # Training with advanced techniques like learning rate warmup, cosine decay and gradient clipping.
        total_steps = len(train_loader) * llm_configs.get_setting('training.num_epochs')
        warmup_steps = int(0.2 * total_steps) # 20% warmup
        
        train_losses, val_losses, tokens_seen, lrs = train_model_advanced(
            model, train_loader, val_loader, 
            optimizer, device, n_epochs=llm_configs.get_setting('training.num_epochs'),
            eval_freq=5, 
            eval_iter=1, 
            start_context=llm_configs.get_setting('inference.start_context'),
            tokenizer=tokenizer, warmup_steps=warmup_steps, 
            initial_lr=1e-8, min_lr=1e-5, temperature= llm_configs.get_setting('inference.temperature')
        )
    else:
        # Training in a more simple way (without learning rate warmup, cosine decay and gradient clipping).
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=llm_configs.get_setting('training.num_epochs'), 
            eval_freq=5, 
            eval_iter=5,
            start_context=llm_configs.get_setting('inference.start_context'), 
            tokenizer=tokenizer, 
            temperature= llm_configs.get_setting('inference.temperature')
        )

    # Printing time it took to train
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Plottong train and val losses per epoch
    plot_losses(llm_configs.get_setting('training.num_epochs'), tokens_seen, train_losses, val_losses)

    # Save the trained model (for furture reuse)
    base_path = os.path.join(llm_configs.get_setting('model.output_dir'),llm_configs.get_setting('model.model_type') )
    os.makedirs(base_path, exist_ok=True)
    full_path = os.path.join(base_path, llm_configs.get_setting('model.save_name'))
    save_model(full_path, model, optimizer, train_losses, val_losses, llm_configs.get_setting('model.model_type'), llm_configs.get_setting('training.num_epochs'),logger)

if __name__ == "__main__":
    main()
