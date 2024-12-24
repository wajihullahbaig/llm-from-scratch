# This file trains a GPT model on a large dataset (English Wikipedia)
import os
import torch
import time
import tiktoken
import logging 
import sys

from data.dataset_processors import DataLoaderConfig, DatasetProcessor
from utils.misc import print_lib_versions, set_device
from utils.model_summary import print_model_summary
from utils.model_persistance import GPTModel, save_model
from utils.plot import plot_losses
from utils.eval import calc_loss_loader
from text_generation.text_generator import TextGenerator
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
    ds = cdl.download_wikipedia()
    
    ds_split = ds.train_test_split(test_size=llm_configs.get_setting("training.test_train_ratio"), shuffle=False)
    test_data = ds_split["test"]  # For final testing (independent of hyperparameters and model size/architecture). Not used in the code for noe, can be implemented later
    ds = ds_split["train"]  # Remaining for train/val split later

    logger.info(f"Testing dataset size (not used for now): {test_data.num_rows:,}")

    if llm_configs.get_setting('training.subset_ratio') and llm_configs.get_setting('training.subset_ratio') < 1:
        logger.info(f"Taking {llm_configs.get_setting('training.subset_ratio') * 100}% from the original dataset.\n")
        ds = ds.select(range(int(llm_configs.get_setting('training.subset_ratio')  * ds.num_rows)))

    # Splitting the data
    ds_split = ds.train_test_split(train_size=llm_configs.get_setting("training.test_train_ratio"), shuffle=False)
    train_data = ds_split["train"]
    val_data = ds_split["test"]

    logger.info(f"Training dataset size: {train_data.num_rows:,}")
    logger.info(f"Validation dataset size: {val_data.num_rows:,}")

    # Initializing tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create configuration object for dataloading
    dataloader_config = DataLoaderConfig(
        batch_size=llm_configs.get_setting('training.batch_size'),
        max_length=llm_configs.get_setting("model_configs.context_length"),
        stride=llm_configs.get_setting("model_configs.context_length"),
        drop_last=True,
        shuffle=False,
        num_workers=0
    )
    
    # Create data loader for training
    train_loader = DatasetProcessor.create_wiki_dataloader(
        tokenizer=tokenizer,
        dataset=train_data,
        config=dataloader_config
    )

    dataloader_config.drop_last = False
    dataloader_config.shuffle = False
     # Create data loader for validation
    val_loader = DatasetProcessor.create_wiki_dataloader(
        tokenizer=tokenizer,
        dataset=val_data,
        config=dataloader_config
    )

    logger.info("Train loader:")
    x, y = next(iter(train_loader))
    logger.info(f"{x.shape}, {y.shape}")

    logger.info("Validation loader:")
    x, y = next(iter(val_loader))
    logger.info(f"{x.shape}, {y.shape}")

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
    if llm_configs.get_setting("training.test_before_training"):
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
        logger.info(f"Training loss:  {train_loss}")
        logger.info(f"Validation loss: {val_loss}")

        # Checking initial model output
        logger.info("Model output before training:")
        tg = TextGenerator(model, tokenizer, context_size=llm_configs.get_setting("model_configs.context_length"))
        # Generate from text prompt
        output_text = tg.generate_text(
            prompt=llm_configs.get_setting("inference.start_context"),
            max_new_tokens=llm_configs.get_setting("inference.max_new_tokens"),
            temperature=llm_configs.get_setting("inference.temperature"),
            top_k=llm_configs.get_setting("inference.top_k"),
            top_p=llm_configs.get_setting("inference.top_p")
        )
        logger.info(output_text)


    logger.info(f"{50 * '='}")
    logger.info("\t\tSTART TRAINING")
    logger.info(f"{50 * '='}")

    model = GPTModel(llm_configs)
    print_model_summary(llm_configs, logger, model)        
    model.to(device)

    # Creating optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    # Double checking that the model is on the right device
    logger.info(f"Model located on device: {next(model.parameters()).device}")

    # Starting the time counter
    start_time = time.time()

    if llm_configs.get_setting("training.advanced_training"):
        # Training with advanced techniques like learning rate warmup, cosine decay and gradient clipping.
        total_steps = len(train_loader) * llm_configs.get_setting("training.num_epochs")
        warmup_steps = int(0.2 * total_steps) # 20% warmup
        
        train_losses, val_losses, tokens_seen, lrs = train_model_advanced(
            model, train_loader, val_loader, optimizer, device, 
            eval_freq=5, eval_iter=1, 
            tokenizer=tokenizer, warmup_steps=warmup_steps, 
            initial_lr=1e-5, min_lr=1e-5, llm_configs=llm_configs,logger=logger
        )
    else:
        # Training in a more simple way (without learning rate warmup, cosine decay and gradient clipping).
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            eval_freq=5, eval_iter=5,
            tokenizer=tokenizer, llm_configs=llm_configs,logger=logger
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
