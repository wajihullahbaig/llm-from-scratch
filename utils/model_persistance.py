# Implementation of the GPT model and it's components
from logging import Logger
import torch
import torch.nn as nn

from models.gpt2_model import GPTModel

def save_model(model_save_path, model, optimizer, train_losses, val_losses, config, num_epochs, logger:Logger):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'epochs': num_epochs
    }, model_save_path)
    logger.info(f"\nModel saved to {model_save_path}")

def load_model(model_path):
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path)
    
    # Create a new model with the saved config
    model = GPTModel(checkpoint['config'])
    
    # Load the saved model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model
