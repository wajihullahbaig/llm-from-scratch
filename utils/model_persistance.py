# Implementation of the GPT model and it's components
from logging import Logger
import torch
import torch.nn as nn

from configs.configuration_manager import ConfigurationManager
from models.gpt2_model import GPTModel

import torch
import torch.nn as nn
from logging import Logger
from typing import Dict, Optional, Union

class ModelConfig:
    """Configuration class to replace ConfigurationManager"""
    def __init__(self, config_dict: Dict):
        self.model_type = config_dict.get('model_type', 'gpt2')
        self.vocab_size = config_dict.get('vocab_size', 50257)  
        self.emb_dim = config_dict.get('emb_dim', 768)
        self.context_length = config_dict.get('context_length', 1024)
        self.n_heads = config_dict.get('n_heads', 12)
        self.n_layers = config_dict.get('n_layers', 12)
        self.drop_rate = config_dict.get('drop_rate', 0.1)
        self.qkv_bias = config_dict.get('qkv_bias', False)

    def to_dict(self) -> Dict:
        return {
            'model_type':self.model_type,
            'vocab_size': self.vocab_size,
            'emb_dim': self.emb_dim,
            'context_length': self.context_length,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'drop_rate': self.drop_rate,
            'qkv_bias': self.qkv_bias
        }

def save_model(
    model_save_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_losses: list,
    val_losses: list,
    config_dict: Dict,
    num_epochs: int,
    logger: Optional[Logger] = None
) -> None:
    """
    Save model checkpoint with configuration parameters.
    
    Args:
        model_save_path: Path to save the model
        model: The GPT model instance
        optimizer: The optimizer instance
        train_losses: List of training losses
        val_losses: List of validation losses
        config_dict: Dictionary containing model configuration
        num_epochs: Number of epochs trained
        logger: Optional logger instance
    """
    model_config = ModelConfig(config_dict)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': model_config.to_dict(),
        'epochs': num_epochs
    }
    
    torch.save(checkpoint, model_save_path)
    if logger:
        logger.info(f"Model saved to {model_save_path}")
    else:
        print(f"Model saved to {model_save_path}")

def load_model(model_path: str) -> Union[nn.Module, str]:
    """
    Load a saved model checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        The loaded GPT model in evaluation mode
    """
    checkpoint = torch.load(model_path)
    config = ModelConfig(checkpoint['config']).to_dict()
    
    # Initialize model with loaded config
    model = GPTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model,config

# Example usage:
def load_and_prepare_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load and prepare model for inference.
    
    Args:
        model_path: Path to the saved model
        device: torch.device instance
        
    Returns:
        Loaded model on specified device
    """
    model = load_model(model_path)
    model.to(device)
    return model