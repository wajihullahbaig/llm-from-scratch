from logging import Logger
import torch
from torchinfo import summary
from configs.configuration_manager import ConfigurationManager
from models.gpt2_model import GPTModel

def print_model_summary(cfg:ConfigurationManager,logger:Logger, model:GPTModel) -> None:
    """
    Print a detailed summary of the GPT model architecture
    
    """
    
    # Calculate input shape based on config
    context_length = cfg.get_setting("model_configs.context_length")
    batch_size = cfg.get_setting("training.batch_size")
    vocab_size = cfg.get_setting("model_configs.vocab_size")
    
 # Create a dummy input tensor with valid token indices
    # Using values between 0 and vocab_size-1
    dummy_input = torch.randint(
        0, vocab_size-1, 
        (batch_size, context_length), 
        dtype=torch.long
    )
    
    # Generate model summary
    try:
        model_summary = summary(
            model,
            input_data=dummy_input,  # Using input_data instead of input_size
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=4,  # Increase this to see more nested layers
            device=torch.device('cpu'),  # Change to 'cuda' if using GPU
            verbose=2  # Increased verbosity for debugging
        )
        
        # Print additional model statistics
        logger.info("Model Statistics:")
        logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Print architecture details
        logger.info("Architecture Details:")
        logger.info(f"Number of Transformer Blocks: {cfg.get_setting('model_configs.n_layers')}")
        logger.info(f"Number of Attention Heads: {cfg.get_setting('model_configs.n_heads')}")
        logger.info(f"Embedding Dimension: {cfg.get_setting('model_configs.emb_dim')}")
        logger.info(f"Vocabulary Size: {cfg.get_setting('model_configs.vocab_size')}")
        logger.info(f"Context Length: {cfg.get_setting('model_configs.context_length')}")
        
        return model_summary
        
    except Exception as e:
        logger.info(f"Error in generating model summary: {str(e)}")
        logger.info("Dumping model structure manually:")
        logger.info(model)
        logger.info("Model Parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
    
