# Helper functions for model training and weights updating
from configs.configuration_manager import ConfigurationManager
from text_generation.text_generator import TextGenerator
from utils.eval import calc_loss_batch, evaluate_model
import math
import torch
from logging import Logger

def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       eval_freq, eval_iter,  tokenizer, llm_configs:ConfigurationManager = None, logger:Logger=None):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(llm_configs.get_setting("training.num_epochs")):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if global_step % (eval_freq * 5) == 0:
                # Generate and print a sample from the model to monitor progress
                tg = TextGenerator(model, tokenizer, context_size=llm_configs.get_setting("model_configs.context_length"))
                # Generate from text prompt
                output_text = generate_and_get_text(tg, llm_configs)
                logger.info(output_text)

    return train_losses, val_losses, track_tokens_seen

# This version of training includes learning rate warmup, cosine decay and gradient clipping
def train_model_advanced(model, train_loader, val_loader, optimizer, device,
                eval_freq, eval_iter, tokenizer,
                warmup_steps=1.0, initial_lr=3e-05, min_lr=1e-6, llm_configs:ConfigurationManager=None,logger:Logger = None):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * llm_configs.get_setting("training.num_epochs")

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(llm_configs.get_setting("training.num_epochs")):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # Calculate and backpropagate the loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping after the warmup phase to avoid exploding gradients
            if global_step >= warmup_steps:  # the book originally used global_step > warmup_steps, which lead to a skipped clipping step after warmup
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            tokens_seen += input_batch.numel()

            # Periodically evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                logger.info(f"Epoch {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

            if global_step % (eval_freq * 5) == 0:
                # Generate and print a sample from the model to monitor progress
                tg = TextGenerator(model, tokenizer, context_size=llm_configs.get_setting("model_configs.context_length"))
                # Generate from text prompt
                output_text = generate_and_get_text(tg, llm_configs)
                logger.info(output_text)

    return train_losses, val_losses, track_tokens_seen, track_lrs

def generate_and_get_text(generator:TextGenerator,llm_configs:ConfigurationManager) -> str:
        output_text = generator.generate_text(
                    prompt=llm_configs.get_setting("inference.start_context"),
                    max_new_tokens=llm_configs.get_setting("inference.max_new_tokens"),
                    temperature=llm_configs.get_setting("inference.temperature"),                    
                    num_beams = llm_configs.get_setting("inference.num_beams"),
                    early_stopping = llm_configs.get_setting("inference.early_stopping"),
                    no_repeat_ngram_size = llm_configs.get_setting("inference.no_repeat_ngram_size"),
                    top_k=llm_configs.get_setting(f"inference.top_k"),
                    top_p=llm_configs.get_setting(f"inference.top_p"),                    
                    repetition_penalty=llm_configs.get_setting(f"inference.repetition_penalty"),                    
                    do_sample=llm_configs.get_setting(f"inference.do_sample"),                    
                    ) 

        return output_text