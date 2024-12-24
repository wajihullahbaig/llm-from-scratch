import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict
import numpy as np

class TextGenerator:
    def __init__(self, model, tokenizer, context_size: int):
        """
        Initialize the text generator with a model, tokenizer and context size.
        
        Args:
            model: The transformer model for text generation
            tokenizer: The tokenizer for converting between text and token ids
            context_size: Maximum context length the model can handle
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.device = next(model.parameters()).device  # Get device from model
        
    def generate(self,
                input_ids: torch.Tensor,
                max_new_tokens: int,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                repetition_penalty: float = 1.0,
                do_sample: bool = True,
                num_beams: int = 1,
                early_stopping: bool = True,
                no_repeat_ngram_size: int = 0) -> torch.Tensor:
        """
        Generate token ids using various decoding strategies.
        
        Args:
            input_ids: Input token ids (batch_size, sequence_length)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: If False, use greedy decoding
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop beam search when enough valid candidates are found
            no_repeat_ngram_size: Size of n-grams that shouldn't be repeated
            
        Returns:
            torch.Tensor: Generated token ids
        """
        # Ensure input_ids are on the correct device
        input_ids = input_ids.to(self.device)
        
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        else:
            return self._generate_sample_or_greedy(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        # Ensure input_ids are on the correct device
        input_ids = input_ids.to(self.device)
        """
        Generate text using various decoding strategies.
        
        Args:
            input_ids: Input token ids (batch_size, sequence_length)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep minimum number of tokens whose cumulative probability exceeds p
            repetition_penalty: Penalty for repeating tokens
            do_sample: If False, use greedy decoding
            num_beams: Number of beams for beam search (1 = no beam search)
            early_stopping: Whether to stop beam search when enough valid candidates are found
            no_repeat_ngram_size: Size of n-grams that shouldn't be repeated
            
        Returns:
            torch.Tensor: Generated token ids
        """
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        else:
            return self._generate_sample_or_greedy(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

    def _generate_sample_or_greedy(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: float = 1.0,
            do_sample: bool = True,
            no_repeat_ngram_size: int = 0) -> torch.Tensor:
        """
        Generate text using either sampling or greedy decoding.
        Returns: torch.Tensor of token ids
        """
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get the context window
            context = generated[:, -self.context_size:]
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(context)
            
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for seq in generated:
                    for previous_token in seq.unique():
                        next_token_logits[:, previous_token] /= repetition_penalty
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0:
                banned_tokens = self._get_banned_ngram_tokens(
                    generated, no_repeat_ngram_size, generated.shape[1]
                )
                for banned in banned_tokens:
                    next_token_logits[:, banned] = -float('inf')
            
            if do_sample:
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
        """
        Generate text using either sampling or greedy decoding.
        """
        generated = input_ids.clone().to(self.device)
        
        for _ in range(max_new_tokens):
            # Get the context window
            context = generated[:, -self.context_size:]
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(context)
            
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for seq in generated:
                    for previous_token in seq.unique():
                        next_token_logits[:, previous_token] /= repetition_penalty
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0:
                banned_tokens = self._get_banned_ngram_tokens(
                    generated, no_repeat_ngram_size, generated.shape[1]
                )
                for banned in banned_tokens:
                    next_token_logits[:, banned] = -float('inf')
            
            if do_sample:
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
        return self.decode(generated)

    def _generate_beam_search(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int,
            num_beams: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: float = 1.0,
            early_stopping: bool = True,
            no_repeat_ngram_size: int = 0) -> torch.Tensor:
        """
        Generate text using beam search decoding.
        """
        batch_size = input_ids.shape[0]
        
        # Expand input to num_beams
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, -1)
        
        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            context = generated[:, -self.context_size:]
            
            with torch.no_grad():
                logits = self.model(context)
            
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature and repetition penalty
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if repetition_penalty != 1.0:
                for seq in generated:
                    for previous_token in seq.unique():
                        next_token_logits[:, previous_token] /= repetition_penalty
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0:
                banned_tokens = self._get_banned_ngram_tokens(
                    generated, no_repeat_ngram_size, generated.shape[1]
                )
                for banned in banned_tokens:
                    next_token_logits[:, banned] = -float('inf')
            
            # Apply top-k and top-p filtering
            if top_k is not None or top_p is not None:
                next_token_logits = self._filter_logits(next_token_logits, top_k, top_p)
            
            # Calculate log probabilities
            scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_scores = scores + beam_scores[:, None]
            
            # Reshape for beam search
            next_scores = next_scores.view(batch_size, num_beams * next_scores.shape[-1])
            
            # Get top-k scores and tokens
            next_scores, next_tokens = torch.topk(
                next_scores, num_beams, dim=1, largest=True, sorted=True
            )
            
            # Convert token indices
            next_beam_indices = next_tokens // scores.shape[-1]
            next_tokens = next_tokens % scores.shape[-1]
            
            # Prepare next iteration
            beam_outputs = torch.cat([
                generated[next_beam_indices.view(-1)],
                next_tokens.view(-1, 1)
            ], dim=-1)
            
            generated = beam_outputs
            beam_scores = next_scores.view(-1)
            
            # Early stopping if all beams have EOS
            if early_stopping and self._is_generation_done(generated, batch_size, num_beams):
                break
        
        # Select best beam
        generated = generated.view(batch_size, num_beams, -1)
        best_beam = beam_scores.view(batch_size, num_beams).argmax(dim=1)
        generated = generated[torch.arange(batch_size), best_beam]
        
        return generated  # Return token ids only

    def _filter_logits(
            self,
            logits: torch.Tensor,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits."""
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
            
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
            
        return logits

    def _get_banned_ngram_tokens(
            self,
            generated: torch.Tensor,
            no_repeat_ngram_size: int,
            cur_len: int) -> List[List[int]]:
        """Get list of banned tokens to prevent repetition of n-grams."""
        banned_tokens = []
        
        for beam_idx in range(generated.shape[0]):
            generated_ngrams = {}
            for i in range(cur_len - no_repeat_ngram_size + 1):
                ngram = tuple(generated[beam_idx, i:i + no_repeat_ngram_size].tolist())
                generated_ngrams[ngram] = generated_ngrams.get(ngram, []) + [i]
            
            banned_tokens_beam = []
            for ngram, prev_positions in generated_ngrams.items():
                if len(prev_positions) > 1:
                    banned_tokens_beam.append(ngram[-1])
            
            banned_tokens.append(banned_tokens_beam)
        
        return banned_tokens

    def decode(self, token_ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Convert token ids to text using tiktoken decoder.
        
        Args:
            token_ids: Token ids as tensor, list, or nested list
            
        Returns:
            Union[str, List[str]]: Decoded text (single string if single sequence, list of strings otherwise)
        """
        # Convert input to list format
        if isinstance(token_ids, torch.Tensor):
            token_lists = token_ids.detach().cpu().numpy().tolist()
        elif isinstance(token_ids, list):
            # Check if it's a nested list or single list
            if token_ids and isinstance(token_ids[0], list):
                token_lists = token_ids
            else:
                token_lists = [token_ids]
        else:
            raise TypeError(f"token_ids must be tensor or list, not {type(token_ids)}")
        
        # Decode each sequence in the batch
        decoded_texts = []
        for tokens in token_lists:
            # Remove padding tokens (0s)
            tokens = [t for t in tokens if t != 0]
            
            # Convert tokens to text using tiktoken's decode
            text = self.tokenizer.decode(tokens)
            decoded_texts.append(text)
        
        # Return single string for single sequence, list otherwise
        return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Convert text to token ids using tiktoken encoder.
        
        Args:
            text: Input text (string or list of strings)
            
        Returns:
            torch.Tensor: Encoded token ids
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        # Encode each text sequence
        encoded_sequences = []
        for t in text:
            # tiktoken's encode method returns a list of integers
            tokens = self.tokenizer.encode(t)
            
            # Truncate if longer than context size
            if len(tokens) > self.context_size:
                tokens = tokens[:self.context_size]
                
            encoded_sequences.append(tokens)
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in encoded_sequences)
        padded_sequences = []
        
        for seq in encoded_sequences:
            # Pad with zeros if necessary
            padded_seq = seq + [0] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        # Convert to tensor and move to correct device
        return torch.tensor(padded_sequences, device=self.device)

    def generate_text(
            self,
            prompt: Union[str, List[str]],
            max_new_tokens: int,
            **kwargs) -> Union[str, List[str]]:
        """
        Generate text from a prompt string.
        
        Args:
            prompt: Input text prompt(s)
            max_new_tokens: Number of new tokens to generate
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Union[str, List[str]]: Generated text
        """
        # Encode the prompt to token ids
        input_ids = self.encode(prompt)
        
        # Generate new tokens (returns token ids)
        generated_ids = self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # Decode the generated tokens to text
        return self.decode(generated_ids)

    def _is_generation_done(
            self,
            generated: torch.Tensor,
            batch_size: int,
            num_beams: int,
            eos_token_id: int = None) -> bool:
        """Check if generation is done based on EOS token."""
        if eos_token_id is None:
            return False
            
        generated = generated.view(batch_size, num_beams, -1)
        for batch_idx in range(batch_size):
            for beam_idx in range(num_beams):
                if eos_token_id not in generated[batch_idx, beam_idx]:
                    return False
        return True