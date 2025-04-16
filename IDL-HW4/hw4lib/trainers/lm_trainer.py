from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from ..utils import create_scheduler
from ..decoding.sequence_generator import SequenceGenerator

class LMTrainer(BaseTrainer):
    """
    Language Model Trainer class that handles the training, validation, and generation loops.

    This trainer implements:
    1. Training loop with gradient accumulation and mixed precision training
    2. Validation loop for model evaluation
    3. Generation capabilities with different decoding strategies

    You only need to fill in the TODOs in the code. 
    Please do not modify any other code without understanding what you are doing.
    
    Implementation Tasks:
    - TODO: Initialize the criterion in __init__
    - TODO: Implement key parts of the training loop in _train_epoch
    - TODO: Use your greedy generation implementation in generate
    - TODO: Implement key parts of the the validation loop in _validate_epoch
    - TODO: Implement key parts of the full training loop in train

    Implementation Notes:
    1. For __init__:
        - Initialize CrossEntropyLoss with appropriate padding index and label smoothing
        
    2. For _train_epoch:
        - Unpack the batch (shifted inputs, golden targets, lengths)
        - Get model predictions and attention weights
        - Calculate loss
        
    3. For _validate_epoch:
        - Similar to _train_epoch but without gradient calculations
        - Use torch.inference_mode() for validation
        
    4. For train:
        - Implement the epoch loop with training and validation and generation
        
    5. For generate:
        - Use the greedy decoding method you implemented in SequenceGenerator
        - Post-process sequences using appropriate tokenizer methods
        - Format results
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        # TODO: Implement the __init__ method
        # TODO: Initialize the criterion
        # How would you set the ignore_index? 
        # Use value in config to set the label_smoothing argument
        self.criterion = NotImplementedError
        raise NotImplementedError # Remove once implemented

    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Training metrics and attention weights
        """

        # TODO: In-fill the _train_epoch method
        raise NotImplementedError # Remove once implemented
        
        # Initialize training variables
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Training LM]")
        running_ce_loss = 0.0
        total_tokens = 0

        # Only zero gradients when starting a new accumulation cycle
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # TODO: Unpack batch from the dataloader
            # TODO: Move the batch elements to self.device
            targets_shifted, targets_golden, lengths = batch
        

            with torch.autocast(device_type=self.device, dtype=torch.float16):

                # TODO: Get raw logits and attention weights from model
                raw_preds, attn_weights = NotImplementedError

                # TODO: Calculate raw loss first
                # What is the shape of raw_preds and targets_golden? 
                # Would you need to change the shape of the inputs to the criterion?
                # Hint: See the documentation for CrossEntropyLoss
                raw_loss = NotImplementedError
                
            # Calculate metrics with raw loss (DO NOT MODIFY THIS)
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += raw_loss.item() * batch_tokens

            # Normalize loss for gradient accumulation
            loss = raw_loss / self.config['training']['gradient_accumulation_steps']
            
            # TODO: Backpropagate the loss
            self.scaler = NotImplementedError
        
            # Only update weights after accumulating enough gradients
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                # Only step scheduler here if it's not ReduceLROnPlateau
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()  # Reset gradients after update

            # Calculate metrics
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Clean up
            del targets_shifted, targets_golden, lengths, raw_preds, loss
            torch.cuda.empty_cache()

        # Handle any remaining gradients at the end of the epoch
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            # Only step scheduler here if it's not ReduceLROnPlateau
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        return {
            'ce_loss_token': avg_ce_loss,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, attn_weights
            
            
    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        
        Args:
            dataloader: DataLoader for validation data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Validation metrics and attention weights
        """

        # TODO: In-fill the _validate_epoch method

        raise NotImplementedError # Remove once implemented
        
        # Initialize validation variables
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Validating LM]")
        running_ce_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(dataloader):
            # TODO: Unpack batch
            # TODO: Move the batch elements to self.device
            targets_shifted, targets_golden, lengths = batch

            # Forward pass
            with torch.inference_mode():
                # TODO: Get raw predictions and attention weights from model
                raw_preds, attn_weights = NotImplementedError

                # TODO: Calculate loss
                # What is the shape of raw_preds and targets_golden? 
                # Would you need to change the shape of the inputs to the criterion?
                # Hint: See the documentation for CrossEntropyLoss
                loss = NotImplementedError

            # Calculate metrics
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += loss.item() * batch_tokens

            # Update the progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
            )
            batch_bar.update()

            # Clean up
            del targets_shifted, targets_golden, lengths, raw_preds, loss
            torch.cuda.empty_cache()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        return {
            'ce_loss_token': avg_ce_loss,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, attn_weights
        

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for language model training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: int, number of epochs to train
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")
        
        # TODO: In-fill the train method
        raise NotImplementedError # Remove once implemented

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            
            # TODO: Train for one epoch
            train_metrics, train_attn = NotImplementedError
            
            # TODO: Validate
            val_metrics, val_attn = NotImplementedError

            # TODO: Generate with the validation set
            gen_results = NotImplementedError
            
            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['ce_loss_char'])

            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)
            
            # Save attention plots
            train_attn_keys = list(train_attn.keys())
            val_attn_keys = list(val_attn.keys())
            self._save_attention_plot(train_attn[train_attn_keys[0]][0], epoch, "train_self")
            self._save_attention_plot(val_attn[val_attn_keys[0]][0], epoch, "val_self")

            # Save generated text
            self._save_generated_text(gen_results, f'val_epoch_{epoch}')

            # Save checkpoints
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Check if this is the best model
            val_loss = val_metrics['ce_loss_char']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metric = val_loss
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1


    def evaluate(self, test_dataloader):
        """
        Evaluate the model on the test set.
        
        Args:
            test_dataloader: DataLoader for test data
        Returns:
            Tuple[Dict[str, float], Dict[str, Dict[str, Dict]]]: A tuple containing:
                - test_metrics: Test metrics
                - generation_results: Generation results for each config
        """
        test_metrics, test_attn = self._validate_epoch(test_dataloader)

        # Log metrics
        metrics = {
            'test': test_metrics
        }
        self._log_metrics(metrics, self.current_epoch)  

        # Save attention plots
        test_attn_keys = list(test_attn.keys())
        self._save_attention_plot(test_attn[test_attn_keys[0]][0], self.current_epoch, "test_self")

        # Generate with evaluation configs and collect results
        generation_results = {}
        eval_configs = self._get_evaluation_generation_configs()
        for config_name, config in eval_configs.items():
            try:
                gen_results = self.generate(test_dataloader, generation_config=config)
                generation_results[config_name] = gen_results
                self._save_generated_text(gen_results, f'test_epoch_{self.current_epoch}_{config_name}')
            except Exception as e:
                print(f"Could not generate results for {config_name}: {e}")
                continue
        return test_metrics, generation_results

    def generate(self, dataloader, generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating sequences from prompts.
        
        Args:
            dataloader: DataLoader containing the evaluation data
            generation_config: Optional dictionary containing generation parameters:
                - num_samples: int, number of samples to generate
                - prompt_length: int, length of prompts
                - seed: int, random seed
                - max_length: int, maximum sequence length
                - temperature: float, sampling temperature
                - beam_width: int, beam search width
                - repeat_penalty: float, penalty for repeated tokens
                - top_k: int, top-k filtering value
                - top_p: float, nucleus sampling threshold
        Returns:
            Dict containing generation results with prompts, originals, and generated sequences
        """

        # TODO: In-fill the generate method
        # You just need to implement the greedy search generation
        # See the TODO below
        raise NotImplementedError # Remove once implemented

        if generation_config is None:
            # Greedy search (default)
            generation_config = {
                'num_samples': 10,
                'prompt_length': 20,
                'seed': 11785,
                'max_length': self.model.max_len,
                'temperature': 1.0,
                'beam_width': 1,
                'repeat_penalty': 1.0,
                'top_k': 0,
                'top_p': 0.0    
            }

        # Create sequence generator
        generator = SequenceGenerator(
            score_fn=lambda x: self.model.score(x),
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device
        )

        # Sample prompts and get original sequences
        prompts, originals = dataloader.dataset.sample_prompts(
            num_samples=generation_config.get('num_samples', 10),
            prompt_length=generation_config.get('prompt_length', 10),
            seed=generation_config.get('seed', 11785)
        )
        prompts = prompts.to(self.device)

        # Generate sequences based on method
        self.model.eval()
        with torch.inference_mode():
            if generation_config.get('top_k', 0) > 0 or generation_config.get('top_p', 0) > 0:
                print("Generating with sampling...")
                seqs, scores = NotImplementedError, NotImplementedError
                raise NotImplementedError # Remove if you implemented the sampling method
            elif generation_config.get('beam_width', 1) > 1:
                print("Generating with beam search...")
                seqs, scores = NotImplementedError, NotImplementedError
                raise NotImplementedError # Remove if you implemented the beam search method
                # Take best beam and score
                seqs = seqs[:, 0]
                scores = scores[:, 0]
            else:
                # TODO: Use the prompts and the generate_greedy method you implemented in the SequenceGenerator class to generate sequences
                print("Generating with greedy search...")
                seqs, scores = NotImplementedError, NotImplementedError
                raise NotImplementedError # Remove if you implemented the greedy search method

        # Post-process sequences (trim upto EOS token)
        processed_seqs = generator.post_process_sequence(seqs, self.tokenizer)

        # Compile results
        # results is a dictionary with the following keys:
        # - prompt: the decoded prompt
        # - generated: the decoded generated sequence after the prompt
        # - original: the decoded original sequence after the prompt
        # - score: the score of the generated sequence
        # NOTE: You might find the H4Tokenizer class useful here
        results = []
        for _, (prompt, seq, score, original) in enumerate(zip(prompts, processed_seqs, scores, originals)):
            results.append({
                'prompt': self.tokenizer.decode(prompt.tolist()),
                'original': self.tokenizer.decode(original[len(prompt):].tolist()),
                'generated': self.tokenizer.decode(seq[len(prompt):].tolist()),
                'score': score.item()
            })

        return results

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of generation configurations for evaluation.
        
        Returns:
            Dictionary containing generation configurations
        """
        common_config = {
            'num_samples': 50,
            'prompt_length': 10,
            'seed': 11785,
            'max_length': self.model.max_len,
        }
        
        greedy_config = common_config.copy()
        greedy_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 0,
            'top_p': 0.0
        })
        
        beam_config = common_config.copy()
        beam_config.update({
            'temperature': 1.0,
            'beam_width': 10,
            'repeat_penalty': 1.2,
            'top_k': 0,
            'top_p': 0.0
        })

        sample_config = common_config.copy()
        sample_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 10,
            'top_p': 0.95
        })
        
        return {
            'greedy': greedy_config,
            'beam': beam_config,
            'sample': sample_config
        }