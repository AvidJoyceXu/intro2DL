import wandb
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from hw4lib.data.tokenizer import H4Tokenizer
from hw4lib.utils import create_optimizer
from hw4lib.model import DecoderOnlyTransformer, EncoderDecoderTransformer
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from torchinfo import summary


class BaseTrainer(ABC):
    """
    Base Trainer class that provides common functionality for all trainers.

    This trainer implements:
    1. Experiment tracking and logging (with wandb support)
    2. Checkpoint management
    3. Metric logging and visualization
    4. Directory structure management
    5. Device handling

    Key Components:
    1. Experiment Management:
    - Creates organized directory structure for experiments
    - Handles config file copying and model architecture saving
    - Manages checkpoint saving and loading
    
    2. Logging and Visualization:
    - Supports both local and wandb logging
    - Saves attention visualizations
    - Tracks training metrics and learning rates
    - Saves generated text outputs
    
    3. Training Infrastructure:
    - Handles device placement
    - Manages optimizer creation
    - Supports gradient scaling for mixed precision
    - Implements learning rate scheduling

    4. Abstract Methods (to be implemented by child classes):
    - _train_epoch: Single training epoch implementation
    - _validate_epoch: Single validation epoch implementation
    - train: Full training loop implementation
    - evaluate: Evaluation loop implementation

    Args:
        model (nn.Module): The model to train
        tokenizer (H4Tokenizer): Tokenizer for text processing
        config (dict): Configuration dictionary
        run_name (str): Name for the training run
        config_file (str): Path to config file
        device (Optional[str]): Device to run on ('cuda' or 'cpu')

    Directory Structure:
        expts/
        └── {run_name}/
            ├── config.yaml
            ├── model_arch.txt
            ├── checkpoints/
            │   ├── checkpoint-best-metric-model.pth
            │   └── checkpoint-last-epoch-model.pth
            ├── attn/
            │   └── {attention visualizations}
            └── text/
                └── {generated text outputs}
    """
    def __init__(
            self,
            model: nn.Module,
            tokenizer: H4Tokenizer,
            config: dict,
            run_name: str,
            config_file: str,
            device: Optional[str] = None
    ):
        # If device is not specified, determine it
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize optimizer and scheduler
        self.optimizer = None  # Should be set by child class
        self.scheduler = None  # Will be set when training starts
        self.scaler = torch.amp.GradScaler(device=self.device)
        self.use_wandb = config['training'].get('use_wandb', False)
        # Initialize experiment directories
        self.expt_root, self.checkpoint_dir, self.attn_dir, self.text_dir, \
        self.best_model_path, self.last_model_path = self._init_experiment(run_name, config_file)

        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_history = []
    
    @abstractmethod
    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Train for one epoch."""
        pass

    @abstractmethod
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate for one epoch."""
        pass

    @abstractmethod
    def train(self, train_dataloader, val_dataloader):
        """Full training loop."""
        pass

    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluation loop."""
        pass


    def _init_experiment(self, run_name: str, config_file: str):
        """Initialize experiment directories and save initial files."""
        # Create experiment directory
        expt_root = Path(os.getcwd()) / 'expts' / run_name
        expt_root.mkdir(parents=True, exist_ok=True)

        # Copy config
        shutil.copy2(config_file, expt_root / "config.yaml")

        # Save model architecture with torchinfo summary
        with open(expt_root / "model_arch.txt", "w") as f:
            # Get a sample input shape from your model's expected input
            if isinstance(self.model, DecoderOnlyTransformer):
                batch_size = self.config['data'].get('batch_size', 8)
                max_len    = self.model.max_len
                input_size = [(batch_size, max_len), (batch_size,)]
                dtypes     = [torch.long, torch.long]
                # Generate the summary
                model_summary = summary(
                    self.model,
                    input_size=input_size,  # Adjust these dimensions based on your model's input
                    dtypes=dtypes
                )
                # Write the summary string to file
                f.write(str(model_summary))
            elif isinstance(self.model, EncoderDecoderTransformer):
                batch_size = self.config['data'].get('batch_size', 8)
                max_len = 1000
                num_feats = self.config['data']['num_feats']
                input_data = [
                    torch.randn(batch_size, max_len, num_feats).to(self.device), 
                    torch.randint(0, self.model.num_classes, (batch_size, max_len//10)).to(self.device), 
                    torch.randint(max_len//2, max_len, (batch_size,)).to(self.device), 
                    torch.randint(max_len//20, max_len//10, (batch_size,)).to(self.device)
                ]
                dtypes = [torch.float32, torch.long, torch.long, torch.long]
                # Generate the summary
                model_summary = summary(
                    self.model,
                    input_data=input_data,  # Adjust these dimensions based on your model's input
                    dtypes=dtypes
                )
                # Write the summary string to file
                f.write(str(model_summary))
            else:
                raise NotImplementedError("Model architecture summary not implemented")

        # Create subdirectories
        checkpoint_dir = expt_root / 'checkpoints'
        attn_dir = expt_root / 'attn'
        text_dir = expt_root / 'text'
        
        checkpoint_dir.mkdir(exist_ok=True)
        attn_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)

        # Define checkpoint paths
        best_model_path = checkpoint_dir / 'checkpoint-best-metric-model.pth'
        last_model_path = checkpoint_dir / 'checkpoint-last-epoch-model.pth'

        # Wandb initialization
        if self.use_wandb:
            """Initialize Weights & Biases logging."""
            run_id = self.config['training'].get('wandb_run_id', None)
            if run_id and run_id.lower() != "none":
                self.wandb_run = wandb.init(
                    project=self.config['training'].get('wandb_project', 'default-project'),
                    id=run_id,
                    resume="must",
                    config=self.config
                )
            else:
                self.wandb_run = wandb.init(
                    project=self.config['training'].get('wandb_project', 'default-project'),
                    config=self.config,
                    name=run_name
                )

        return expt_root, checkpoint_dir, attn_dir, text_dir, best_model_path, last_model_path

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]], step: int):
        """Generic metric logging method."""
        self.training_history.append({
            'epoch': step,
            **metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {}
            for split, split_metrics in metrics.items():
                for metric_name, value in split_metrics.items():
                    wandb_metrics[f'{split}/{metric_name}'] = value
            wandb_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(wandb_metrics, step=step)
        
        # Print metrics with tree structure
        print(f"\n📊 Metrics (Epoch {step}):")
        
        # Print metrics by split
        splits = sorted(metrics.keys())
        for i, split in enumerate(splits):
            is_last_split = i == len(splits) - 1
            split_prefix = "└──" if is_last_split else "├──"
            print(f"{split_prefix} {split.upper()}:")
            
            # Print metrics within split
            split_metrics = sorted(metrics[split].items())
            for j, (metric_name, value) in enumerate(split_metrics):
                is_last_metric = j == len(split_metrics) - 1
                metric_prefix = "    └──" if is_last_metric else "    ├──"
                if is_last_split:
                    metric_prefix = "    └──" if is_last_metric else "    ├──"
                else:
                    metric_prefix = "│   └──" if is_last_metric else "│   ├──"
                print(f"{metric_prefix} {metric_name}: {value:.4f}")
        
        # Print learning rate
        print("└── TRAINING:")
        print(f"    └── learning_rate: {self.optimizer.param_groups[0]['lr']:.6f}")


    def _save_attention_plot(self, attn_weights: torch.Tensor, epoch: int, attn_type: str = "self"):
        """Save attention weights visualization."""
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, cmap="viridis", cbar=True)
        plt.title(f"Attention Weights - Epoch {epoch}")
        plt.xlabel("Source Sequence")
        plt.ylabel("Target Sequence")
        
        plot_path = os.path.join(self.attn_dir, f"{attn_type}_attention_epoch{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        
        if self.use_wandb:
            wandb.log({f"{attn_type}_attention": wandb.Image(plot_path)}, step=epoch)


    def _save_generated_text(self, text: dict, suffix: str):
        """Save generated text to JSON file."""
        text_path = os.path.join(self.text_dir, f"text_{suffix}.json")
        with open(text_path, "w") as f:
            json.dump(text, f, indent=4)
    
        if self.use_wandb:
            wandb.save(text_path)


    def save_checkpoint(self, filename: str):
        """Save a checkpoint of the model and training state."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        if self.use_wandb:
            wandb.save(str(checkpoint_path))


    def load_checkpoint(self, filename: str):
        """
        Load a checkpoint.
        
        Attempts to load each component of the checkpoint separately,
        continuing even if some components fail to load.
        """
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file: {e}")

        # Dictionary to track loading status of each component
        load_status = {}

        # Try loading model state
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            load_status['model'] = True
        except Exception as e:
            print(f"Warning: Failed to load model state: {e}")
            load_status['model'] = False

        # Try loading optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            load_status['optimizer'] = True
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}")
            load_status['optimizer'] = False

        # Try loading scheduler state if it exists
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                load_status['scheduler'] = True
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")
                load_status['scheduler'] = False

        # Try loading scaler state
        try:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            load_status['scaler'] = True
        except Exception as e:
            print(f"Warning: Failed to load scaler state: {e}")
            load_status['scaler'] = False

        # Try loading training state
        try:
            self.current_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            self.training_history = checkpoint['training_history']
            load_status['training_state'] = True
        except Exception as e:
            print(f"Warning: Failed to load training state: {e}")
            load_status['training_state'] = False

        # Summarize what was loaded successfully
        successful_loads = [k for k, v in load_status.items() if v]
        failed_loads = [k for k, v in load_status.items() if not v]
        
        if not successful_loads:
            raise RuntimeError("Failed to load any checkpoint components")
        
        print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Successfully loaded: {', '.join(successful_loads)}")
        if failed_loads:
            print(f"Failed to load: {', '.join(failed_loads)}")


    def cleanup(self):
        """Cleanup resources."""
        if self.use_wandb and self.wandb_run:
            wandb.finish()