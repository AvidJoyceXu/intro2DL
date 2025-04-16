import torch
from typing import Dict, Any, Optional
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import copy

def _epochs_to_steps(epochs: int, train_loader: torch.utils.data.DataLoader, gradient_accumulation_steps: int = 1) -> int:
    """Convert epochs to total steps based on the train loader length."""
    return epochs * len(train_loader)

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    gradient_accumulation_steps: int = 1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on config settings.
    All schedulers except ReduceLROnPlateau are configured to be step-based.
    """
    scheduler_name = scheduler_config['name'].lower()
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps

    print("\n📈 Configuring Learning Rate Scheduler:")
    print(f"├── Type: {scheduler_name.upper()}")

    # Check for invalid warmup + ReduceLROnPlateau combination
    if scheduler_name == 'reduce_lr' and scheduler_config.get('warmup', {}).get('enabled', False):
        raise ValueError(
            "ReduceLROnPlateau scheduler cannot be combined with warmup. "
            "Please either disable warmup or use a different scheduler (cosine, cosine_warm)."
        )

    # Create base scheduler
    if scheduler_name == 'reduce_lr':
        reduce_config = scheduler_config['reduce_lr']
        patience_epochs = reduce_config.get('patience', 10)
        cooldown_epochs = reduce_config.get('cooldown', 0)
        
        print("├── ReduceLROnPlateau Settings:")
        print(f"│   ├── Mode: {reduce_config.get('mode', 'min')}")
        print(f"│   ├── Factor: {reduce_config.get('factor', 0.1)}")
        print(f"│   ├── Patience: {patience_epochs} epochs")
        print(f"│   ├── Threshold: {reduce_config.get('threshold', 0.0001)}")
        print(f"│   ├── Threshold Mode: {reduce_config.get('threshold_mode', 'rel')}")
        print(f"│   ├── Cooldown: {cooldown_epochs} epochs")
        print(f"│   └── Min LR: {reduce_config.get('min_lr', 0.00001)}")

        base_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=reduce_config.get('mode', 'min'),
            factor=reduce_config.get('factor', 0.1),
            patience=patience_epochs,  # Keep as epochs
            threshold=reduce_config.get('threshold', 0.0001),
            threshold_mode=reduce_config.get('threshold_mode', 'rel'),
            cooldown=cooldown_epochs,  # Keep as epochs
            min_lr=reduce_config.get('min_lr', 0.00001),
            eps=reduce_config.get('eps', 1e-8)
        )
        return base_scheduler

    elif scheduler_name == 'cosine':
        cosine_config = scheduler_config['cosine']
        T_max_epochs = cosine_config.get('T_max', 60)
        T_max_steps = _epochs_to_steps(T_max_epochs, train_loader, gradient_accumulation_steps)
        
        print("├── Cosine Annealing Settings:")
        print(f"│   ├── T_max: {T_max_epochs} epochs ({T_max_steps} steps)")
        print(f"│   └── Min LR: {cosine_config.get('eta_min', 0.0001)}")

        base_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max_steps,  # Convert to steps
            eta_min=cosine_config.get('eta_min', 0.0001),
            last_epoch=cosine_config.get('last_epoch', -1)
        )

    elif scheduler_name == 'cosine_warm':
        warm_config = scheduler_config['cosine_warm']
        T_0_epochs = warm_config.get('T_0', 10)
        T_0_steps = _epochs_to_steps(T_0_epochs, train_loader, gradient_accumulation_steps)
        
        print("├── Cosine Annealing Warm Restarts Settings:")
        print(f"│   ├── T_0: {T_0_epochs} epochs ({T_0_steps} steps)")
        print(f"│   ├── T_mult: {warm_config.get('T_mult', 2)}")
        print(f"│   └── Min LR: {warm_config.get('eta_min', 0.0001)}")

        base_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0_steps,  # Convert to steps
            T_mult=warm_config.get('T_mult', 2),
            eta_min=warm_config.get('eta_min', 0.0001),
            last_epoch=warm_config.get('last_epoch', -1)
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            f"Supported: ['reduce_lr', 'cosine', 'cosine_warm']"
        )

    # Add warmup if enabled
    if scheduler_config.get('warmup', {}).get('enabled', False):
        warmup_config = scheduler_config['warmup']
        warmup_epochs = warmup_config.get('epochs', 5)
        warmup_steps = warmup_epochs * steps_per_epoch
        print("├── Warmup Settings:")
        print(f"│   ├── Duration: {warmup_epochs} epochs ({warmup_steps} steps)")
        print(f"│   ├── Start Factor: {warmup_config.get('start_factor', 0.1)}")
        print(f"│   └── End Factor: {warmup_config.get('end_factor', 1.0)}")

        scheduler = create_warmup_scheduler(
            optimizer,
            base_scheduler,
            warmup_config,
            train_loader
        )
    else:
        print("└── Warmup: Disabled")
        scheduler = base_scheduler

    return scheduler


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    base_scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Create a warmup scheduler wrapped around the base scheduler.
    """
    warmup_epochs = warmup_config.get('epochs', 5)
    start_factor = warmup_config.get('start_factor', 0.1)
    end_factor = warmup_config.get('end_factor', 1.0)

    # Calculate the number of warmup steps
    warmup_steps = len(train_loader) * warmup_epochs

    # Create warmup scheduler
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_steps
    )

    # Combine warmup with main scheduler
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps]
    )

    return scheduler


def plot_lr_schedule(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    gradient_accumulation_steps: int = 1,
    max_groups: int = 5  # Maximum number of groups to plot
) -> None:
    """
    Plot the learning rate schedule over epochs.
    
    Args:
        scheduler: The learning rate scheduler
        num_epochs: Total number of epochs to plot
        train_loader: Training data loader to determine steps per epoch
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_groups: Maximum number of parameter groups to plot
    """
    # Save initial states
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler_state = copy.deepcopy(scheduler.__dict__)
    else:
        scheduler_state = copy.deepcopy(scheduler.state_dict())
    
    optimizer_state = copy.deepcopy(scheduler.optimizer.state_dict())
    
    # Store initial learning rates
    initial_lr = [group['lr'] for group in scheduler.optimizer.param_groups]
    num_groups = len(initial_lr)
    
    # If there are too many groups, only plot a subset
    groups_to_plot = min(num_groups, max_groups)
    if num_groups > max_groups:
        print(f"Warning: Only showing {max_groups} out of {num_groups} parameter groups for clarity")
    
    lrs = [[] for _ in range(groups_to_plot)]
    
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # For ReduceLROnPlateau, simulate epoch-wise updates
        for epoch in range(num_epochs):
            # Record current learning rates
            for idx, group in enumerate(scheduler.optimizer.param_groups[:groups_to_plot]):
                lrs[idx].extend([group['lr']] * len(train_loader))
            
            # Step the scheduler with a dummy metric that triggers LR reduction
            # every patience+1 epochs to show the behavior
            scheduler.step(1.0 if epoch % (scheduler.patience + 1) != 0 else 0.0)
        
        x = np.linspace(0, num_epochs, num_epochs * len(train_loader))
    else:
        # For step-based schedulers
        total_steps = _epochs_to_steps(num_epochs, train_loader, gradient_accumulation_steps)
        
        # Simulate training loop
        for step in range(total_steps):
            # Record current learning rates
            for idx, group in enumerate(scheduler.optimizer.param_groups[:groups_to_plot]):
                lrs[idx].append(group['lr'])
            
            # Step the scheduler
            scheduler.optimizer.step()
            scheduler.step()
        
        x = np.linspace(0, num_epochs, total_steps)
    
    # Restore initial states
    scheduler.optimizer.load_state_dict(optimizer_state)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.__dict__.update(scheduler_state)
    else:
        scheduler.load_state_dict(scheduler_state)
    
    # Plot the learning rates with better styling
    plt.figure(figsize=(12, 4))
    
    # Define a set of distinct colors and line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    
    for idx, lr_list in enumerate(lrs[:groups_to_plot]):
        color = colors[idx % len(colors)]
        style = line_styles[idx % len(line_styles)]
        plt.plot(x, lr_list, 
                label=f'Group {idx}', 
                color=color, 
                linestyle=style,
                linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    
    # Create a second x-axis for steps
    ax2 = plt.gca().twiny()
    ax2.set_xlim(0, len(x))
    ax2.set_xlabel('Steps', fontsize=12)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()