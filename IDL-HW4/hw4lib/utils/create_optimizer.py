import torch

def create_optimizer(model, opt_config):
    """
    Create optimizer with flexible parameter grouping and learning rates.
    Supports:
    - Layer-wise decay within groups
    - Different learning rates for different parameter groups
    - Custom parameter group matching by name patterns
    """
    opt_name = opt_config['name'].lower()
    base_lr = opt_config['lr']

    print(f"\n🔧 Configuring Optimizer:")
    print(f"├── Type: {opt_name.upper()}")
    print(f"├── Base LR: {base_lr}")
    print(f"├── Weight Decay: {opt_config['weight_decay']}")

    # Prepare parameter groups
    parameter_groups = []
    
    # Handle parameter groups if specified
    param_groups_config = opt_config.get('param_groups', [])
    if param_groups_config:
        print("├── Parameter Groups:")
        # Track which parameters have been assigned to groups
        assigned_params = set()
        
        for group_cfg in param_groups_config:
            group_params = []
            group_lr = group_cfg.get('lr', base_lr)
            patterns = group_cfg['patterns']  # List of patterns to match parameter names
            
            print(f"│   ├── Group: {group_cfg['name']}")
            print(f"│   │   ├── LR: {group_lr}")
            print(f"│   │   └── Patterns: {patterns}")

            # Layer-wise decay settings for this group
            use_layer_decay = group_cfg.get('layer_decay', {}).get('enabled', False)
            decay_rate = group_cfg.get('layer_decay', {}).get('decay_rate', 1.0)
            
            # Collect parameters matching the patterns
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Check if parameter matches any pattern for this group
                if any(pattern in name for pattern in patterns):
                    if name in assigned_params:
                        continue
                    
                    assigned_params.add(name)
                    
                    # Apply layer-wise decay if enabled for this group
                    if use_layer_decay:
                        depth = name.count('.')
                        actual_lr = group_lr * (decay_rate ** depth)
                    else:
                        actual_lr = group_lr
                    
                    group_params.append({
                        'params': param,
                        'lr': actual_lr,
                        'name': name
                    })
            
            if group_params:
                parameter_groups.extend(group_params)
    
        # Handle remaining parameters
        remaining_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad or name in assigned_params:
                continue
            remaining_params.append({
                'params': param,
                'lr': base_lr,
                'name': name
            })
        
        if remaining_params:
            print("│   └── Default Group (unmatched parameters)")
            parameter_groups.extend(remaining_params)
    else:
        # Model parameters with layer-wise learning rates
        if opt_config.get('layer_decay', {}).get('enabled', False):
            base_lr = opt_config['lr']
            decay_rate = opt_config['layer_decay']['decay_rate']
            print(f"├── Layer-wise Decay: Enabled")
            print(f"│   └── Decay Rate: {decay_rate}")

            # Track layers for printing
            layer_lrs = {}

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Calculate layer depth and corresponding lr
                depth = name.count('.')
                lr = base_lr * (decay_rate ** depth)

                # Store for printing
                layer_lrs[name] = lr

                parameter_groups.append({
                    'params': param,
                    'lr': lr,
                    'name': f"model.{name}"
                })

            # Print first few layer LRs as example
            print("├── Layer Learning Rates (sample):")
            for i, (name, lr) in enumerate(layer_lrs.items()):
                if i < 3:  # Show first 3 layers
                    print(f"│   ├── {name}: {lr:.6f}")
                elif i == 3:
                    print(f"│   └── ... ({len(layer_lrs)-3} more layers)")
                else:
                    break
        else:
            # Without layer-wise decay
            print(f"├── Layer-wise Decay: Disabled")
            parameter_groups.append({
                'params': model.parameters(),
                'lr': opt_config['lr'],
                'name': "model"
            })

    # Create optimizer with specific parameters
    if opt_name == 'sgd':
        print("└── SGD Specific:")
        print(f"    ├── Momentum: {opt_config['sgd']['momentum']}")
        print(f"    ├── Nesterov: {opt_config['sgd']['nesterov']}")
        print(f"    └── Dampening: {opt_config['sgd']['dampening']}")

        optimizer = torch.optim.SGD(
            parameter_groups,
            momentum=opt_config['sgd']['momentum'],
            weight_decay=opt_config['weight_decay'],
            nesterov=opt_config['sgd']['nesterov'],
            dampening=opt_config['sgd']['dampening']
        )
    elif opt_name == 'adam':
        print("└── Adam Specific:")
        print(f"    ├── Betas: {opt_config['adam']['betas']}")
        print(f"    ├── Epsilon: {opt_config['adam']['eps']}")
        print(f"    └── AMSGrad: {opt_config['adam']['amsgrad']}")

        optimizer = torch.optim.Adam(
            parameter_groups,
            betas=opt_config['adam']['betas'],
            eps=opt_config['adam']['eps'],
            weight_decay=opt_config['weight_decay'],
            amsgrad=opt_config['adam']['amsgrad']
        )
    elif opt_name == 'adamw':
        print("└── AdamW Specific:")
        print(f"    ├── Betas: {opt_config['adamw']['betas']}")
        print(f"    ├── Epsilon: {opt_config['adamw']['eps']}")
        print(f"    └── AMSGrad: {opt_config['adamw']['amsgrad']}")

        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=opt_config['adamw']['betas'],
            eps=opt_config['adamw']['eps'],
            weight_decay=opt_config['weight_decay'],
            amsgrad=opt_config['adamw']['amsgrad']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return optimizer