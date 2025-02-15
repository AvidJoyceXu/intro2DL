

# [11785 hw1p2] lingyun3@andrew.cmu.edu's implementation 

## Collaborator
- xinyuea@andrew.cmu.edu

> We work independently during code development and discuss intensively during hyperparameter tuning, and finally I adopt xinyuea@'s architecture for better performance.

## Wandb project
- https://wandb.ai/avidjoyce23-peking-university/hw1p2/overview

## Code development timeline
- git commit history: https://github.com/AvidJoyceXu/intro2DL/commits/main/

> The repo has been private until 2025/02/14. Hopefully, it won't cause plagarism.

## Experiment runs
### Local runs
> I comment out the wandb logging because it takes some time for `wandb.init()` to launch, which occupies relatively long time for quickly testing hyperparameter for few runs on a small dataset.

| subset | train/val acc | context | archetype | activations | learning_rate | dropout | ~alternate | num_layers | optimizer | scheduler        | batch_size | weight_decay | weight_initialization |
| ------ | ------------- | ------- | --------- | ----------- | ------------- | ------- | ---------- | ---------- | --------- | ---------------- | ---------- | ------------ | --------------------- |
| 0.1    | 53.73/61.72   | 30      | diamond   | GELU        | 0.001         | 0.25    | False      | 8          | SGD       | ReduceLR         | 2048       |              | None                  |
|        | 80/68         | 30      | cylinder  | RELU        |               | 0       |            | 4          | Adam      | ReduceLRonPlateu | 2048       | 0            |                       |
|        | 58/57         |         |           |             |               |         |            |            |           |                  |            | 0.005        |                       |
|        | 75/68         |         |           |             |               |         |            |            |           |                  |            | 0.001        |                       |
|        | 81/68         |         |           | gelu        |               |         |            |            |           |                  |            |              |                       |
| 1      |               |         |           | relu        | 0.005         | 0.1     | True       |            |           |                  |            | 0.002        |                       |
|        |               |         |           |             |               |         |            |            |           |                  |            |              | kaiming_normal        |
|        | 90/84         |         | diamond   |             | 0.001         | 0.05    |            |            |           |                  |            | 0.001        |                       |
|        |               |         |           |             |               | 0.2     |            |            |           |                  |            | 0.002        |                       |
|        |               |         | diamond   | relu        | 0.002         | 0.15    | True       |            |           |                  | 2048       |              | kaiming_normal        |

### Online runs
- https://wandb.ai/avidjoyce23-peking-university/hw1p2/overview

## Architecture design
- First week: I implement the entire train/eval loop and extensively test the diamond architecture

- Second week: After meeting with TA, I switch to 'pencil' architecture and use hyperparameters recommended by the TA.

- Final model size: 
> Params:    19,985.64K

> Mult-Adds: 19.96M

- Exact architecture: refer to `class Network`'s `__init__` function and the `config` dictionary

```Python
config = {
    'Name': 'Lingyun', # Write your name here
    'subset': 1, # Subset of dataset to use (1.0 == 100% of data)
    'context': 50,
    'activations': 'leakyrelu', # gelu, relu, leakyrelu, softplus, tanh, sigmoid
    'batch_norm': True, 
    'batchnorm_alternate': False, # Whether apply batchnorm to alternate layers
    'learning_rate': 0.001,
    'dropout': 0.225, # 0-0.5
    'dropout_alternate': True, # Whether apply dropout to alternate layers
    'num_layers': 7, # 2-8
    'optimizers': 'adamw',
    'scheduler': 'cosineanneal', # steplr, reducelronplateau, exponential, cosineanneal
    'epochs': 150,
    'batch_size': 2048, # 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    'weight_decay': 0.001, # don't tune this
    'weight_initialization': 'kaiming_uniform', # kaiming_normal, kaiming_uniform, uniform, xavier_normal or xavier_uniform
    'augmentations': 'FreqMask', # Options: ["FreqMask", "TimeMask", "Both", null]
    'freq_mask_param': 4,
    'time_mask_param': 8,
}
```

```Python
self.model = nn.Sequential(
            nn.Linear(input_size, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.24),
            
            nn.Linear(constant, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.225),
            
            nn.Linear(constant, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.215),
            
            nn.Linear(constant, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.205),
            
            nn.Linear(constant, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            
            nn.Linear(constant, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            
            nn.Linear(constant, constant),
            nn.BatchNorm1d(constant),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            
            nn.Linear(constant, output_size),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # Initialize weights if specified
        if config['weight_initialization'] is not None:
            self.initialize_weights()
```

## Reproduction
- run the `hw1p2-s25-starter-notebook.ipynb`