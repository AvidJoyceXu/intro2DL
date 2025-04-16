
# [11785 hw3p2] lingyun3@andrew.cmu.edu's implementation 

## Collaborator
- None

## Wandb project
- https://wandb.ai/avidjoyce23-peking-university/hw3p2-ablations/overview


## Reproduction

- run the `train.py` and set `mode = 'TRAIN'` for pretraining best-performance checkpoints 

- run `train.py` and set `mode = 'EVAL'` for evaluation

## Experiments

- BLSTM=2 performs far better than 3 or more (excessive input compression hinders performance)

- encoder/feature extraction layer: 
    
    - bottleneck (hiddendim <> inputdim>) doesn't work well

    - adopt pencil architecture
