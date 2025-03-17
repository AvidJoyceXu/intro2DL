

# [11785 hw2p2] lingyun3@andrew.cmu.edu's implementation 

## Collaborator
- xinyuea@andrew.cmu.edu

> We work independently during code development and discuss intensively during hyperparameter tuning, and finally I adopt xinyuea@'s architecture for better performance.

## Wandb project
- https://wandb.ai/avidjoyce23-peking-university/hw2p2-ablations?nw=nwuseravidjoyce23

## Experiment runs
### ConvNext backbone

I try to use the most advanced convolution network architecture, namely ConvNext. However, I find it hard to narrow the train-val gap for convnext, and my groupmates all play with Resnet backbone, so I switch to Resnet afterwards.

In addition to the wandb logs, I had several rounds of experiments as follows: 

| batchsize | lr    | epoch | augment | trainacc | valacc |
| --------- | ----- | ----- | ------- | -------- | ------ |
| 64        | 0.001 | 20    | None    | 99.99    | 64.79  |
| 8192      |       |       |         | 44.6     | 32.85  |
|           | 0.01  | 200   |         | 100      | 54.96  |
| 8192      |       |       | True    | 93.37    | 64.73  |

### Resnet backbone

- https://wandb.ai/avidjoyce23-peking-university/hw2p2-ablations?nw=nwuseravidjoyce23

## Reproduction

- run the `train.py` for pretraining best-performance checkpoints 

- run `HW2P2_Student_Starter_Spring25.ipynb` for evaluation (make sure `FLAG="eval"`)
