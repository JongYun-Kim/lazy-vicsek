# lazy-vicsek


## Environment settings
| Parameter                    | Value   |
|------------------------------|---------|
| Time step length (Δt)         | 0.1 s   |
| Episode length                | 30 s    |
| Communication radius (r)      | 10 m    |
| Speed (v)                     | 5 m/s   |
| Number of agents (N)          | 20      |
| Square side length            | 100 m   |

## RL Hyperparameters
| Hyperparameter             | Value   |
|----------------------------|---------|
| Number of parallel environments | 48      |
| Train batch size            | 14,440  |
| Minibatch size              | 256     |
| Number of epochs            | 8       |
| Discount factor (γ)         | 0.992   |
| GAE Lambda (λ)              | 0.96    |
| KL coefficient              | 0       |
| Clip parameter              | 0.22    |
| Gradient clipping           | 0.5     |
| KL target                   | 0.01    |
| Entropy coefficient         | 0       |
| Learning rate               | 1e-4    |
| Optimizer                   | Adam    |

## Network settings
| Parameter                                 | Value   |
|-------------------------------------------|---------|
| share_layers                              | False   |
| raw embedding dim                         | 6       |
| d_embed_input                             | 256     |
| d_embed_context                           | 256     |
| d_model                                   | 256     |
| d_model_decoder                           | 256     |
| n_layers_encoder                          | 3       |
| n_layers_decoder                          | 1       |
| num_heads                                 | 8       |
| d_ff                                      | 512     |
| d_ff_decoder                              | 512     |
| norm_eps                                  | 1e-5    |
| bias in attention transformation matrices | False   |


## Assumptions
- Centralized _training_ (decentralized-executable tho)
- No noise in training envs
  - Training with noise is out of the scope of this project 
- Communication
  - Locally connected
  - Distance-based communication
- Model
  - Observation divided into n sub-observations (n = number of agents)
  - Forward-passes are done in parallel
  - Sub-logits are concatenated to form the final logits (believe RLlib updates the model accordingly; PPO updates are pretty simple!)
  - Action distribution: Multi-Binary (implemented by Multi-MultiBinomial with two actions to facilitate the built-in RLlib ActionDistribution)

## Dependencies (see requirements.txt)
- gym==0.23.1        # not gymnasium
- ray==2.1.0         # uses gym
- pydantic==1.10.13  # not V2.x.x
- Python 3.9.x
