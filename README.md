# lazy-vicsek

```
- **Let's test out the baseline model, Vicsek in Lazy framework**
- 
```

## TODOs
- [x] ~~Open repository~~
- [x] ~~Make directories~~
  - [x] ~~./env~~
  - [x] ~~./model
  - [x] ~~./utils~~
  - [x] ~~./data~~
  - [x] ~~./experiments~~
- [x] ~~Create an environment file~~
  - [x] ~~LazyVicsekEnv~~
  - [x] ~~LazyVicsekEnvConfig~~; ~~config using pydantic~~
- [x] ~~Build model~~
  - [x] ~~lazy listener~~
  - [x] ~~lazy controller~~
- [x] ~~Env & Model Updated: local comm integrated~~
- [x] ~~Hyperparam tuning~~
- [x] ~~Train and save the model~~
- [x] Evaluate the model
  - [x] ~~Average performance in the base environment (validation)~~
  - [x] ~~Order parameter changes over time~~
  - [ ] Noise test (test)
  - [ ] Scalability
- [x] ~~Collect data~~
- [x] ~~Save data~~
- [x] ~~Data representation~~


## Assumptions
- Centralized training (decentralized executable tho)
- No noise in training phase
  - Training with noise is out of the scope of this project 
- Communications: ~~Fully connected~~ -> Must be locally connected; distance-based communication
  - ~~Plan to test out partially connected communication setting for scalability and real-world applicability~~
  - With fully connected network, Vicsek beats every other neighbor selections cuz it converges with a single time step update
- Model
  - Observation divided into n sub-observations (n = number of agents)
  - Forward-passes are done in parallel
  - Sub-logits are concatenated to form the final logits (believe RLlib updates the model accordingly; PPO updates are pretty simple!)
  - Action distribution: Multi-Binary (implemented by Multi-MultiBinomial with two actions to facilitate the built-in RLlib ActionDistribution)
-

## Dependencies (see requirements.txt)
- gym==0.23.1        # not gymnasium
- ray==2.1.0         # uses gym
- pydantic==1.10.13  # not V2.x.x
- Python 3.9.x
