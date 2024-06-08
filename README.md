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
- [x] Create an environment file
  - Less abstractions but quicker! just like before
  - But how are you going to deal with Laziest models (i.e. both lazy listenr and controller) ???
    - idk...
- [ ] Build model
  - [ ] lazy listener
  - [ ] lazy controller
- [ ] Hyperparam tuning
- [ ] Train and save the model
- [ ] Evaluate the model
  - [ ] Average performance in the base environment (validation)
  - [ ] Noise test (test)
  - [ ] Scalability
- [ ] Collect data
- [ ] Save data
- [ ] Data representation


## Assumptions
- Centralized training (decentralized executable tho)
- No noise in training phase
  - Training with noise is out of the scope of this project 
- Communications: Fully connected
  - Plan to test out partially connected communication setting for scalability and real-world applicability


## Communication Checks
- Don't yak, my yam!