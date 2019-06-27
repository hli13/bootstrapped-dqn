# Deep Reinforcement Learning via Bootstrapped DQN for Playing Atari Games

This project was a team effort of implementing and investigating the bootstrapped deep Q-network (DQN) for playing selected classic Atari games. It was developed based on [this](https://arxiv.org/abs/1602.04621) paper. The implementation was inspired by [this](https://github.com/pianomania/DQN-pytorch) repo.

[OpenAI Gym](https://gym.openai.com/envs/#atari) was integrated in the pipeline to obtain the dataset for Atari games. The deep learning agent was trained on [BlueWaters](http://www.ncsa.illinois.edu/enabling/bluewaters) GPU nodes for approximately 600 hours.

The bootstrapped DQN architecture aims to enable deep and more efficient exploration as compared to classical algorithms. We showed that the boostratpped DQN is able to reach human performance 26% faster than the classical DQN with epsilon-greedy exploration.

## Dependencies

```
numpy==1.16.4
matplotlib==3.1.0
torch==0.4.1
torchvision==0.2.1
gym==0.10.9
```

## Dataset

The Atari game dataset is provided by [OpenAI Gym](https://gym.openai.com/envs/#atari). Among all the games available in the environment, we selected [Pong](https://gym.openai.com/envs/Pong-v0/) and [Breakout](https://gym.openai.com/envs/Breakout-v0/) to train and test the performance of the agent.

## Implementation

TBD

## Hyerparameters

TBD

## Running the model

TBD

## Result

TBD

## Contributors
 - Haoyang Li
 - Qianli Chen
 - Muyun Lihan
