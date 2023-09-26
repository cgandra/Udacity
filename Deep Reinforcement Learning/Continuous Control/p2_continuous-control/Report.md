[//]: # (Image References)

[image1]: images/ddpg_algo.png
[image2]: images/ddpg_actor.png
[image3]: images/ddpg_critic.png
[image4]: images/CC_1_Train_Test.png
[image5]: images/CC_20_Train_Test.png

# Project 2: Continuous Control

### Goal
For this project, we train an agent to control various simulated double-jointed arm and track them. The goal of each agent is to maintain its position at the target location for as many time steps as possible.

### Learning Algorithm
The agent implements [DDPG](https://arxiv.org/pdf/1509.02971.pdf) algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

DDPG combines [DPG](http://proceedings.mlr.press/v32/silver14.pdf) with [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).  Recall that DQN (Deep Q-Network) stabilizes the learning of Q-function by experience replay and the frozen target network. The original DQN works in discrete space, and DDPG extends it to continuous space with the actor-critic framework to learn the action-value function Q(s,a;φ) while also learning a deterministic policy μθ(s) via DPG. Policy learning in DDPG is fairly simple. We want to learn a deterministic policy μθ(s) which gives the action that maximizes Q(s,a)

In order to do better exploration, an exploration policy μ’ is constructed by adding noise N
```
    μ′(s)=μθ(s)+N
```
The authors of the original DDPG paper recommended time-correlated [OU noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well. Current implementation handles only OU noise

In addition, DDPG does soft updates (“conservative policy iteration”) on the parameters of both actor and critic, with τ≪1: θ′←τθ+(1−τ)θ′. In this way, the target network values are constrained to change slowly, different from the design in DQN that the target network stays frozen for some period of time.

![alt text][image1]

### Network Architecture for Actor Network

1. DDPG Actor network:
```
Actor(
  (conv): Sequential(
    (0): Linear(in_features=33, out_features=256, bias=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
  )
  (fcout): Linear(in_features=128, out_features=4, bias=True)
)
```
![alt text][image2]

2. DDPG Critic network:
```
Critic(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv): Sequential(
    (0): Linear(in_features=260, out_features=128, bias=True)
    (1): ReLU()
  )
  (fcout): Linear(in_features=128, out_features=1, bias=True)
)
```
![alt text][image3]

```
  # Actor Network (w/ Target Network)
  self.policy = Model(actor_model_cfg, trainable=training)
  self.policy_target = Model(actor_model_cfg, target=True, trainable=False)

  # Critic Network (w/ Target Network)
  self.critic = Model(critic_model_cfg, critic=True, trainable=training)
  self.critic_target = Model(critic_model_cfg, critic=True, target=True, trainable=False)
```

### Hyperparameters
1. Single Agent:

    <class 'config.AgentConfig'>: {'state_size': 33, 'action_size': 4, 'num_agents': 1, 'n_episodes': 2000, 'max_t_steps': 2000, 'replay_buf_size': 1000000, 'min_replay_buf_size': 256, 'minibatch_size': 128, 'training_freq': 1, 'update_tgt_freq': 2, 'tau': 0.001, 'gamma': 0.99}

    Actor <class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0, 'input_dim': 33, 'output_dim': 4, 'append_dim': 0, 'nflts': [256, 128]}
    Critic <class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0, 'input_dim': 33, 'output_dim': 1, 'append_dim': 4, 'nflts': [256, 128]}

2. 20 Agents:

    <class 'config.AgentConfig'>: {'state_size': 33, 'action_size': 4, 'num_agents': 20, 'n_episodes': 2000, 'max_t_steps': 2000, 'replay_buf_size': 1000000, 'min_replay_buf_size': 256, 'minibatch_size': 128, 'training_freq': 1, 'update_tgt_freq': 2, 'tau': 0.001, 'gamma': 0.99}

    Actor <class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0, 'input_dim': 33, 'output_dim': 4, 'append_dim': 0, 'nflts': [256, 128]}
    Critic <class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0, 'input_dim': 33, 'output_dim': 1, 'append_dim': 4, 'nflts': [256, 128]}

### Results for Single Agent
Plot showing the score per episode over all the episodes during training and testing. The title of subplot shows the number of episodes needed to solve the environment
![alt text][image4]

Test results can be reproduced by running below
```
from utils import *
from workspace_utils import active_session
%matplotlib inline

with active_session():
    run_option('test', 0, agent_type='single', n_episodes=100, seed=0, chkpt_type='best_ma')
```

### Results for 20 Agents
Plot showing the score per episode over all the episodes during training and testing. The title of subplot shows the number of episodes needed to solve the environment
![alt text][image5]

Test results can be reproduced by running below
```
from utils import *
from workspace_utils import active_session
%matplotlib inline

with active_session():
    run_option('test', 0, agent_type='multi', n_episodes=100, seed=0, chkpt_type='best_ma')
```

### Future Improvements

1. Try uncorrelated, mean-zero Gaussian noise. And reduce the scale of the noise over the course of training to facilitate getting higher-quality training data
2. Add prioritized experience replay — Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated to the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.
3. Experiment with other algorithms such as [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477), [Proximal Policy Optimization (PPO)](Proximal Policy Optimization Algorithms), [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617) or [A3C](https://arxiv.org/pdf/1602.01783.pdf) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience for the 20 agents case
4. Study different network architectures (depths) for actor/critic and impact on training/inference
