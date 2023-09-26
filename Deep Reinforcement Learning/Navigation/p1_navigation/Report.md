[//]: # (Image References)

[image2]: images/dqn.png
[image3]: images/duel_dqn.png
[image4]: images/dqn_3D.png
[image5]: images/duel_dqn_3D.png
[image6]: images/dqn_2D.png
[image7]: images/duel_dqn_3D.png
[image8]: images/Nav_Train.png
[image9]: images/Nav_Test.png
[image10]: images/Nav_Train_3D.png
[image11]: images/Nav_Test_3D.png
[image12]: images/Nav_Train_2D.png
[image13]: images/Nav_Test_2D.png

# Project 1: Navigation

### Goal
For this project, we train an agent to navigate (and collect bananas!) in a large, square world. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. In order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Learning Algorithm
The agent implements Deep Q-learning ([DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) and its variants ([double DQN](https://arxiv.org/abs/1509.06461), [dueling DQN](https://arxiv.org/abs/1511.06581) and [double dueling DQN](https://arxiv.org/abs/1511.06581)) to learn a suitable policy in a model-free reinforcement learning setting. [Experience replay](https://dl.acm.org/doi/book/10.5555/168871) is implemented as default with all algo variants

### Network Architecture for Basic Q Network

1. DQN network:
```
FcModel(
  (conv): Sequential(
    (0): Linear(in_features=37, out_features=64, bias=True)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=64, out_features=64, bias=True)
    (4): ReLU()
  )
  (fc3): Linear(in_features=64, out_features=4, bias=True)
)
```
![alt text][image2]

2. Duel DQN network:
```
FcModel(
  (conv): Sequential(
    (0): Linear(in_features=37, out_features=64, bias=False)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=64, out_features=64, bias=True)
    (4): ReLU()
  )
  (fc3): Linear(in_features=64, out_features=4, bias=True)
  (fc3_value): Linear(in_features=64, out_features=1, bias=True)
)
```
![alt text][image3]

### Network Architecture for Visual Banana
1. DQN network using Conv3D: States have shape: (1, 3, 4, 84, 84)

```
ConvModel(
  (conv): Sequential(
    (0): Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    (4): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    (7): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): Conv3d(256, 512, kernel_size=(4, 3, 3), stride=(1, 2, 2))
    (10): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=8192, out_features=256, bias=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=4, bias=True)
  )
)
```
![alt text][image4]

2. Duel DQN network using Conv3D: States have shape: (1, 3, 4, 84, 84)

```
ConvModel3D(
  (conv): Sequential(
    (0): Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    (4): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2))
    (7): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): Conv3d(256, 512, kernel_size=(4, 3, 3), stride=(1, 2, 2))
    (10): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=8192, out_features=256, bias=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=4, bias=True)
  )
  (fc_value): Sequential(
    (0): Linear(in_features=8192, out_features=256, bias=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=1, bias=True)
  )
)
```
![alt text][image5]

3. DQN network using Conv2D: States have shape: (1, 12, 84, 84)

```
ConvModel2D(
  (conv): Sequential(
    (0): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Conv2d(12, 256, kernel_size=(3, 3), stride=(2, 2))
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
    (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (8): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU()
    (10): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (11): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=8192, out_features=64, bias=True)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=64, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=4, bias=True)
  )
)
ConvModel2D(
  (conv): Sequential(
    (0): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Conv2d(12, 256, kernel_size=(3, 3), stride=(2, 2))
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
    (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (8): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU()
    (10): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (11): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=8192, out_features=64, bias=True)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=64, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=4, bias=True)
  )
)
```
![alt text][image6]

4. Duel DQN network using Conv2D: States have shape: (1, 12, 84, 84)
```
ConvModel2D(
  (conv): Sequential(
    (0): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Conv2d(12, 256, kernel_size=(3, 3), stride=(2, 2))
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
    (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (8): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU()
    (10): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (11): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=8192, out_features=64, bias=True)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=64, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=4, bias=True)
  )
  (fc_value): Sequential(
    (0): Linear(in_features=8192, out_features=64, bias=True)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=64, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=1, bias=True)
  )
)
```
![alt text][image7]

### Hyperparameters
1. Banana:

    <class 'config.AgentConfig'>: {'n_episodes': 2000, 'max_t_steps': 1000, 'replay_buf_size': 100000, 'min_replay_buf_size': 256, 'minibatch_size': 64, 'training_freq': 4, 'update_tgt_freq': 4, 'tau': 0.001, 'gamma': 0.99, 'eps_schedule': 'geometric', 'eps_start': 1.0, 'eps_end': 0.01, 'eps_decay': 0.995}

    <class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0}

2. Visual Banana w 3D conv network:

  <class 'config.AgentConfig'>: {'n_episodes': 2000, 'max_t_steps': 1000, 'replay_buf_size': 100000, 'min_replay_buf_size': 68, 'minibatch_size': 64, 'training_freq': 4, 'update_tgt_freq': 4, 'tau': 0.001, 'gamma': 0.99, 'eps_schedule': 'geometric', 'eps_start': 1.0, 'eps_end': 0.01, 'eps_decay': 0.99}

  <class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0}

3. Visual Banana w 2D conv network:

<class 'config.AgentConfig'>: {'n_episodes': 2000, 'max_t_steps': 1000, 'replay_buf_size': 100000, 'min_replay_buf_size': 260, 'minibatch_size': 64, 'training_freq': 4, 'update_tgt_freq': 4, 'tau': 0.001, 'gamma': 0.99, 'eps_schedule': 'geometric', 'eps_start': 1.0, 'eps_end': 0.01, 'eps_decay': 0.995}

<class 'config.ModelConfig'>: {'learning_rate': 0.0001, 'weight_decay': 0}

### Results for Banana
Plot showing the score per episode over all the episodes during training. The title of subplot shows the algo as well as the number of episodes needed to solve the environment
![alt text][image8]

Plot showing the score per episode over all the episodes during testing
![alt text][image9]

The top performing agent was duel double DQN that was able to solve the environment in 812 episodes and had an average score of 17.47 over 100 episodes during testing. Test results can be reproduced by running below
```
from utils import *
from workspace_utils import active_session
%matplotlib inline

eps_decay = [0.995]
a_double = [True]
a_duel = [True]

with active_session():
    run_option('test', eps_decay, a_double, a_duel, 4, n_episodes=100, chkpt_type='best_ma')
```

### Results for Visual Banana using Conv3D
Visual Banana training was done on local system and being slow have limited results

Plot showing the score per episode over all the episodes during training. The title of subplot shows the algo as well as the number of episodes needed to solve the environment
![alt text][image10]

Plot showing the score per episode over all the episodes during testing
![alt text][image11]

Test results can be reproduced by running below
```
from utils import *
from workspace_utils import active_session
%matplotlib inline

eps_decay = [0.995]
a_double = [False]
a_duel = [False]

with active_session():
    run_option('test', eps_decay, a_double, a_duel, 1, n_episodes=100, conv=ConvType.CONV3D, input_gray=False, chkpt_type='best_ma')
```

### Results for Visual Banana using Conv2D
Visual Banana training was done on local system and being slow have limited results

Plot showing the score per episode over all the episodes during training. The title of subplot shows the algo as well as the number of episodes needed to solve the environment
![alt text][image12]

Plot showing the score per episode over all the episodes during testing
![alt text][image13]

Test results can be reproduced by running below
```
from utils import *
from workspace_utils import active_session
%matplotlib inline

eps_decay = [0.995]
a_double = [False]
a_duel = [False]

with active_session():
    run_option('test', eps_decay, a_double, a_duel, 2, n_episodes=100, conv=ConvType.CONV2D, input_gray=False, chkpt_type='best_ma')
```

### Future Improvements

1. Add prioritized experience replay — Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated to the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.
2. Replace conventional exploration heuristics with [Noisy DQN](https://arxiv.org/abs/1706.10295). Parametric noise is added to the weights to induce stochasticity to the agent's policy, yielding more efficient exploration.
3. Explore other DQN improvements using [Rainbow](https://arxiv.org/pdf/1710.02298.pdf)
4. For visual banana, check out the duel, double dqn features, identify best network architecture, hyper parameters and above improvements too