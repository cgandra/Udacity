[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction
The goal of this project is to train two RL agents to play tennis. The goal of each agent is to keep the ball in play as long as possible

### Environment

For this project, we will work with modified version the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

* In this environment, two agents control rackets to bounce a ball over a net.
* Each agent has an observation space of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own local observation for 3 timestamps, hence input state is 24 dims.  
* For each agent two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
* If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
* The environment is solved when the agents get an average score of +0.5 over 100 consecutive episodes
  - After each episode, the rewards that each agent received are added up(without discounting), to get a score for each agent. This yields 2 (potentially different) scores.
  - Maximum of these 2 scores yields a single **score** for each episode.
* The task is episodic. It resets either by reaching the maximum number of steps (1000) or with the ball dropping to the floor or thrown out of bounds.

### Getting Started

1. Create (and activate) a new environment with Python 3.6
    
	```
    conda create --name drlnd python=3.6
    activate drlnd
	```
2. Minimal install of OpenAI gym
	```
	git clone https://github.com/openai/gym.git
	cd gym
	pip install -e .
	```
3. Clone the repository and navigate to the python/ folder. Then, install several dependencies
	```
	git clone https://github.com/udacity/deep-reinforcement-learning.git
	cd deep-reinforcement-learning/python
	pip install .
	```
4. Create an IPython kernel for the drlnd environment
	```
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
	```
5. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

6. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Files and Folders
* Tennis.ipynb: Notebook for the project. It mainly calls the high level function run_option for training/inference with different features. 
* utils.py: High-level calls to construct the environment, agent, run training/inference.
* agent.py: Agent Class definition handles MADDPG algo
* ddpg_agent.py: Agent Class definition handles DDPG algo
* replayBuffer.py: Memory Buffer Class definition.
* model.py: Class definitions for Actor/Critic Networks.
* noise.py: Ornstein-Uhlenbeck noise & Gaussian noise classes.
* config.py: ModelConfig/AgentConfig Class definition.
* Folder results contains the saved model weights of the agent
* Folder images contains the image files used in reports

### Instructions

The main function to call for training/inference is run_options in utils.py
```
run_option(mode, i, moving_avg_tgt=30.0, n_episodes=2000, chkpt_type='best_ma', seed=0)
mode - "train" or "test" or "train_test"
i - training/inference starting iteration number
moving_avg_tgt - target average score over 100 consecutive episodes
n_episodes - number of episodes to run training/inference for
``` 

1. Training your own agent

In `Tennis.ipynb` run training as

```
from utils import *
from workspace_utils import active_session
%matplotlib inline
with active_session():
    run_option('train', 0, moving_avg_tgt=1.8, n_episodes=5000, seed=0)
```
  
2. Testing your own agent

In `Tennis.ipynb` run inference as

```
from utils import *
from workspace_utils import active_session
%matplotlib inline

with active_session():
    run_option('test', 0, n_episodes=100, seed=0)
```

3. Saved Model Weights

Pls use results\best_ma_chkpt_0.pth

### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Soccer** environment.

![Soccer][image2]

In this environment, the goal is to train a team of agents to play soccer.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Soccer.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agents on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agents without enabling a virtual screen, but you will be able to train the agents.  (_To watch the agents, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
