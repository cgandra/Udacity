[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

This project consists of an implementation of an RL-based agent to control various simulated double-jointed arm and track them. The goal of each agent is to maintain its position at the target location for as many time steps as possible.

### Environment

For this project, we work with the modified version of the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

* There are two separate versions of the Unity environment:
  A. The first version contains a single agent.
  B. The second version contains 20 identical agents, each with its own copy of the environment.  
* Each agent has an observation space consisting of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
* Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.
* A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
* The task is episodic, with a maximum of 1000 steps per episode
* In order to solve the environment
  * In first version, the agent must get an average score of +30 over 100 consecutive episodes
  * In second version, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically
    - After each episode, the rewards that each agent received (without discounting) are added up to get a score for each agent. This yields 20 (potentially different) scores. 
    - Average of these 20 scores (over all 20 agents) is taken. This yields an **average score** for each episode.
    - The average over 100 episodes of these **average scores** is at least +30

### Observation-space
Each agent has an observation space consisting of 33 variables corresponding to 
  * Relative position of the two links (3d vector) and their orientation (4d quaternion).
  * Linear velocity (3d-vector) and angular velocity (3d-vector) of each link.
  * Relative positions of the end effector (3d-vector) and the goal (3d-vector).
  * Speed (scalar) of the moving goal.

### Continuous Action-space
The action space is continuous, which allows each agent to execute more complex and precise movements to control the robotic arm. The actions correspond to the torques applicable to two joints along the x and z axes respectively for each joint

### Solving the Environment

Project submission solves both versions of the environment. 

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

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

6. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Files and Folders
* Continuous_Control.ipynb: Notebook for the project. It mainly calls the high level function run_option for training/inference with different features. 
* utils.py: High-level calls to construct the environment, agent, run training/inference.
* agent.py: Agent Class definition handles DDPG algo
* replayBuffer.py: Memory Buffer Class definition.
* model.py: Class definitions for Actor/Critic Networks.
* noise.py: Ornstein-Uhlenbeck noise class.
* config.py: ModelConfig/AgentConfig Class definition.
* Folders results_1/results_20 contain the saved model weights of the agent for single agent/20 agents versions of the enviroment respectively
* Folder images contains the image files used in reports

### Instructions

The main function to call for training/inference is run_options in utils.py
```
run_option(mode, i, agent_type='single', moving_avg_tgt=30.0, n_episodes=2000, chkpt_type='best', seed=0)
mode - "train" or "test" or "train_test"
i - training/inference starting iteration number
agent_type - 'single' or 'multi'
moving_avg_tgt - target average score over 100 consecutive episodes
n_episodes - number of episodes to run training/inference for
``` 

1. Training your own agent

In `Continuous_Control.ipynb` run training as

```
from utils import *
from workspace_utils import active_session
%matplotlib inline

with active_session():
    run_option('train', 0, agent_type='single', moving_avg_tgt=35.0, n_episodes=2000, seed=0)```

or

with active_session():
    run_option('train', 0, agent_type='multi', moving_avg_tgt=35.0, n_episodes=2000, seed=0)```
```
  
2. Testing your own agent

In `Continuous_Control.ipynb` run inference as

```
from utils import *
from workspace_utils import active_session
%matplotlib inline

with active_session():
    run_option('test', 0, agent_type='single', n_episodes=100, seed=0)

or

with active_session():
    run_option('test', 0, agent_type='multi', n_episodes=100, seed=0)

```

3. Saved Model Weights

Pls use results_1\best_ma_chkpt_0.pth & results_20\best_ma_chkpt_0.pth

### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Crawler** environment.

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

