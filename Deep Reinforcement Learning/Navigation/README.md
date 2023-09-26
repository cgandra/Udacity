[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

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
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

6. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Files and Folders
* Navigation.ipynb: Notebook for the basic-banana project. It mainly calls the high level function run_option for training/inference with different features. 
* Navigation_Pixels.ipynb: Notebook for the visual banana challenge project. It mainly calls the high level function run_option for training/inference with different features.
* utils.py: High-level calls to construct the environment, agent, run training/inference.
* dqn.py: the deep reinforcement learning algorithm routine.
* agent.py: Agent Class definition handles standard Q-Network or a Double-Q-Network.
* replayBuffer.py: Memory Buffer Class definition.
* model.py: Class definitions for DNN models to handle state input(FcModel) or pixel input(ConvModel3D, ConvModel2D). FcModel supports dqn and duel dqn networks.
* config.py: ModelConfig/AgentConfig Class definition.
* enum.py: enums used in various classes
* Folders results/results_pixels contain the saved model weights of the agent for banana/visual banana respectively
* Folder images contains the image files used in reports

### Instructions

The main function to call for training/inference is run_options in utils.py
```
run_option(mode, eps_decay, a_double, a_duel, i, moving_avg_tgt=25.0, n_episodes=2000, conv=ConvType.CONV1D, input_gray=False, chkpt_type='best', seed=0)
mode - "train" or "test" or "train_test"
eps_decay - list of decay values to iterate training/inference
a_double - true if double_dqn enabled else false
a_duel - true is dueling_dqn enabled else false
i - training/inference starting iteration number
moving_avg_tgt - target average score over 100 consecutive episodes
n_episodes - number of episodes to run training/inference for
conv - used to select pipeline for training basic banana(CONV1D) or visual banana(CONV2D/CONV3D) 
input_gray - gray or rgb input fro visual banana project
``` 
1. Training your own agent
In `Navigation.ipynb` run training as

```
from utils import *
from workspace_utils import active_session
%matplotlib inline

eps_decay = [0.995]
a_double = [True, False, True, False]
a_duel = [True, True, False, False]

with active_session():
    run_option('train', eps_decay, a_double, a_duel, 0, moving_avg_tgt=17.0, n_episodes=2000)
```
  
2. Testing your own agent
```
In `Navigation.ipynb` run inference as
from utils import *
from workspace_utils import active_session
%matplotlib inline

eps_decay = [0.995]
a_double = [True, False, True, False]
a_duel = [True, True, False, False]

with active_session():
    run_option('test', eps_decay, a_double, a_duel, 0, n_episodes=100, chkpt_type='best_ma')
```

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
