[//]: # (Image References)

[image1]: images/maddpg_algo.png
[image2]: images/maddpg_algo2.png
[image3]: images/maddpg_actor.png
[image4]: images/maddpg_critic.png
[image5]: images/maddpg_Train_Test.png

# Project 3: Collaboration and Competition

### Goal
The goal of this project is to train wo separate agents to play tennis. The goal of each agent is to keep the ball in play as long as possible. In this setup the agents need to collaborate under certain situations (eg not let ball hit the ground) and compete under other situations (eg gather as many + rewards as possible). 

### Learning Algorithm

1. Multi Agent RL Challenges
    - Partial observability
      - Agents do not have full access to the true state **s<sup>t</sup>**
      - Each agent receives a private partial observation **o<sup>t</sup><sub>i</sub>** correlated with **s<sup>t</sup>**
      - And chooses an action according to a policy conditioned on its own private observation, i.e. **a<sup>t</sup><sub>i</sub> = µ(o<sup>t</sup><sub>i</sub>; &theta;<sub>i</sub>)**
    - Non-stationarity
      - Environment moves into the next state **s<sup>t+1</sup>** according to actions of all
        agents, i.e. **s<sup>t+1</sup> = &Tau;(s<sup>t</sup> ; a<sup>t</sup><sub>1</sub> ; . . . . ; at<sup>t</sup><sub>N</sub>)**
      - It is non-stationary from the viewpoint of any agent: when any **µ<sub>i</sub> <> µ'<sub>i</sub>**    
        **P(o'<sub>i</sub>|o<sub>i</sub>, a<sub>1</sub>, . . . , a<sub>N</sub>, µ<sub>1</sub>, . . . , µ<sub>N</sub>) <> 
        P(o'<sub>i</sub>|o<sub>i</sub>, a<sub>1</sub>, . . . , a<sub>N</sub>, µ'<sub>1</sub>, . . . , µ'<sub>N</sub>)**
    - Markov assumption is violated due to partial observability and non-stationarity
    - Transitions in the experience replay become invalid due to non-stationarity
    - In most single agent algorithms it is assumed that the environment is stationary which leads to certain convergence guarantees. Hence under non-stationarity these no longer hold and this presents learning stability challenges
    - Policy gradient methods, on the other hand,  usually exhibit very high variance when coordination of multiple agents is required
 
2. Naive Solutions
    - Train agents independently 
      - Train in decentralised manner, i.e. **Q<sub>i</sub><sup>µ</sup>(o<sub>i</sub>, a<sub>i</sub>)**
      - Execute in decentralised manner, i.e. **µ<sub>i</sub>(o<sub>i</sub>)**

    - Train agents as a single Meta-agent
      - Takes into account existence of other agents. A single policy is learnt for all the agents.
      - Train in centralised manner, i.e. **Q<sub>i</sub><sup>µ</sup>(o<sub>1</sub>, a<sub>1</sub>, . . . , o<sub>N</sub> , a<sub>N</sub>)**
      - Execute in centralised manner, i.e. **µ<sub>i</sub>(o<sub>1</sub>, . . . , o<sub>N</sub> )**
      - Has Scalability issue: input size is multiplied by N for each one of N agents

3. Alternately Multi-agent Deep Deterministic Policy Gradient [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) proposes algorithm that
    - Leads to learned policies that only use local information (i.e. agents use their own observations) at execution time
    - Does not assume a differentiable model of the environment dynamics or any particular structure on the communication method between agents
    - Is applicable not only to cooperative interaction but to competitive or mixed interaction involving both physical and communicative behavior
  
4. Key approach of MADDPG
    - Train in centralised manner, i.e. **Q<sub>i</sub><sup>µ</sup>(o<sub>1</sub>, a<sub>1</sub>, . . . , o<sub>N</sub> , a<sub>N</sub>)**
    - Execute in decentralised manner, i.e. **µ<sub>i</sub>(o<sub>i</sub>)**
    - If agents know the actions taken by other agents, the environment is
      stationary even when any **µ<sub>i</sub> <> µ'<sub>i</sub>**    
        **P(o'<sub>i</sub>|o<sub>i</sub>, a<sub>1</sub>, . . . , a<sub>N</sub>, µ<sub>1</sub>, . . . , µ<sub>N</sub>) = 
        P(o'<sub>i</sub>|o<sub>i</sub>, a<sub>1</sub>, . . . , a<sub>N</sub>, µ'<sub>1</sub>, . . . , µ'<sub>N</sub>)**
    - During training, agents learn coordinated behaviours
      - Acheived with a simple extension of actor-critic policy gradient methods where the critic is augmented with extra information 
        about the policies of other agents and actor only has access to local information
    - At execution time, each agent acts according to its own learnt coordinated behaviour based on its own unique observations of the environment

    ![alt text][image1]
    ![alt text][image2]

5. In this project MADDPG algo was implemented to handle the competitive/cooperative requirement of training the two agents. 
   ```
    class MADDPGAgent:
        self.maddpg_agents = []
        for a in range(self.agent_cfg.num_agents):
            self.maddpg_agents.append(DDPGAgent(a, agent_cfg, actor_model_cfg, critic_model_cfg, training))
   ```
 
    A. The original [DDPG](https://arxiv.org/pdf/1509.02971.pdf) algorithm from which MADDPG is extended is described below
      - DDPG concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.
      - DDPG combines [DPG](http://proceedings.mlr.press/v32/silver14.pdf) with [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).  Recall that DQN (Deep Q-Network) stabilizes the learning of Q-function by experience replay and the frozen target network. The original DQN works in discrete space, and DDPG extends it to continuous space with the actor-critic framework to learn the action-value function Q(s,a;φ) while also learning a deterministic policy μθ(s) via DPG. Policy learning in DDPG is fairly simple. We want to learn a deterministic policy μθ(s) which gives the action that maximizes Q(s,a)
      - In addition, DDPG does soft updates (“conservative policy iteration”) on the parameters of both actor and critic, with τ≪1: θ′←τθ+(1−τ)θ′. In this way, the target network values are constrained to change slowly, different from the design in DQN that the target network stays frozen for some period of time.

    B. As done with Double Q-Learning in prev project, local and target networks are used to improve stability. This is where one set of parameters w is used to select the best action, and another set of parameters w' is used to evaluate that action. In this project, local and target networks are implemented separately for both the actor and the critic.
      ```
       class DDPGAgent():
            # Actor Network (w/ Target Network)
            self.policy = Model(actor_model_cfg, trainable=training)
            self.policy_target = Model(actor_model_cfg, trainable=False)
            self.policy.soft_update(self.policy_target, 1.0)

            # Critic Network (w/ Target Network)
            self.critic = Model(critic_model_cfg, critic=True, trainable=training)
            self.critic_target = Model(critic_model_cfg, critic=True, trainable=False)
            self.critic.soft_update(self.critic_target, 1.0)
      ```

    C. Experience replay
      - Allows the RL agent to learn from past experience
      - Experiences are stored in a single replay buffer as each agent interacts with the environment
      - These experiences are then utilized by the central critic, thereby allowing the agents to learn from each others' experiences. 
      ```
        class MADDPGAgent:
            def step_update(self, states, actions, rewards, next_states, dones, train):
                ....
                # Save experience in replay buffer
                self.replayBuf.add(states, actions, rewards, next_states, dones)
      ```

    D. Exploration vs Exploitation
      - In DDPG in order to do better exploration, an exploration policy μ’ is constructed by adding noise N
        ```
         μ′(s)=μθ(s)+N
        ```
      - The authors of the original DDPG paper recommended time-correlated [OU noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well
      - Current implementation handles uses Gaussian Noise per ddpg agent. Noise is reduced by scaling over the course of training to facilitate getting higher-quality training data
      ```
        class DDPGAgent():
           def act(self, states, scale, add_noise=True):
            """Returns actions for given state as per current policy."""        

            states = states.astype(np.float32)
            states = np.expand_dims(states, 0)
            actions = self.policy.predict(states, cpu_mem=True)
            if add_noise:
                noise = scale*self.noise.sample()
                self.avg_noise += np.mean(abs(noise))
                actions += noise

            return np.clip(actions, -1, 1)
      ```

    E. Learning
      ```
      class MADDPGAgent:
        def learn(self):
            experiences = self.replayBuf.sample()
            states, actions, rewards, next_states, dones = experiences

            actions_next = [self.maddpg_agents[a].policy_target.predict_detach(next_states[:, a, :].squeeze()) for a in range(self.agent_cfg.num_agents)]
            actions_next = self.maddpg_agents[0].policy_target.cat_tensor(actions_next, 1)

            actions_pred = [self.maddpg_agents[a].policy.forward(states[:, a, :].squeeze()) for a in range(self.agent_cfg.num_agents)]
            actions_pred = self.maddpg_agents[0].policy.cat_tensor(actions_pred, 1)
        
            next_states = next_states.reshape(self.agent_cfg.minibatch_size,-1)
            states = states.reshape(self.agent_cfg.minibatch_size,-1)
            actions = actions.reshape(self.agent_cfg.minibatch_size,-1)
            states, actions, rewards, next_states, dones = self.maddpg_agents[0].policy.get_tensors(states, actions, rewards, next_states, dones)

            for a in range(self.agent_cfg.num_agents):
                self.maddpg_agents[a].learn(states, actions, rewards[a], next_states, dones[a], actions_next, actions_pred)  
      ```
      ```
      class DDPGAgent:
        def learn(self, states, actions, rewards, next_states, dones, actions_next, actions_pred):

            # ---------------------------- update critic ---------------------------- #
            Q_targets_next = self.critic_target.predict_detach(next_states, actions_next).squeeze()
    
            ## compute and minimize the loss
            self.critic.train(states, actions, Q_targets_next, dones, rewards, self.agent_cfg.gamma)

            # ---------------------------- update actor ---------------------------- #
            actor_loss = -self.critic.forward(states, actions_pred).mean()

            ## compute and minimize the loss
            self.policy.optimize(actor_loss, retain_graph=True)
      ```

### Network Architecture for Actor Network
Network architecture was chosen based off experiments done with DDPG for continuous control project. Additional arch's need to be studied

1. DDPG Actor network:
```
Actor(
  (conv): Sequential(
    (0): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Linear(in_features=24, out_features=320, bias=False)
    (2): BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Linear(in_features=320, out_features=160, bias=False)
    (5): BatchNorm1d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
  )
  (fcout): Linear(in_features=160, out_features=2, bias=True)
)
```
![alt text][image3]

2. DDPG Critic network:
```
Critic(
  (bn): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=48, out_features=320, bias=False)
  (bn1): BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv): Sequential(
    (0): Linear(in_features=324, out_features=160, bias=True)
    (1): ReLU()
  )
  (fcout): Linear(in_features=160, out_features=1, bias=True)
)
```
![alt text][image4]

```
  MADDPGAgent:
  self.maddpg_agents = []
  for a in range(self.agent_cfg.num_agents):
    self.maddpg_agents.append(DDPGAgent(a, agent_cfg, actor_model_cfg, critic_model_cfg, training))

  DDPGAgent:
  # Actor Network (w/ Target Network)
  self.policy = Model(actor_model_cfg, trainable=training)
  self.policy_target = Model(actor_model_cfg, target=True, trainable=False)

  # Critic Network (w/ Target Network)
  self.critic = Model(critic_model_cfg, critic=True, trainable=training)
  self.critic_target = Model(critic_model_cfg, critic=True, target=True, trainable=False)
```

### Hyperparameters
These hyperparameters were tuned via experimentation

    <class 'config.AgentConfig'>: {'state_size': 24, 'action_size': 2, 'num_agents': 2, 'algo': 'maddpg', 'noise': 'Gaussian', 'n_episodes': 100, 'max_t_steps': 2000, 'replay_buf_size': 1000000, 'min_replay_buf_size': 256, 'minibatch_size': 256, 'training_freq': 1, 'update_tgt_freq': 1, 'tau': 0.001, 'gamma': 0.99, 'eps_schedule': 'geometric', 'eps_start': 1.0, 'eps_end': 0.01, 'eps_decay': 0.9999, 'seed': 0]
    Actor <class 'config.ModelConfig'>: {'learning_rate': 5e-05, 'weight_decay': 0, 'input_dim': 24, 'output_dim': 2, 'append_dim': 0, 'nflts': [320, 160], 'seed': 0}
    Critic <class 'config.ModelConfig'>: {'learning_rate': 5e-05, 'weight_decay': 0, 'input_dim': 48, 'output_dim': 1, 'append_dim': 4, 'nflts': [320, 160], 'seed': 0}

### Results
Plot showing the score per episode over all the episodes during training and testing. The title of subplot shows the number of episodes needed to solve the environment
![alt text][image5]

Test results can be reproduced by running below
```
from utils import *
from workspace_utils import active_session
%matplotlib inline

with active_session():
    run_option('test', 0, n_episodes=100, seed=0)
```

### Future Improvements

1. Run the experiment for larger moving avg target to see if it always stabilizes (issue identified in udacity's Benchmark implementation)
2. Add prioritized experience replay — Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated to the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.
3. Experiment with other algorithms such as [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477), [Proximal Policy Optimization (PPO)](Proximal Policy Optimization Algorithms), [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617) or [A3C](https://arxiv.org/pdf/1602.01783.pdf) instead of DDPG
4. Study different network architectures (depths) for actor/critic and impact on training/inference. I found different filter sizes in each layer leads to instability sometimes.
