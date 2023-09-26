import random
import pickle
import numpy as np

from enums import ConvType
from model import Model
from replayBuffer import ReplayBuffer

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_cfg, model_cfg, training=True):
        """Initialize an Agent object.
        
        Params
        ======
            agent_cfg: Agents configuration parameters
            model_cfg: Model configuration parameters
            training (bool): Training/Eval mode
        """
        self.agent_cfg = agent_cfg
        self.model_cfg = model_cfg
        self.agent_cfg.min_replay_buf_size += self.agent_cfg.state_augment_size

        random.seed(agent_cfg.seed)
        print(self.agent_cfg)
        print(self.model_cfg)

        # Q-Network
        self.policy = Model(model_cfg, trainable=training)
        self.policy_target = Model(model_cfg, trainable=False)
        self.policy.soft_update(self.policy_target, 1.0)

        # Replay replayBuf
        self.replayBuf = ReplayBuffer(agent_cfg.replay_buf_size, agent_cfg.minibatch_size, agent_cfg.seed, augment_size=self.agent_cfg.state_augment_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.episode_done = False

        # Initialize epsilon for epsilon-greedy action selection
        self.eps = self.agent_cfg.eps_start

        self.set_training_mode(training)
        self.start_episode = self.load(agent_cfg.ckpt_path, agent_cfg.replay_path)

    def save(self, ckpt_path, episode, replay_path=None):
        if ckpt_path is None:
            return

        if replay_path and (len(self.replayBuf) >= 1):
            f = open(replay_path, 'wb')
            states, actions, rewards, next_states, dones = self.replayBuf.stack(self.replayBuf.memory)
            replay_dict = {'states': states,
                          'actions': actions,
                          'rewards': rewards,
                          'next_states': next_states,
                          'dones': dones}
            pickle.dump(replay_dict, f)
            f.close()

        state_dict = {'policy': self.policy.net.state_dict(),
                'policy_target': self.policy_target.net.state_dict(),
                'policy_optim': self.policy.optimizer.state_dict(),
                'episode' : episode}
        self.policy.save(state_dict, ckpt_path)

    def load(self, ckpt_path, replay_path=None):
        if ckpt_path is None:
            return 1

        state_dict = self.policy.load(ckpt_path)
        print('load episode: {}'.format(state_dict['episode']))
        self.policy.net.load_state_dict(state_dict['policy'])
        if self.training:
            self.policy_target.net.load_state_dict(state_dict['policy_target'])
            self.policy.optimizer.load_state_dict(state_dict['policy_optim'])

        if self.training and replay_path:
            f = open(replay_path, 'rb')
            replay_dict = pickle.load(f)
            for i in range(len(replay_dict['states'])):
                self.replayBuf.add(replay_dict['states'][i], replay_dict['actions'][i], replay_dict['rewards'][i], replay_dict['next_states'][i], replay_dict['dones'][i])
            f.close()

        if self.training==False:
            state_dict['episode'] = 0
        
        state_dict['episode'] += 1
        return state_dict['episode']

    def set_training_mode(self, training=True):
        self.training = training
        if training == False:
            self.eps = 0.

    def augment_state(self, state):
        if self.model_cfg.conv == ConvType.CONV1D:
            return state

        return self.replayBuf.augment_state(state, self.model_cfg.conv)

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """

        # Epsilon-greedy action selection
        rval = random.random()
        if self.training and (rval <= self.eps):
            action = random.randrange(self.agent_cfg.action_size)
            return int(action)

        state = state.astype(np.float32)
        qvals_online = self.policy.predict(state, cpu_mem=True)
        action = np.argmax(qvals_online)

        return int(action)

    def step_update(self, state, action, reward, next_state, done, train):
        if self.episode_done:
            self.t_step = 0

        self.t_step = self.t_step + 1
        self.episode_done = done or ((self.t_step%self.agent_cfg.max_t_steps)==0)

        if train==False:
            return self.episode_done

        # Save experience in replay buffer
        self.replayBuf.add(state, action, reward, next_state, done)
        if len(self.replayBuf) < self.agent_cfg.min_replay_buf_size:
            return self.episode_done

        self.update_epsilon(self.episode_done)

        # Learn every training_freq time steps.
        # If enough samples are available in replayBuf, get random subset and learn
        if ((self.t_step % self.agent_cfg.training_freq)  == 0):
            self.learn()

        # Update target network every update_tgt_freq time steps.
        if ((self.t_step % self.agent_cfg.update_tgt_freq)  == 0):
            self.policy.soft_update(self.policy_target, self.agent_cfg.tau)

        return self.episode_done

    def update_epsilon(self, ret):
        if (self.agent_cfg.eps_schedule == 'linear'):
            self.eps = max(self.agent_cfg.eps_end, self.eps - self.agent_cfg.eps_decay)

        if (self.agent_cfg.eps_schedule == 'geometric') and ret:
            self.eps = max(self.agent_cfg.eps_end, self.agent_cfg.eps_decay*self.eps)


    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        experiences = self.replayBuf.sample(self.model_cfg.conv)
        
        states, actions, rewards, next_states, dones = experiences
        next_states = self.policy.get_tensor(next_states)

        # Get max predicted Q values (for next states) from target model
        qvals_tgt = self.policy_target.predict_detach(next_states)
        if self.model_cfg.double_dqn:
            # DQN network selects best action to take for the next state, i.e action with the highest Q value             
            actions_ns = self.policy.predict_detach(next_states)
            best_actions_indices = actions_ns.argmax(dim=1, keepdim=True)

            # Target network calculates the target Q value of taking that action at the next state
            Q_targets_next = qvals_tgt.gather(1, best_actions_indices)
        else:
            # Max Q among all possible actions from next state
            Q_targets_next, _ = qvals_tgt.max(dim=1, keepdim=True)

        ## compute and minimize the loss
        self.policy.train(states, actions, Q_targets_next, dones, rewards, self.agent_cfg.gamma)