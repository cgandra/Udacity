import pickle
import numpy as np

from ddpg_agent import DDPGAgent
from replayBuffer import ReplayBuffer

class MADDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_cfg, actor_model_cfg, critic_model_cfg, training=True):
        """Initialize an Agent object.
        
        Params
        ======
            agent_cfg: Agents configuration parameters
            actor_model_cfg: Actor model configuration parameters
            critic_model_cfg: Critic model configuration parameters
            training (bool): Training/Eval mode
        """
        self.agent_cfg = agent_cfg

        self.maddpg_agents = []
        for a in range(self.agent_cfg.num_agents):
            self.maddpg_agents.append(DDPGAgent(a, agent_cfg, actor_model_cfg, critic_model_cfg, training))

        # Replay replayBuf
        self.replayBuf = ReplayBuffer(agent_cfg.num_agents, agent_cfg.replay_buf_size, agent_cfg.minibatch_size, agent_cfg.seed)

        # Initialize epsilon
        self.eps = self.agent_cfg.eps_start
        self.episode_done = False
        self.reset()

        self.training = training
        self.start_episode = self.load(agent_cfg.ckpt_paths, agent_cfg.replay_path)

    def reset(self):
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        for a in range(self.agent_cfg.num_agents):
            self.maddpg_agents[a].reset()
        self.update_epsilon(self.episode_done)
        self.episode_done = False

    def get_avg_noise(self, agent_id):
        avg_noise = self.maddpg_agents[agent_id].avg_noise/self.t_step
        return avg_noise

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for a in range(self.agent_cfg.num_agents):
            actions_a = self.maddpg_agents[a].act(states[a], self.eps, add_noise)
            actions.append(actions_a)
        actions = np.concatenate(actions, 0)
        
        return actions

    def step_update(self, states, actions, rewards, next_states, dones, train):
        if self.episode_done:
            print('\nt_step: {}'.format(self.t_step))
            self.t_step = 0

        self.t_step = self.t_step + 1
        self.episode_done = np.all(dones) or ((self.t_step%self.agent_cfg.max_t_steps)==0)

        if train==False:
            return self.episode_done

        # Save experience in replay buffer
        self.replayBuf.add(states, actions, rewards, next_states, dones)

        if len(self.replayBuf) < self.agent_cfg.min_replay_buf_size:
            return self.episode_done

        # Learn every training_freq time steps.
        # If enough samples are available in replayBuf, get random subset and learn
        if ((self.t_step % self.agent_cfg.training_freq)  == 0):
            self.learn()

        # Update target network every update_tgt_freq time steps.
        if ((self.t_step % self.agent_cfg.update_tgt_freq)  == 0):
            for a in range(self.agent_cfg.num_agents):
                self.maddpg_agents[a].soft_update()
        
        return self.episode_done

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, policy_target(next_state))

        Params
        ======
            experiences (Tuple): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
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

    def save(self, ckpt_paths, episode, replay_path=None):
        if (ckpt_paths is None) or (len(ckpt_paths) < self.agent_cfg.num_agents):
            return

        for a in range(self.agent_cfg.num_agents):
            self.maddpg_agents[a].save(ckpt_paths[a], episode)

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

    def load(self, ckpt_paths, replay_path=None):
        if (ckpt_paths is None) or (len(ckpt_paths) < self.agent_cfg.num_agents):
            return 1

        for a in range(self.agent_cfg.num_agents):
            episode = self.maddpg_agents[a].load(ckpt_paths[a])
        print('load episode: {}'.format(episode))

        if self.training and replay_path:
            f = open(replay_path, 'rb')
            replay_dict = pickle.load(f)
            for i in range(len(replay_dict['states'][0])):
                states = np.vstack([replay_dict['states'][a][i] for a in range(self.num_agents)])
                actions = np.vstack([replay_dict['actions'][a][i] for a in range(self.num_agents)])
                rewards = np.vstack([replay_dict['rewards'][a][i] for a in range(self.num_agents)])
                next_states = np.vstack([replay_dict['next_states'][a][i] for a in range(self.num_agents)])
                dones = np.vstack([replay_dict['dones'][a][i] for a in range(self.num_agents)])
                self.replayBuf.add(states, actions, rewards, next_states, dones)
            f.close()

        if self.training==False:
            episode = 0
        
        episode += 1
        return episode

    def update_epsilon(self, ret):
        if (self.agent_cfg.eps_schedule == 'geometric') and ret:
            self.eps = max(self.agent_cfg.eps_end, self.agent_cfg.eps_decay*self.eps)
