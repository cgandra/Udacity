import pickle
import numpy as np

from model import Model
from replayBuffer import ReplayBuffer
from noise import OUNoise

class Agent():
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
        self.actor_model_cfg = actor_model_cfg
        self.critic_model_cfg = critic_model_cfg

        print(self.agent_cfg)
        print(self.actor_model_cfg)
        print(self.critic_model_cfg)

        # Actor Network (w/ Target Network)
        self.policy = Model(actor_model_cfg, trainable=training)
        self.policy_target = Model(actor_model_cfg, target=True, trainable=False)
        self.policy.soft_update(self.policy_target, 1.0)

        # Critic Network (w/ Target Network)
        self.critic = Model(critic_model_cfg, critic=True, trainable=training)
        self.critic_target = Model(critic_model_cfg, critic=True, target=True, trainable=False)
        self.critic.soft_update(self.critic_target, 1.0)

        # Noise process
        self.noise = OUNoise((agent_cfg.num_agents, agent_cfg.action_size), agent_cfg.seed)

        # Replay replayBuf
        self.replayBuf = ReplayBuffer(agent_cfg.action_size, agent_cfg.replay_buf_size, agent_cfg.minibatch_size, agent_cfg.seed)

        self.reset()
        self.training=training
        self.start_episode = self.load(agent_cfg.ckpt_path, agent_cfg.replay_path)

    def reset(self):
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.avg_noise = 0
        self.noise.reset()
        self.episode_done = False

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
                'critic': self.critic.net.state_dict(),
                'policy_target': self.policy_target.net.state_dict(),
                'target_critic': self.critic_target.net.state_dict(),
                'policy_optim': self.policy.optimizer.state_dict(),
                'critic_optim': self.critic.optimizer.state_dict(),
                'episode' : episode}
        self.policy.save(state_dict, ckpt_path)

    def load(self, ckpt_path, replay_path=None):
        if ckpt_path is None:
            return 1

        state_dict = self.policy.load(ckpt_path)
        print('load episode: {}'.format(state_dict['episode']))
        self.policy.net.load_state_dict(state_dict['policy'])
        if self.training:
            self.critic.net.load_state_dict(state_dict['critic'])
            self.policy_target.net.load_state_dict(state_dict['policy_target'])
            self.critic_target.net.load_state_dict(state_dict['target_critic'])
            self.policy.optimizer.load_state_dict(state_dict['policy_optim'])
            self.critic.optimizer.load_state_dict(state_dict['critic_optim'])

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

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""        

        states = states.astype(np.float32)
        actions = self.policy.predict(states, cpu_mem=True)
        if add_noise:
            noise = self.noise.sample()
            self.avg_noise += np.mean(abs(noise))
            actions += noise
        return np.clip(actions, -1, 1)


    def step_update(self, states, actions, rewards, next_states, dones, train):
        if self.episode_done:
            self.t_step = 0

        self.t_step = self.t_step + 1
        self.episode_done = np.any(dones) or ((self.t_step%self.agent_cfg.max_t_steps)==0)

        if train==False:
            return self.episode_done

        # Save experience in replay buffer
        for n in range(self.agent_cfg.num_agents):
            self.replayBuf.add(states[n], actions[n], rewards[n], next_states[n], dones[n])

        if len(self.replayBuf) < self.agent_cfg.min_replay_buf_size:
            return self.episode_done

        # Learn every training_freq time steps.
        # If enough samples are available in replayBuf, get random subset and learn
        if ((self.t_step % self.agent_cfg.training_freq)  == 0):
            self.learn()

        # Update target network every update_tgt_freq time steps.
        if ((self.t_step % self.agent_cfg.update_tgt_freq)  == 0):
            self.policy.soft_update(self.policy_target, self.agent_cfg.tau)
            self.critic.soft_update(self.critic_target, self.agent_cfg.tau)
        
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
        next_states = self.policy_target.get_tensor(next_states)
        states = self.policy_target.get_tensor(states)

        # ---------------------------- update critic ---------------------------- #
        actions_next = self.policy_target.predict(next_states)
        Q_targets_next = self.critic_target.predict(next_states, actions_next)

        ## compute and minimize the loss
        self.critic.train(states, actions, Q_targets_next, dones, rewards, self.agent_cfg.gamma)

        # ---------------------------- update actor ---------------------------- #
        actions_pred = self.policy.forward(states)
        actor_loss = -self.critic.forward(states, actions_pred).mean()

        ## compute and minimize the loss
        self.policy.optimize(actor_loss)