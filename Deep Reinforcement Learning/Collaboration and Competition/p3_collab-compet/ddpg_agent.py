import numpy as np

from model import Model
from noise import OUNoise, GaussianNoise

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_id, agent_cfg, actor_model_cfg, critic_model_cfg, training=True):
        """Initialize an Agent object.
        
        Params
        ======
            agent_cfg: Agents configuration parameters
            actor_model_cfg: Actor model configuration parameters
            critic_model_cfg: Critic model configuration parameters
            training (bool): Training/Eval mode
        """
        self.agent_id = agent_id
        self.agent_cfg = agent_cfg
        self.actor_model_cfg = actor_model_cfg
        self.critic_model_cfg = critic_model_cfg

        print(self.agent_cfg)
        print(self.actor_model_cfg)
        print(self.critic_model_cfg)

        # Actor Network (w/ Target Network)
        self.policy = Model(actor_model_cfg, trainable=training)
        self.policy_target = Model(actor_model_cfg, trainable=False)
        self.policy.soft_update(self.policy_target, 1.0)

        # Critic Network (w/ Target Network)
        self.critic = Model(critic_model_cfg, critic=True, trainable=training)
        self.critic_target = Model(critic_model_cfg, critic=True, trainable=False)
        self.critic.soft_update(self.critic_target, 1.0)

        # Noise process
        if agent_cfg.noise == 'Gaussian':
            self.noise = GaussianNoise(agent_cfg.action_size, agent_cfg.seed+agent_id)
        else:
            self.noise = OUNoise((1, agent_cfg.action_size), agent_cfg.seed+agent_id)

        self.training = training
        self.reset()

    def reset(self):
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.avg_noise = 0
        self.noise.reset()

    def save(self, ckpt_path, episode):
        if ckpt_path is None:
            return

        state_dict = {'policy': self.policy.net.state_dict(),
                'critic': self.critic.net.state_dict(),
                'target_policy': self.policy_target.net.state_dict(),
                'target_critic': self.critic_target.net.state_dict(),
                'policy_optim': self.policy.optimizer.state_dict(),
                'critic_optim': self.critic.optimizer.state_dict(),
                'episode' : episode}
        self.policy.save(state_dict, ckpt_path)

    def load(self, ckpt_path):
        if ckpt_path is None:
            return 0

        state_dict = self.policy.load(ckpt_path)
        self.policy.net.load_state_dict(state_dict['policy'])
        if self.training:
            self.critic.net.load_state_dict(state_dict['critic'])
            self.policy_target.net.load_state_dict(state_dict['target_policy'])
            self.critic_target.net.load_state_dict(state_dict['target_critic'])
            self.policy.optimizer.load_state_dict(state_dict['policy_optim'])
            self.critic.optimizer.load_state_dict(state_dict['critic_optim'])

        return state_dict['episode']

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

    def learn(self, states, actions, rewards, next_states, dones, actions_next, actions_pred):

        # ---------------------------- update critic ---------------------------- #
        Q_targets_next = self.critic_target.predict_detach(next_states, actions_next).squeeze()

        ## compute and minimize the loss
        self.critic.train(states, actions, Q_targets_next, dones, rewards, self.agent_cfg.gamma)

        # ---------------------------- update actor ---------------------------- #
        actor_loss = -self.critic.forward(states, actions_pred).mean()

        ## compute and minimize the loss
        self.policy.optimize(actor_loss, retain_graph=True)

    def soft_update(self):
        self.critic.soft_update(self.critic_target, self.agent_cfg.tau)
        self.policy.soft_update(self.policy_target, self.agent_cfg.tau)
