import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, model_cfg):
        """Initialize parameters and build model.
        Params
        ======
            model_cfg: Configuration to build model
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.model_cfg = model_cfg
        self.conv = nn.Sequential(
            nn.BatchNorm1d(model_cfg.input_dim),
            nn.Linear(model_cfg.input_dim, model_cfg.nflts[0], bias=False),
            nn.BatchNorm1d(model_cfg.nflts[0]),
            nn.ReLU(),
            nn.Linear(model_cfg.nflts[0], model_cfg.nflts[1], bias=False),
            nn.BatchNorm1d(model_cfg.nflts[1]),
            nn.ReLU()
            )

        self.fcout = nn.Linear(model_cfg.nflts[1], model_cfg.output_dim)
        self.reset_parameters()
        
    def init_normal(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(*hidden_init(m))

    def reset_parameters(self):
        self.conv.apply(self.init_normal)
        self.fcout.weight.data.uniform_(-3e-3, 3e-3)
        self.fcout.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv(state)
        qvals = F.tanh(self.fcout(x))  
        return qvals

class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, model_cfg):
        """Initialize parameters and build model.
        Params
        ======
            model_cfg: Configuration to build model
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()

        self.model_cfg = model_cfg
        self.bn = nn.BatchNorm1d(model_cfg.input_dim)
        self.fc1 = nn.Linear(model_cfg.input_dim, model_cfg.nflts[0], bias=False)
        self.bn1 = nn.BatchNorm1d(model_cfg.nflts[0])

        self.conv = nn.Sequential(
            nn.Linear(model_cfg.nflts[0]+model_cfg.append_dim, model_cfg.nflts[1]),
            nn.ReLU()
            )

        self.fcout = nn.Linear(model_cfg.nflts[1], model_cfg.output_dim)
        self.reset_parameters()
        
    def init_normal(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(*hidden_init(m))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.conv.apply(self.init_normal)
        self.fcout.weight.data.uniform_(-3e-3, 3e-3)
        self.fcout.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fc1(self.bn(state))))
        x = torch.cat((xs, action), dim=1)
        x = self.conv(x)
        return self.fcout(x)

class Model(object):

    def __init__(self, model_cfg, critic=False, trainable=True):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
            trainable : Model is trainable or not
        """
        super(Model, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(model_cfg.seed)
        torch.cuda.manual_seed_all(model_cfg.seed)
        self.trainable = trainable
        self.model_cfg = model_cfg
        self.critic = critic

        if critic:
            self.net = Critic(model_cfg)
        else:
            self.net = Actor(model_cfg)

        model = self.net.to(self.device)
        print(model)

        if self.trainable:
            self.optimizer = optim.Adam( self.net.parameters(), lr=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def load(self, ckpt_path):
        if (torch.cuda.is_available()):
            state_dict = torch.load(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        return state_dict

    def save(self, state_dict, ckpt_path):
        torch.save(state_dict, ckpt_path)

    def soft_update(self, tgt_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_online + (1 - τ)*θ_target

        Params
        ======
            online_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, online_param in zip(tgt_net.net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)

    def train(self, states, actions, q_targets_next, dones, rewards, gamma):
        if self.trainable==False:
            return

        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from online model
        q_expected = self.net(states, actions).squeeze()

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimize(loss)

    def optimize(self, loss, retain_graph=False):
        if self.trainable==False:
            return

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        #torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.optimizer.step()

    def predict(self, state, actions=None, detach=False, cpu_mem=False, clamp=False):
        if torch.is_tensor(state)==False:
            state = torch.from_numpy(state).float().to(self.device)
        self.net.eval()
        if detach:
            if self.critic:
                q_vals = self.net(state, actions).detach()
            else:
                q_vals = self.net(state).detach()
        else:
            with torch.no_grad():
                q_vals = self.net(state)

        if clamp:
            q_vals = torch.clamp(q_vals, -1, 1)    
        if cpu_mem:
        	q_vals = q_vals.cpu().data.numpy()
        self.net.train()
        return q_vals

    def predict_detach(self, state, actions=None, clamp=False):
        if torch.is_tensor(state)==False:
            state = torch.from_numpy(state).float().to(self.device)
        if (actions is not None) and (torch.is_tensor(actions)==False):
            actions = torch.from_numpy(actions).float().to(self.device)

        with torch.no_grad():
            if self.critic:
                q_vals = self.net(state, actions)
            else:
                q_vals = self.net(state)
        if clamp:
            q_vals = torch.clamp(q_vals, -1, 1)
        return q_vals

    def forward(self, state, actions=None):
        if torch.is_tensor(state)==False:
            state = torch.from_numpy(state).float().to(self.device)

        if (actions is not None) and (torch.is_tensor(actions)==False):
            actions = torch.from_numpy(actions).float().to(self.device)

        if self.critic:
            return self.net(state, actions)
        else:
            return self.net(state)

    def get_tensor(self, input):
        input = torch.from_numpy(input).float().to(self.device)
        return input

    def cat_tensor(self, input, dim):
        return torch.cat(input, dim=dim)

    def get_tensors(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(states).float().to(self.device)                
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        return states, actions, rewards, next_states, dones
