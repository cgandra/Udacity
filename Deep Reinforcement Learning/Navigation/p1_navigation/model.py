import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from enums import ConvType

class FcModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, model_cfg, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            model_cfg: Configuration to build model
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FcModel, self).__init__()

        self.model_cfg = model_cfg
        self.conv = nn.Sequential(
            nn.Linear(model_cfg.input_dim, fc1_units, bias=True),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU()
            )

        self.fc3 = nn.Linear(fc2_units, model_cfg.output_dim)
        if self.model_cfg.dueling_dqn:
            #fc3 is advantage stream,  value is value stream
            self.fc3_value= nn.Linear(fc2_units, 1)

    def forward(self, input):
        """Build a network that maps state -> action values."""
        x = self.conv(input)
        if self.model_cfg.dueling_dqn:
            advantage = self.fc3(x)
            value = self.fc3_value(x)
            qvals = value + (advantage - advantage.mean()) 
        else:
            qvals = self.fc3(x)        
        return qvals

class ConvModel3D(nn.Module):
    
    def __init__(self, model_cfg):
        """Initialize parameters and build model.
        Params
        ======
            model_cfg: Configuration to build model
        """
        super(ConvModel3D, self).__init__()
        
        nflts = [64, 128, 256, 512]
        self.model_cfg = model_cfg        
        self.conv = nn.Sequential(
            #nn.BatchNorm3d(model_cfg.input_dim[1]),
            nn.Conv3d(model_cfg.input_dim[1], nflts[0], kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(nflts[0]),
            nn.ReLU(),
            nn.Conv3d(nflts[0], nflts[1], kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(nflts[1]),
            nn.ReLU(),
            nn.Conv3d(nflts[1], nflts[2], kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(nflts[2]),
            nn.ReLU(),
            nn.Conv3d(nflts[2], nflts[3], kernel_size=(4, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(nflts[3]),
            nn.ReLU()
            )

        nfcflts = [256, 64]
        self.fc_input_dim = self.feature_size()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, nfcflts[0]),
            nn.BatchNorm1d(nfcflts[0]),
            nn.ReLU(),
            nn.Linear(nfcflts[0], nfcflts[1]),
            nn.ReLU(),
            nn.Linear(nfcflts[1], model_cfg.output_dim)
        )

        if self.model_cfg.dueling_dqn:
            #fc is advantage stream,  fc_value is value stream           
            self.fc_value= nn.Sequential(
                nn.Linear(self.fc_input_dim, nfcflts[0]),
                nn.BatchNorm1d(nfcflts[0]),
                nn.ReLU(),
                nn.Linear(nfcflts[0], nfcflts[1]),
                nn.ReLU(),
                nn.Linear(nfcflts[1], 1)
            )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        if self.model_cfg.dueling_dqn:
            advantage = self.fc(features)
            value = self.fc_value(features)
            qvals = value + (advantage - advantage.mean())
        else:
            qvals = self.fc(features)
        return qvals

    def feature_size(self):
        input = torch.autograd.Variable(torch.zeros(self.model_cfg.input_dim))
        out = self.conv(input)
        return out.view(1, -1).size(1)

class ConvModel2D(nn.Module):
    
    def __init__(self, model_cfg):
        """Initialize parameters and build model.
        Params
        ======
            model_cfg: Configuration to build model
        """
        super(ConvModel2D, self).__init__()
        
        nflts = [256, 512, 512, 512]
        self.model_cfg = model_cfg        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(model_cfg.input_dim[1]),
            nn.Conv2d(model_cfg.input_dim[1], nflts[0], kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(nflts[0]),
            nn.ReLU(),
            nn.Conv2d(nflts[0], nflts[1], kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(nflts[1]),
            nn.ReLU(),
            nn.Conv2d(nflts[1], nflts[2], kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(nflts[2]),
            nn.ReLU(),
            nn.Conv2d(nflts[2], nflts[3], kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(nflts[3]),
            nn.ReLU()
            )

        nfcflts = [64, 64]
        self.fc_input_dim = self.feature_size()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, nfcflts[0]),
            nn.BatchNorm1d(nfcflts[0]),
            nn.ReLU(),
            nn.Linear(nfcflts[0], nfcflts[1]),
            nn.ReLU(),
            nn.Linear(nfcflts[1], model_cfg.output_dim)
        )

        if self.model_cfg.dueling_dqn:
            #fc is advantage stream,  fc_value is value stream           
            self.fc_value= nn.Sequential(
                nn.Linear(self.fc_input_dim, nfcflts[0]),
                nn.BatchNorm1d(nfcflts[0]),
                nn.ReLU(),
                nn.Linear(nfcflts[0], nfcflts[1]),
                nn.ReLU(),
                nn.Linear(nfcflts[1], 1)
            )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        if self.model_cfg.dueling_dqn:
            advantage = self.fc(features)
            value = self.fc_value(features)
            qvals = value + (advantage - advantage.mean())
        else:
            qvals = self.fc(features)
        return qvals

    def feature_size(self):
        input = torch.autograd.Variable(torch.zeros(self.model_cfg.input_dim))
        out = self.conv(input)
        return out.view(1, -1).size(1)

class Model(object):

    def __init__(self, model_cfg, trainable=True):
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

        if model_cfg.conv==ConvType.CONV3D:
            self.net = ConvModel3D(model_cfg)
        elif model_cfg.conv==ConvType.CONV2D:
            self.net = ConvModel2D(model_cfg)
        else:
            self.net = FcModel(model_cfg)

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

    def train(self, inputs, actions, q_targets_next, dones, rewards, gamma):
        if self.trainable==False:
            return

        dones = torch.from_numpy(dones).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from online model
        inputs = torch.from_numpy(inputs).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        q_expected = self.net(inputs).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, input, detach=False, cpu_mem=False):
        input = torch.from_numpy(input).float().to(self.device)
        self.net.eval()
        if detach:
            q_vals = self.net(input).detach()
        else:
            with torch.no_grad():
                q_vals = self.net(input)
        if cpu_mem:
            q_vals = q_vals.cpu().data.numpy()
        self.net.train()
        return q_vals

    def predict_detach(self, state):
        with torch.no_grad():
            q_vals = self.net(state)
        return q_vals

    def get_tensor(self, input):
        input = torch.from_numpy(input).float().to(self.device)
        return input