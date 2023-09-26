import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def stack(self, experiences):
        
        states, actions, rewards, next_states, dones = map(lambda x: np.asarray(x), zip(*experiences))

        states = states.astype(np.float32)
        actions = actions.astype(np.float32)
        rewards = np.transpose(rewards, (1, 0)).astype(np.float32)
        next_states = next_states.astype(np.float32)
        dones = np.transpose(dones, (1, 0)).astype(np.uint8)
  
        return (states, actions, rewards, next_states, dones)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(len(self.memory), self.batch_size))
        return self.stack(experiences)  

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)