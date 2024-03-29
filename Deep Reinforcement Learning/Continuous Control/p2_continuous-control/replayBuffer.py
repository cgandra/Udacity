import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def stack(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32)
        rewards = np.vstack([e.reward for e in experiences if e is not None]).astype(np.float32)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32)
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)
 
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return self.stack(experiences)  

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)