import numpy as np
import random
from collections import namedtuple, deque
from enums import ConvType

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, augment_size=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.states_q = deque(maxlen=augment_size) 
        self.batch_size = batch_size
        self.augment_size = augment_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def stack(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.long)
        rewards = np.vstack([e.reward for e in experiences if e is not None]).astype(np.float32)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32)
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)
 
    def sample(self, conv):
        """Randomly sample a batch of experiences from memory."""
        if conv==ConvType.CONV1D:
            experiences = random.sample(self.memory, k=self.batch_size)
            experiences= self.stack(experiences)
        else:
            experiences = self.augmented_sample(conv)
        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def augmented_sample(self, conv):
        states_a = []
        actions_a = []
        rewards_a = []
        next_states_a = []
        dones_a = []
        if conv==ConvType.CONV3D:
            axis = 4
            transpose_order = (0, 3, 4, 1, 2)
        else:
            axis = 3
            transpose_order = (0, 3, 1, 2)

        while len(states_a) < self.batch_size:
            ridx = random.sample(range(self.augment_size-1, len(self.memory)), 1)[0]

            states_tmp = np.concatenate([self.memory[idx].state for idx in range(ridx, ridx-self.augment_size, -1)], axis=axis)
            next_states_tmp = np.concatenate([self.memory[idx].next_state for idx in range(ridx, ridx-self.augment_size, -1)], axis=axis)
            states_tmp = np.transpose(states_tmp, transpose_order)
            next_states_tmp = np.transpose(next_states_tmp, transpose_order)

            states_a.append(states_tmp)
            actions_a.append(self.memory[ridx].action)
            rewards_a.append(self.memory[ridx].reward)
            next_states_a.append(next_states_tmp)
            dones_a.append(self.memory[ridx].done)

        states_a = np.vstack(states_a).astype(np.float32)
        actions_a = np.vstack(actions_a).astype(np.long)
        rewards_a = np.vstack(rewards_a).astype(np.float32)
        next_states_a = np.vstack(next_states_a).astype(np.float32)
        dones_a = np.vstack(dones_a).astype(np.uint8)

        return (states_a, actions_a, rewards_a, next_states_a, dones_a)

    def augment_state(self, state, conv):
        if len(self.states_q) < 1 :
            init_state = np.zeros((state.shape))
            for i in range(self.states_q.maxlen-1) :
                self.states_q.append(init_state)
                
        self.states_q.appendleft(state)
        states_a = list(self.states_q)
        if conv==ConvType.CONV3D:
            axis = 4
            transpose_order = (0, 3, 4, 1, 2)
        else:
            axis = 3
            transpose_order = (0, 3, 1, 2)

        states_a = np.concatenate(states_a, axis=axis)
        states_a = np.transpose(states_a, transpose_order)
        return states_a