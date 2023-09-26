import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(x.shape)
        self.state = x + dx
        return self.state

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class GaussianNoise:
    def __init__(self, size, seed, mu=0., sigma=0.2):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        np.random.seed(seed)

    def reset(self):
        pass

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.size)
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    