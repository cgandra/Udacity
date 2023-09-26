class ModelConfig( object ) :

    def __init__( self ) :
        super( ModelConfig, self ).__init__()

        # training params
        self.learning_rate = 1e-4    # learning rate
        self.weight_decay = 0        # weight decay (L2 penalty)

        # dimensions
        self.input_dim = 1
        self.output_dim = 1
        self.append_dim = 0

        # number of hidden filters
        self.nflts = [256, 128]

        # random seed
        self.seed = 0

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class AgentConfig( object ) :

    def __init__( self, max_episodes=2000, minibatch_size=64 ) :
        super( AgentConfig, self ).__init__()

        # environment state and action info
        self.state_size = 1
        self.action_size = 1
        self.num_agents = 1

        self.n_episodes = max_episodes
        self.max_t_steps = 2000
        
        # replay buffer size
        self.replay_buf_size = int(1e6)
        self.min_replay_buf_size = 0 # replay start size

        # training params
        self.set_minibatch_size(minibatch_size)  # minibatch size
        self.training_freq = 1       # how often to update the network
        self.update_tgt_freq = 2     # how often to update the target network
        self.tau = 1e-3              # interpolation factor between actor & target network parameters for soft update

        # discount factor
        self.gamma = 0.99

        # random seed
        self.seed = 0

        self.ckpt_path = None
        self.replay_path = None

    def set_min_replay_buf_size(self, mrb_size):
        self.min_replay_buf_size = max(self.minibatch_size, mrb_size)

    def set_minibatch_size(self, mb_size):
        self.minibatch_size = mb_size
        self.min_replay_buf_size = max(self.minibatch_size, self.min_replay_buf_size)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)