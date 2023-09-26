from enums import ConvType

class ModelConfig( object ) :

    def __init__( self ) :
        super( ModelConfig, self ).__init__()

        # training params
        self.learning_rate = 1e-4       # learning rate
        self.weight_decay = 0           # weight decay (L2 penalty)

        # model types/features
        self.double_dqn = False
        self.dueling_dqn = False
        self.conv = ConvType.CONV1D     # convolutional or fc only model 

        # dimensions
        self.input_dim = 1
        self.output_dim = 1

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
        self.state_augment_size = 0
        self.input_gray = False

        self.n_episodes = max_episodes
        self.max_t_steps = 1000
        
        # replay buffer size
        self.replay_buf_size = int(1e5)
        self.min_replay_buf_size = 0 # replay start size

        # training params
        self.set_minibatch_size(minibatch_size)  # minibatch size
        self.training_freq = 4       # how often to update the network
        self.update_tgt_freq = 4     # how often to update the target network
        self.tau = 1e-3              # interpolation factor between actor & target network parameters for soft update

        # discount factor
        self.gamma = 0.99

        # epsilon update params
        self.eps_schedule = 'geometric'
        self.set_epsilon()

        # random seed
        self.seed = 0

        self.ckpt_path = None
        self.replay_path = None

    def set_min_replay_buf_size(self, mrb_size):
        self.min_replay_buf_size = max(self.minibatch_size, mrb_size)

    def set_minibatch_size(self, mb_size):
        self.minibatch_size = mb_size
        self.min_replay_buf_size = max(self.minibatch_size, self.min_replay_buf_size)

    def set_epsilon_decay(self, eps_decay):
        self.eps_decay = eps_decay

    def set_epsilon(self, eps_start=1.0, eps_end=0.01, eps_decay=0.995, eps_schedule='geometric'):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_schedule = eps_schedule
        if eps_schedule == 'linear':
            self.eps_decay = (self.eps_start-self.eps_end)/(self.n_episodes*self.max_t_steps/4)
        else:
            self.eps_decay = eps_decay

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)