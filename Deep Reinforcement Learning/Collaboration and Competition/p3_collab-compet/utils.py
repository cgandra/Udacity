from unityagents import UnityEnvironment
import os
import time
import pickle
import numpy as np
from collections import deque
from config import AgentConfig, ModelConfig
from agent import MADDPGAgent
import matplotlib.pyplot as plt
from IPython.display import clear_output

def run_episodes(env, agent, num_agents, algo, n_episodes=2000, train=True, moving_avg_tgt=13.0, ckpt_paths=None, replay_paths=None, scores_paths=None):
    """
   
    Params
    ======
        n_episodes (int): maximum number of training episodes
        train (bool): training/eval mode
        moving_avg_tgt (float): avg reward over 100 frames
    """

    brain_name = env.brain_names[0]
    best_score = 0
    i_best_episode = 0
    best_moving_avg = 0
    i_best_ma_episode = 0

    scores = {}
    scores_window = deque(maxlen=100)    # last 100 avg scores
    steps_window = deque(maxlen=100)     # last 100 avg steps
    noise_window = deque(maxlen=100)     # last 100 avg noise
    scores['all'] = []                   # list containing mean scores from each episode
    scores['min'] = []                   # list of min of moving averages
    scores['max'] = []                   # list of max of moving averages
    scores['mavg'] = []                  # list of scores moving averages

    scores['steps'] = []                 # list of steps per episode
    scores['savg'] = []                  # list of steps moving averages
    scores['navg'] = []                  # list of noise moving averages

    for a in range(num_agents):
        nstr = 'noise_' + str(a)
        scores[nstr] = []                # list of average noise per episode
    
    if train:
        disp_str = 'Training'
    else:
        disp_str = 'Testing'
    
    for i_episode in range(agent.start_episode, n_episodes+1):
        env_info = env.reset(train_mode=train)[brain_name]  # reset the environment
        # get the current state
        next_states = env_info.vector_observations

        agent.reset()
        ep_scores = np.zeros(num_agents)
        done = False
        while not done:
            states = next_states
            actions = agent.act(states, add_noise=train)    # select an action
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            # get the next state
            next_states = env_info.vector_observations

            dones = env_info.local_done                     # see if episode has finished
            rewards = env_info.rewards                      # get the reward
            ep_scores += rewards
            # learning step
            done = agent.step_update(states, actions, rewards, next_states, dones, train)

        score_max = np.max(ep_scores)
        scores_window.append(score_max)                     # save most recent score
        steps_window.append(agent.t_step)
        avg_noise = 0
        for a in range(num_agents):
            nstr = 'noise_' + str(a)
            scores[nstr].append(agent.get_avg_noise(a))     # save most recent mean score
            avg_noise += scores[nstr][-1]

        noise_window.append(avg_noise/num_agents)
        scores['all'].append(score_max)                     # save most recent mean score
        #scores['min'].append(np.min(scores_window))         # save most recent min score
        #scores['max'].append(np.max(scores_window))         # save most recent max score
        scores['mavg'].append(np.mean(scores_window))       # save most recent scores moving average
        
        scores['steps'].append(agent.t_step)                # save steps per episode
        scores['savg'].append(np.mean(steps_window))        # save most recent steps moving average
        scores['navg'].append(np.mean(noise_window))        # save most recent noise moving average

        solved = ((scores['mavg'][-1] >= moving_avg_tgt) and (len(scores_window) == scores_window.maxlen))
        if train and ((i_episode % 10 == 0) or solved):
            agent.save(ckpt_paths, i_episode, replay_paths[0])

        if (i_episode % 10 == 0) or solved:
            #display_results(scores, algo, num_agents, disp_str, clr=True)
            f = open(scores_paths[0], 'wb')
            pickle.dump(scores, f)
            f.close()

        valid_best_ma = (best_moving_avg <= scores['mavg'][-1])
        if train and valid_best_ma:
            agent.save(ckpt_paths[num_agents:], i_episode, replay_paths[1])

        if valid_best_ma:
            best_moving_avg = scores['mavg'][-1]
            i_best_ma_episode = i_episode

        valid_best = (best_score <= score_max)
        if train and valid_best:
            agent.save(ckpt_paths[2*num_agents:], i_episode, replay_paths[2])

        if valid_best:
            best_score = score_max
            i_best_episode = i_episode

        print('\rEpisode {}, Average Score: {:.2f}, Best Average Score: {:.2f} @ Episode {}, Best Score: {:.2f} @ Episode {}'.format(i_episode, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode), end="")

        if train and (i_episode % 50 == 0):
            print('\rEpisode {}, Average Score: {:.2f}, Best Average Score: {:.2f} @ Episode {}, Best Score: {:.2f} @ Episode {}'.format(i_episode, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode))

        if train and solved:
            print('\nEnvironment solved in {:d} episodes!, Average Score: {:.2f}, Best Average Score: {:.2f} @ Episode {}, Best Score: {:.2f} @ Episode {}'.format(i_episode-100, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode), flush=True)
            break
        
    return scores

def run_train_test(env, mode, agent_cfg, actor_model_cfg, critic_model_cfg, i, moving_avg_tgt, chkpt_type):
   # Train the agent using DQN
   agent_cfg.ckpt_paths = None
   agent_cfg.replay_path = None
   res_folder = 'results'
   os.makedirs(res_folder, exist_ok=True)

   ckpt_paths = []
   replay_paths=[]
   scores_paths = []
   scores_paths_tst = []
   
   for a in range(agent_cfg.num_agents):
       ckpt_paths.append(res_folder +'/chkpt_agent_' + str(a) + '_' + str(i) +'.pth')
   

   for a in range(agent_cfg.num_agents):
       ckpt_paths.append(res_folder +'/best_ma_chkpt_agent_' + str(a) + '_' + str(i) +'.pth')

   for a in range(agent_cfg.num_agents):
       ckpt_paths.append(res_folder +'/best_chkpt_agent_' + str(a) + '_' + str(i) +'.pth')

   replay_paths.append(None)
   replay_paths.append(None)
   replay_paths.append(None)
   scores_paths.append(res_folder + '/scores_train_' + str(i) + '.pkl')
   scores_paths_tst.append(res_folder + '/scores_test_' + chkpt_type + '_' + str(i) + '.pkl')
   
   start_time = time.time() # Monitor Training Time
   scores = None
   if (mode == 'train') or (mode == 'train_test'):
       agent = MADDPGAgent(agent_cfg, actor_model_cfg, critic_model_cfg)
       scores = run_episodes(env, agent, agent_cfg.num_agents, agent_cfg.algo, n_episodes=agent_cfg.n_episodes, moving_avg_tgt=moving_avg_tgt, ckpt_paths=ckpt_paths, replay_paths=replay_paths, scores_paths=scores_paths)
       print("Total Training time = {:.1f} min for {:}\n".format((time.time()-start_time)/60, agent_cfg.algo), flush=True)

   ## Test the saved agent
   tst_scores = None
   if (mode == 'test') or (mode == 'train_test'):
       chkpt_types = ['last', 'best_ma', 'best']
       try:
           chkpt_idx = chkpt_types.index(chkpt_type)
       except ValueError:
           chkpt_idx = 0
       agent_cfg.ckpt_paths = ckpt_paths[agent_cfg.num_agents*chkpt_idx:]
       agent = MADDPGAgent(agent_cfg, actor_model_cfg, critic_model_cfg, training=False)
       tst_scores = run_episodes(env, agent, agent_cfg.num_agents, agent_cfg.algo, n_episodes=agent_cfg.n_episodes, train=False, scores_paths=scores_paths_tst)

   return scores, tst_scores

def display_results(scores, algo, num_agents, str_type, clr=False):    
    if clr:
        clear_output(True)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)    
    plt.plot(scores['all'], label=str_type+' Scores')
    plt.plot(scores['mavg'], c='r', label='Moving Avg')
    str_t = 'Algo: ' + str(algo) + " Scores"
    plt.title(str_t)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.grid(True)      
    plt.legend(loc='best');

    plt.subplot(132)
    for a in range(num_agents):
        nstr = 'noise_' + str(a)
        plt.plot(scores[nstr], label='Agent_' + str(a))
    plt.plot(scores['navg'], c='r', label='Moving Avg')
    plt.title("Average noise")
    plt.xlabel("Episodes")
    plt.ylabel("Average noise")
    plt.grid()
    plt.legend(loc='best');
    
    plt.subplot(133)
    plt.plot(scores['steps'], label='Episodes')
    plt.plot(scores['savg'], c='r', label='Moving Avg')
    plt.title("Steps")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid()
    plt.legend(loc='best');

    plt.show()
    
def display_all_results(mode, scores, algo, num_agents, moving_avg_tgt, str_type, solved_episodes=None, clr=False):
    if clr:
        clear_output(True)

    n_scores = len(scores)

    fig = plt.figure(figsize=(5*n_scores, 10))
    f_ax = fig.subplots(2, n_scores, squeeze=False)
    for n in range(n_scores):
        f_ax[0, n].plot(scores[n]['all'], label=str_type[n]+' Scores')
        f_ax[0, n].plot(scores[n]['mavg'], c='r', label='Moving Avg')
        if (mode[n] == 'train') and (solved_episodes is not None):
            str_t = 'Episodes: ' + str(solved_episodes[n])
        elif (mode[n] == 'train'):
            str_t = 'Episodes: ' + str(len(scores[n]['all'])-100)
        elif (mode[n] == 'test'):
            str_t = 'Avg Score: ' + "{:.2f}".format(scores[n]['mavg'][-1])
        f_ax[0, n].set_title(str_t)
        f_ax[0, n].set_ylabel('Rewards')
        f_ax[0, n].grid(True)      
        f_ax[0, n].legend(loc='best')

        for a in range(num_agents):
            nstr = 'noise_' + str(a)
            astr = 'Agent '+ str(a)
            f_ax[1, n].plot(scores[n][nstr], label=astr)
        f_ax[1, n].plot(scores[n]['navg'], c='r', label='Moving Avg')
        f_ax[1, n].set_xlabel("Episodes")
        f_ax[1, n].set_ylabel("Noise")
        f_ax[1, n].grid(True)
        f_ax[1, n].legend(loc='best')
    str_t = 'Algo: ' + str(algo) + ', Agents: ' + str(num_agents) + ', Moving Avg Target: ' + str(moving_avg_tgt)
    fig.suptitle(str_t)

    plt.show()

def display_scores(mode, moving_avg_tgt, i, num_agents, res_folder, solved_episodes=None):
    scores = {}
    if (mode == 'train'):
        str_type = 'Training'
        mode = [mode]
    elif (mode == 'test'):
        str_type = 'Eval'
        mode = [mode]
    elif (mode == 'train_test'):
        str_type = ['Training', 'Eval']
        mode = ['train', 'test']

    algo = 'maddpg'
    for n in range(len(mode)):
        scores_path = res_folder + '/scores_'+ mode[n] + '_' + str(i) + '.pkl'
        scores[n]  = pickle.load(open(scores_path, "rb"))
    display_all_results(mode, scores, algo, num_agents, moving_avg_tgt, str_type, solved_episodes)

def run_option(mode, i, moving_avg_tgt=30.0, n_episodes=2000, chkpt_type='best_ma', seed=0):
    env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis", seed=0)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    minibatch_size=256
    agent_cfg = AgentConfig(n_episodes, minibatch_size)
    actor_model_cfg = ModelConfig()
    critic_model_cfg = ModelConfig()
    
    agent_cfg.state_size = env_info.vector_observations.shape[1]
    agent_cfg.action_size = brain.vector_action_space_size
    agent_cfg.num_agents = len(env_info.agents)

    actor_model_cfg.input_dim = agent_cfg.state_size
    actor_model_cfg.output_dim = agent_cfg.action_size
    actor_model_cfg.seed = agent_cfg.seed = seed
    
    if agent_cfg.algo == 'maddpg':
        critic_model_cfg.input_dim = agent_cfg.state_size*agent_cfg.num_agents
        critic_model_cfg.append_dim = agent_cfg.action_size*agent_cfg.num_agents
    else:
        critic_model_cfg.input_dim = agent_cfg.state_size
        critic_model_cfg.append_dim = agent_cfg.action_size
    critic_model_cfg.output_dim = 1
    critic_model_cfg.seed = agent_cfg.seed = seed

    scores, tst_scores = run_train_test(env, mode, agent_cfg, actor_model_cfg, critic_model_cfg, i, moving_avg_tgt, chkpt_type=chkpt_type)
    if (mode == 'train') or (mode == 'train_test'):
        display_results(scores, agent_cfg.algo, agent_cfg.num_agents, 'Training')
    if (mode == 'test') or (mode == 'train_test'):
        display_results(tst_scores, agent_cfg.algo, agent_cfg.num_agents, 'Eval')

    env.close()