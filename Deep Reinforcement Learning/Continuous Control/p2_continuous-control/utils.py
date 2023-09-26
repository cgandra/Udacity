from unityagents import UnityEnvironment
import os
import time
import pickle
import numpy as np
from collections import deque
from config import AgentConfig, ModelConfig
from agent import Agent
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
    win_len = 100
    scores['all'] = []                   # list containing mean scores from each episode
    scores['min'] = []                   # list of min of moving averages
    scores['max'] = []                   # list of max of moving averages
    scores['mavg'] = []                  # list of scores moving averages

    scores['steps'] = []                 # list of steps per episode
    scores['savg'] = []                  # list of steps moving averages
    scores['noise'] = []                 # list of average noise per episode	
    scores['navg'] = []                  # list of noise moving averages
    
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

        score_mean = np.mean(ep_scores)
        avg_noise = agent.avg_noise/agent.t_step
        scores['all'].append(score_mean)                            # save most recent mean score
        scores['noise'].append(avg_noise)                           # save most recent mean noise
        scores['steps'].append(agent.t_step)                        # save steps per episode

        scores['mavg'].append(np.mean(scores['all'][-win_len:]))    # save most recent scores moving average
        
        scores['savg'].append(np.mean(scores['steps'][-win_len:]))  # save most recent steps moving average
        scores['navg'].append(np.mean(scores['noise'][-win_len:]))  # save most recent noise moving average

        solved = ((scores['mavg'][-1] >= moving_avg_tgt) and (len(scores['mavg']) >= win_len))
        if train and ((i_episode % 10 == 0) or solved):
            agent.save(ckpt_paths[0], i_episode, replay_paths[0])

        if (i_episode % 10 == 0) or solved:
            f = open(scores_paths[0], 'wb')
            pickle.dump(scores, f)
            f.close()

        valid_best_ma = (best_moving_avg <= scores['mavg'][-1])
        if train and valid_best_ma:
            agent.save(ckpt_paths[1], i_episode, replay_paths[1])

        if valid_best_ma:
            best_moving_avg = scores['mavg'][-1]
            i_best_ma_episode = i_episode

        valid_best = (best_score <= score_mean)
        if train and valid_best:
            agent.save(ckpt_paths[2], i_episode, replay_paths[2])

        if valid_best:
            best_score = score_mean
            i_best_episode = i_episode

        print('\rEpisode {:4d}, Average Score: {:5.2f}, Best Average Score: {:5.2f} @ Episode {:4d}, Best Score: {:5.2f} @ Episode {:4d}'.format(i_episode, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode), end="")

        if train and (i_episode % 50 == 0):
            print('\rEpisode {:4d}, Average Score: {:5.2f}, Best Average Score: {:5.2f} @ Episode {:4d}, Best Score: {:5.2f} @ Episode {:4d}'.format(i_episode, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode))

        if train and solved:
            print('\nEnvironment solved in {:d} episodes!, Average Score: {:5.2f}, Best Average Score: {:5.2f} @ Episode {:4d}, Best Score: {:5.2f} @ Episode {:4d}'.format(i_episode-100, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode), flush=True)
            break

    return scores

def run_train_test(env, mode, agent_cfg, actor_model_cfg, critic_model_cfg, algo, i, moving_avg_tgt, chkpt_type):
   # Train the agent using DQN
   agent_cfg.ckpt_path = None
   agent_cfg.replay_path = None
   if agent_cfg.num_agents > 1:
       res_folder = 'results_20'
   else:
       res_folder = 'results_1'
   os.makedirs(res_folder, exist_ok=True)

   ckpt_paths = []
   replay_paths=[]
   scores_paths = []
   scores_paths_tst = []
   ckpt_paths.append(res_folder + '/chkpt_' + str(i) +'.pth')
   ckpt_paths.append(res_folder + '/best_ma_chkpt_' + str(i) +'.pth')
   ckpt_paths.append(res_folder + '/best_chkpt_' + str(i) +'.pth')
   #replay_paths.append(res_folder + '/replay_' + str(i) +'.pkl')
   replay_paths.append(None)
   replay_paths.append(None)
   replay_paths.append(None)
   scores_paths.append(res_folder + '/scores_train_' + str(i) + '.pkl')
   scores_paths_tst.append(res_folder + '/scores_test_' + str(i) + '.pkl')
   
   start_time = time.time() # Monitor Training Time
   scores = None
   if (mode == 'train') or (mode == 'train_test'):
       agent = Agent(agent_cfg, actor_model_cfg, critic_model_cfg)
       scores = run_episodes(env, agent, agent_cfg.num_agents, algo, n_episodes=agent_cfg.n_episodes, moving_avg_tgt=moving_avg_tgt, ckpt_paths=ckpt_paths, replay_paths=replay_paths, scores_paths=scores_paths)
       print("Total Training time = {:.1f} min for {:}\n".format((time.time()-start_time)/60, algo), flush=True)

   ## Test the saved agent
   tst_scores = None
   if (mode == 'test') or (mode == 'train_test'):
       chkpt_types = ['last', 'best_ma', 'best']
       try:
           chkpt_idx = chkpt_types.index(chkpt_type)
       except ValueError:
           chkpt_idx = 0
       agent_cfg.ckpt_path = ckpt_paths[chkpt_idx]
       agent = Agent(agent_cfg, actor_model_cfg, critic_model_cfg, training=False)
       tst_scores = run_episodes(env, agent, agent_cfg.num_agents, algo, n_episodes=agent_cfg.n_episodes, train=False, scores_paths=scores_paths_tst)

   return scores, tst_scores

def display_results(scores, algo, clr=False):    
    if clr:
        clear_output(True)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)    
    plt.plot(scores['all'], label='Episodes')
    plt.plot(scores['mavg'], c='r', label='Moving Avg')
    #plt.plot(scores['min'], c='k', label='Moving Min')
    #plt.plot(scores['max'], c='k', label='Moving Max')
    str_t = 'Algo: ' + str(algo) + " Scores"
    plt.title(str_t)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.grid(True)      
    plt.legend(loc='best');

    plt.subplot(132)
    plt.plot(scores['noise'], label='Episodes')
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
    
def display_all_results(mode, scores, algos, num_agents, moving_avg_tgt, str_type, solved_episodes=None, clr=False):
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
            str_t = 'Episodes: ' + str(len(scores[n]['all']))
        elif (mode[n] == 'test'):
            str_t = 'Avg Score: ' + str(scores[n]['mavg'][-1])
        f_ax[0, n].set_title(str_t)
        f_ax[0, n].set_ylabel('Rewards')
        f_ax[0, n].grid(True)      
        f_ax[0, n].legend(loc='best');

        f_ax[1, n].plot(scores[n]['noise'])
        f_ax[1, n].plot(scores[n]['navg'], c='r', label='Moving Avg')
        f_ax[1, n].set_xlabel("Episodes")
        f_ax[1, n].set_ylabel("Noise")
        f_ax[1, n].grid()
    str_t = 'Algo: ' + str(algos[0]) + ', Agents: ' + str(num_agents) + ', Moving Avg Target: ' + str(moving_avg_tgt)
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

    algos = ['ddpg']
    for n in range(len(mode)):
        scores_path = res_folder + '/scores_'+ mode[n] + '_' + str(i) + '.pkl'
        scores[n]  = pickle.load(open(scores_path, "rb"))
    display_all_results(mode, scores, algos, num_agents, moving_avg_tgt, str_type, solved_episodes)

def run_option(mode, i, agent_type='single', moving_avg_tgt=30.0, n_episodes=2000, chkpt_type='best_ma', seed=0):
    if agent_type == 'multi':
        env = UnityEnvironment(file_name="/data/Reacher_Linux_NoVis/Reacher.x86_64", seed=0)
    else:
        env = UnityEnvironment(file_name="/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64", seed=0)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    minibatch_size=128
    agent_cfg = AgentConfig(n_episodes, minibatch_size)
    actor_model_cfg = ModelConfig()
    critic_model_cfg = ModelConfig()
    
    agent_cfg.state_size = env_info.vector_observations.shape[1]
    agent_cfg.action_size = brain.vector_action_space_size
    agent_cfg.num_agents = len(env_info.agents)
    agent_cfg.set_min_replay_buf_size(256)

    actor_model_cfg.input_dim = agent_cfg.state_size
    actor_model_cfg.output_dim = agent_cfg.action_size
    actor_model_cfg.seed = agent_cfg.seed = seed
    
    critic_model_cfg.input_dim = agent_cfg.state_size
    critic_model_cfg.append_dim = agent_cfg.action_size
    critic_model_cfg.output_dim = 1
    critic_model_cfg.seed = agent_cfg.seed = seed

    algo = 'ddpg'

    scores, tst_scores = run_train_test(env, mode, agent_cfg, actor_model_cfg, critic_model_cfg, algo, i, moving_avg_tgt, chkpt_type=chkpt_type)
    if scores is not None:
        display_results(scores, algo)
    if tst_scores is not None:
        display_results(tst_scores, algo)

    env.close()