from unityagents import UnityEnvironment
import os
import time
import pickle
import numpy as np
from collections import deque
from enums import ConvType
from config import AgentConfig, ModelConfig
from agent import Agent
import matplotlib.pyplot as plt

def pre_process(state, conv, input_gray=False) :
    if input_gray:
        state = 0.299 * state[0,:,:,0] + 0.587 * state[0,:,:,1] + 0.114 * state[0,:,:,2]
        state = np.expand_dims(state, 0)
        state = np.expand_dims(state, 3)

    if conv==ConvType.CONV3D:
        state = np.expand_dims(state, 4)
    return state

def run_episodes(env, agent, n_episodes, conv, input_gray, train=True, moving_avg_tgt=13.0, ckpt_paths=None, replay_paths=None, scores_paths=None):
    """Deep Q-Learning.
   
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
    scores['all'] = []                   # list containing mean scores from each episode
    scores['eps'] = []                   # list containing eps from each episode
    scores['mavg'] = []                  # list of moving averages
    scores['steps'] = []                 # list of steps per episode

    for i_episode in range(agent.start_episode, n_episodes+1):
        env_info = env.reset(train_mode=train)[brain_name]  # reset the environment
        # get the current state
        if conv != ConvType.CONV1D:
            next_state = pre_process(env_info.visual_observations[0], conv, input_gray=input_gray)
        else:
            next_state = np.expand_dims(env_info.vector_observations[0], 0)

        score = 0
        done = False
        while not done:
            state = next_state
            state_a = agent.augment_state(state)
            action = agent.act(state_a)                # select an action
            env_info = env.step(action)[brain_name]     # send the action to the environment
            # get the next state
            if conv != ConvType.CONV1D:
                next_state = pre_process(env_info.visual_observations[0], conv, input_gray=input_gray)
            else:
                next_state = np.expand_dims(env_info.vector_observations[0], 0)

            done = env_info.local_done[0]       # see if episode has finished
            reward = env_info.rewards[0]        # get the reward
            score += reward
            # learning step
            done = agent.step_update(state, action, reward, next_state, done, train)

        scores_window.append(score)                     # save most recent score

        scores['all'].append(score)                     # save most recent score
        scores['eps'].append(agent.eps)                 # save most recent epsilon
        scores['mavg'].append(np.mean(scores_window))   # save moving average
        scores['steps'].append(agent.t_step)            # save steps per episode

        solved = ((scores['mavg'][-1] >= moving_avg_tgt) and (scores['all'][-1] >= moving_avg_tgt) and (len(scores_window) == scores_window.maxlen))
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

        valid_best = (best_score <= score)
        if train and valid_best:
            agent.save(ckpt_paths[2], i_episode, replay_paths[2])

        if valid_best:
            best_score = score
            i_best_episode = i_episode

        print('\rEpisode {:4d}, Average Score: {:5.2f}, Best Average Score: {:5.2f} @ Episode {:4d}, Best Score: {:2.0f} @ Episode {:4d}'.format(i_episode, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode), end="")

        if train and (i_episode % 50 == 0):
            print('\rEpisode {:4d}, Average Score: {:5.2f}, Best Average Score: {:5.2f} @ Episode {:4d}, Best Score: {:2.0f} @ Episode {:4d}'.format(i_episode, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode))

        if train and solved:
            print('\nEnvironment solved in {:d} episodes!, Average Score: {:5.2f}, Best Average Score: {:5.2f} @ Episode {:4d}, Best Score: {:2.0f} @ Episode {:4d}'.format(i_episode-100, scores['mavg'][-1], best_moving_avg, i_best_ma_episode, best_score, i_best_episode), flush=True)
            break

    return scores

def get_algo_names(a_double, a_duel):
    algos = []
    for i in range(len(a_double)):
        astr = ''
        if a_duel[i]:
            astr = astr + 'duel_'
        if a_double[i]:
            astr = astr + 'd'
        
        astr = astr + 'dqn'
        algos.append(astr)

    return algos


def run_train_test(env, mode, agent_cfg, model_cfg, algo, e_decay, i, moving_avg_tgt, chkpt_type):
   # Train the agent using DQN
   agent_cfg.ckpt_path = None
   agent_cfg.replay_path = None
   if model_cfg.conv != ConvType.CONV1D:
       res_folder = 'results_pixels'
   else:
       res_folder = 'results'
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
       agent = Agent(agent_cfg, model_cfg)
       scores = run_episodes(env, agent, agent_cfg.n_episodes, model_cfg.conv, agent_cfg.input_gray, moving_avg_tgt=moving_avg_tgt, ckpt_paths=ckpt_paths, replay_paths=replay_paths, scores_paths=scores_paths)
       print("Total Training time = {:.1f} min for {:}, e_decay {:.3f}\n".format((time.time()-start_time)/60, algo, e_decay), flush=True)

   ## Test the saved agent
   tst_scores = None
   if (mode == 'test') or (mode == 'train_test'):
       chkpt_types = ['last', 'best_ma', 'best']
       try:
           chkpt_idx = chkpt_types.index(chkpt_type)
       except ValueError:
           chkpt_idx = 0
       agent_cfg.ckpt_path = ckpt_paths[chkpt_idx]
       agent = Agent(agent_cfg, model_cfg, training=False)
       tst_scores = run_episodes(env, agent, agent_cfg.n_episodes, model_cfg.conv, agent_cfg.input_gray, train=False, scores_paths=scores_paths_tst)
   return scores, tst_scores

def display_results(scores, algo, e_decay, str_type, clr=False):
    if clr:
        clear_output(True)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)    
    plt.plot(scores['all'], label=str_type+' Scores')
    plt.plot(scores['mavg'], c='r', label='Moving Avg')
    str_t = 'Algo: ' + str(algo) + ', e_decay: '  + str(e_decay)
    plt.title(str_t)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.grid(True)      
    plt.legend(loc='best');

    plt.subplot(132)
    plt.plot(scores['eps'])
    plt.title("Epsilon scheduling")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.grid()

    plt.subplot(133)
    plt.plot(scores['steps'])
    str_t = str_type + " Steps"
    plt.title(str_t)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid()

    plt.show()
    
def display_all_results(mode, scores, algo, e_decay, moving_avg_tgt, str_type, solved_episodes=None, clr=False):
    if clr:
        clear_output(True)

    n_scores = len(algo)

    fig = plt.figure(figsize=(5*n_scores, 10))
    f_ax = fig.subplots(2, n_scores, squeeze=False)
    for n in range(n_scores):
        f_ax[0, n].plot(scores[n]['all'], label=str_type+' Scores')
        f_ax[0, n].plot(scores[n]['mavg'], c='r', label='Moving Avg')
        str_t = 'Algo: ' + str(algo[n])
        if (mode == 'train') and (solved_episodes is not None):
            str_t = str_t + ', Episodes: ' + str(solved_episodes[n])
        elif (mode == 'train'):
            str_t = str_t + ', Episodes: ' + str(len(scores[n]['all']))
        elif (mode == 'test'):
            str_t = str_t + ', Avg Score: ' + str(scores[n]['mavg'][-1])
        f_ax[0, n].set_title(str_t)
        f_ax[0, n].set_ylabel('Rewards')
        f_ax[0, n].grid(True)      
        f_ax[0, n].legend(loc='best');

        f_ax[1, n].plot(scores[n]['eps'])
        f_ax[1, n].set_xlabel("Episodes")
        f_ax[1, n].set_ylabel("Epsilon")
        f_ax[1, n].grid()
    str_t = 'Moving Avg Target: ' + str(moving_avg_tgt) + ', Epsilon decay: '  + str(e_decay)
    fig.suptitle(str_t)

    plt.show()

def display_scores(mode, eps_decay, moving_avg_tgt, a_double, a_duel, i, res_folder, solved_episodes=None):
    scores = {}
    if (mode == 'train'):
        str_type = 'Training'
    elif (mode == 'test'):
        str_type = 'Eval'

    algos = get_algo_names(a_double, a_duel)
    for a in range(len(a_double)):
        scores_path = res_folder + '/scores_'+ mode + '_' + str(i) + '.pkl'
        scores[a]  = pickle.load(open(scores_path, "rb"))
        i = i+1
    display_all_results(mode, scores, algos, eps_decay, moving_avg_tgt, str_type, solved_episodes)

def run_option(mode, eps_decay, a_double, a_duel, i, moving_avg_tgt=25.0, n_episodes=2000, conv=ConvType.CONV1D, input_gray=False, chkpt_type='best', seed=0):
    if conv==ConvType.CONV1D:
        env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64", seed=seed)
    else:
        env = UnityEnvironment(file_name="VisualBanana_Linux_NoVis/Banana.x86_64", seed=seed)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space
    if conv==ConvType.CONV1D:
        augment_size = 0
        state = env_info.vector_observations[0]
        state_size = len(state)
    else:
        augment_size=4
        state = env_info.visual_observations[0]
   
        if conv==ConvType.CONV3D:
            state = np.expand_dims(state, 4)
            axis = 4
            transpose_order = (0, 3, 4, 1, 2)
        else:
            axis = 3
            transpose_order = (0, 3, 1, 2)

        state = np.repeat(state, augment_size, axis=axis)
        state = np.transpose(state, transpose_order)
        state_size = state.shape
        print('States have shape:', state_size)

    minibatch_size=64
    agent_cfg = AgentConfig(n_episodes, minibatch_size)
    model_cfg = ModelConfig()

    agent_cfg.input_gray = input_gray
    agent_cfg.state_augment_size = augment_size
    agent_cfg.state_size = state_size
    agent_cfg.action_size = action_size
    #agent_cfg.set_min_replay_buf_size(256)
    agent_cfg.set_epsilon(eps_end=0.01, eps_schedule='geometric')

    model_cfg.input_dim = state_size
    model_cfg.output_dim = action_size
    model_cfg.conv = conv
    model_cfg.seed = agent_cfg.seed = seed
    
    algos = get_algo_names(a_double, a_duel)

    for a in range(len(a_double)):
        for e_decay in eps_decay:
            agent_cfg.set_epsilon_decay(e_decay)
            model_cfg.double_dqn = a_double[a]
            model_cfg.dueling_dqn = a_duel[a]
            scores, tst_scores = run_train_test(env, mode, agent_cfg, model_cfg, algos[a], e_decay, i, moving_avg_tgt, chkpt_type=chkpt_type)
            if scores is not None:
                display_results(scores, algos[a], e_decay, 'Training')
            if tst_scores is not None:
                display_results(tst_scores, algos[a], e_decay, 'Eval')
            i = i + 1
    env.close()