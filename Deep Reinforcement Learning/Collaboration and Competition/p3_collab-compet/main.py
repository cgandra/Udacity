import os
import time
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
            str_t = 'Episodes: ' + str(len(scores[n]['all'])-100)
        elif (mode[n] == 'test'):
            str_t = 'Avg Score: ' + "{:.2f}".format(scores[n]['mavg'][-1])
        f_ax[0, n].set_title(str_t)
        f_ax[0, n].set_ylabel('Rewards')
        f_ax[0, n].grid(True)      
        f_ax[0, n].legend(loc='best');

        for a in range(num_agents):
            nstr = 'noise_' + str(a)
            astr = 'Agent '+ str(a)
            f_ax[1, n].plot(scores[n][nstr], label=astr)
        f_ax[1, n].plot(scores[n]['navg'], c='r', label='Moving Avg')
        f_ax[1, n].set_xlabel("Episodes")
        f_ax[1, n].set_ylabel("Noise")
        f_ax[1, n].grid(True)
        f_ax[1, n].legend(loc='best')
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

if __name__ == '__main__':
    display_scores('train_test', 1.8, 2, 2, 'results', [2210])