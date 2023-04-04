import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from dqn import DQN
import plot_pend


def main():
    BUFFER_SIZE = 100000
    NUM_EPISODES = 100
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 10000
    # Paths for ablation study
    PATH = 'qnet3.pth'
    PATH_notarget = 'no_target.pth'
    PATH_noreplay = 'no_replay.pth'
    PATH_notargetreplay = 'no_target_replay.pth'

    # Init algorithm
    dqn = DQN(BUFFER_SIZE, NUM_EPISODES, EPS_START, EPS_END, EPS_DECAY, PATH_notargetreplay)

    # train dqn
    steps = dqn.train()

    # record scores and returns
    scores = dqn.score[1:]
    returns = dqn.returns[1:]

    # plot scores and returns
    plot_pend.plot_reward(np.arange(0, len(scores), 1), scores, 'Plots/avg_score2.png')
    plot_pend.plot_reward(np.arange(0, len(returns), 1), returns, 'Plots/learning_curve3.png')

    # run dqn to get video and trajectory
    S, A, R, steps = dqn.run()
    
    # plot trajectory
    plot_pend.plot_trajectory(steps, S, A, R, 'Plots/No_TargetReplay/trajectory2.png')
    
    # get policy and value function for contour plot
    theta_grid, thetadot_grid, policy, val_func = dqn.plotting()
    plot_pend.plot_contour(theta_grid, thetadot_grid, policy, 'Policy Plot', 'Plots/policy2.png')
    plot_pend.plot_contour(theta_grid, thetadot_grid, val_func, 'Value Function', 'Plots/val_func2.png')

    # rew_list = []
    # for i in range(200):
    #     rew_list.append(dqn.get_avg())
    # print(f'average score for no target {np.mean(rew_list)}')





if __name__ == '__main__':
    main()