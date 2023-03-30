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
    # e = 0.1

    dqn = DQN(BUFFER_SIZE, NUM_EPISODES, EPS_START, EPS_END, EPS_DECAY)
    steps = dqn.train()
    rewards = dqn.rewards[1:]
    plot_pend.plot_reward(np.arange(0, len(rewards), 1), rewards)
    S, A, R, steps = dqn.run()
    # # print('STATES')
    # # print(S)
    # # print('ACTION')
    # # print(A)
    # # print('REWARD')
    # # print(R)
    # print(np.arange(0, NUM_EPISODES, 1))
    plot_pend.plot_trajectory(steps, S, A, R)



if __name__ == '__main__':
    main()