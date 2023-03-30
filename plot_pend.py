import matplotlib.pyplot as plt
import numpy as np

def plot_reward(episodes, returns):
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(episodes, returns, label='G', linewidth=3)
    ax.set_title('Learning Curve', size=10)
    ax.set_xlabel('Episodes', size=8)
    ax.set_ylabel('Return', size=8)
    ax.grid()
    ax.legend()
    fig.savefig('Plots/learning_curve.png')

def plot_trajectory(num_steps, S, A, R):
    steps = np.arange(0, num_steps, 1)
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(steps, S, label='state', linewidth=3)
    ax.plot(steps, A, label='action', linewidth=3)
    ax.plot(steps, R, label='reward', linewidth=3)
    ax.set_title('Example Trajectory', size=10)
    ax.set_xlabel('Time', size=8)
    ax.set_ylabel('Trajectory', size=8)
    ax.grid()
    ax.legend()
    fig.savefig('Plots/trajectory.png')

def plot_policy(S, A):
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(S, A, label='policy', linewidth=3)
    ax.set_title('Policy Plot', size=10)
    ax.set_xlabel('State', size=8)
    ax.set_ylabel('Action Taken', size=8)
    ax.grid()
    ax.legend()
    fig.savefig('Plots/polciy.png')
