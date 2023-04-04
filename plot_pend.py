import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def plot_reward(episodes, returns, filename):
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(episodes, returns, label='G', linewidth=3)
    ax.set_title('Learning Curve', size=10)
    ax.set_xlabel('Episodes', size=8)
    ax.set_ylabel('Return', size=8)
    ax.grid()
    ax.legend()
    fig.savefig(filename)

def plot_trajectory(num_steps, S, A, R, filename):
    theta = []
    thetadot = []
    for i in range(len(S)):
        theta.append(S[i][0])
        thetadot.append(S[i][1])
    steps = np.arange(0, num_steps+1, 1)
    fig, (s, a, r) =  plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    s.plot(steps, theta, color='orange', label='theta', linewidth=3)
    s.plot(steps, thetadot, '--', color='blue', label='theta', linewidth=3)
    s.set_title('Example Trajectory', size=20) 
    s.set_ylabel('State', size=20)
    s.legend()
    a.plot(steps, A, label='action', linewidth=3)
    a.set_ylabel('Action', size=20)
    a.legend()
    r.plot(steps, R, label='reward', linewidth=3)
    r.set_ylabel('Reward', size=20)
    r.legend()
    r.set_xlabel('Time Step', size=20)
    fig.savefig(filename)

def plot_contour(theta, thetadot, pol_or_val, title, filename):
    fig, (ax0) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    c = ax0.contourf(theta, thetadot, pol_or_val)
    fig.colorbar(c)
    ax0.set_title(title, size=10)
    ax0.set_xlabel('Theta', size=8)
    ax0.set_ylabel('ThetaDot', size=8)
    ax0.grid()
    ax0.legend()
    fig.savefig(filename)


