import torch 
import random
import numpy as np
import itertools
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple, deque
from discreteaction_pendulum import Pendulum

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

# Network Parameters and Learning Parameters
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
# LR = 0.00025
MIN_BUFFER_SIZE = 1000

# steps_done = 0

# Init a named tuple for pulling from the buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class Qnet(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Qnet, self).__init__()
        # Define network
        self.layer1 = nn.Linear(num_states, 64)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(64, 64)
        self.relu = nn.Tanh()
        self.layer3 = nn.Linear(64, num_actions)

        for param in self.parameters():
            param.requires_grad = True


    def forward(self, x):
        # Forward function
        out = self.layer1(x)
        out = self.tanh(out)
        out = self.layer2(out)
        out = self.relu(out)
        return self.layer3(out)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action
    

class Buffer(object):
    def __init__(self, capacity):
        # Init a buffer to max capacity
        self.buffer = deque([], maxlen=capacity)

    def add(self, *args):
        # Add transition to the buffer
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        # Get a random sample from the replay buffer
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

class DQN():
    def __init__(self, capacity, num_episodes, e_start, e_end, e_decay):

        # Init env 
        self.env = Pendulum()

        # Init global vars
        self.e_start = e_start
        self.e_end = e_end
        self.e_decay = e_decay
        self.num_episodes = num_episodes

        # Init main model (trained every step) and target model (what is predicted against)
        self.Q = Qnet(self.env.num_states, self.env.num_actions)
        self.target_Q = Qnet(self.env.num_states, self.env.num_actions)
        self.target_Q.load_state_dict(self.Q.state_dict())

        # Init optimizer 
        self.optimizer = optim.Adam(self.Q.parameters(), lr=LR) 

        # Init replay buffer to capacity size N 
        self.buffer = Buffer(capacity)

        # Lists to store reward
        self.rewards = []
        

    def fill_buffer(self, min_buffer_size):
        s = self.env.reset()
        for _ in range(min_buffer_size):
            a = random.randrange(self.env.num_actions)

            s1, r, done = self.env.step(a)
            self.buffer.add(s, a, s1, r, done)

            if done:
                s = self.env.reset()


    def get_action(self, s, step):
        # Get random action if under epsilon otherwise take from action value
        e = np.interp(step, [0, self.e_decay], [self.e_start, self.e_end])
        sample = random.random()
        if sample <= e:
            a = random.randrange(self.env.num_actions)
            return a
        else:
            a = self.Q.act(s)
            return a 


    def get_return(self, reward_list):
        Return = reward_list.pop(0)
        for i in range(len(reward_list)):
            Return += (GAMMA**i) * reward_list[i]
        return Return
                

    def optimize(self, step):
        transitions = self.buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # make training batches        
        state_batch = np.array(batch.state)
        action_batch = np.asarray(batch.action)
        next_state_batch = np.asarray(batch.next_state)
        reward_batch = np.asarray(batch.reward)
        dones_batch = np.asarray(batch.done)

        state_batch_t = torch.as_tensor(state_batch, dtype=torch.float32)
        action_batch_t = torch.as_tensor(action_batch, dtype=torch.int64).unsqueeze(-1)
        next_state_batch_t = torch.as_tensor(next_state_batch, dtype=torch.float32)
        reward_batch_t = torch.as_tensor(reward_batch, dtype=torch.float32).unsqueeze(-1)
        dones_batch_t = torch.as_tensor(dones_batch, dtype=torch.float32).unsqueeze(-1)
        
        
        # compute targets
        target_q_vals = self.target_Q(next_state_batch_t)
        max_target_q_vals = target_q_vals.max(dim=1, keepdim=True)[0]
        targets = reward_batch_t + GAMMA * (1 - dones_batch_t) * max_target_q_vals

        # get action q vals
        q_vals = self.Q(state_batch_t)
        action_q_vals = torch.gather(input=q_vals, dim=1, index=action_batch_t)

        criterion = nn.SmoothL1Loss()
        loss = criterion(action_q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target net
        if step % 100 == 0:
            self.target_Q.load_state_dict(self.Q.state_dict())

        if step % 100 == 0:
            print()
            print('Step:', step)
            print('Avg Reward:', np.mean(self.rewards))

        

    def train(self):
        self.fill_buffer(MIN_BUFFER_SIZE)

        s = self.env.reset()
        epi_reward = 0.0
        for step in itertools.count():
            a = self.get_action(s, step)

            s1, r, done = self.env.step(a)
            self.buffer.add( s, a, s1, r, done)
            s = s1

            epi_reward += r

            if done:
                s = self.env.reset()
                self.rewards.append(r)
                epi_reward = 0.0

            self.optimize(step)

            if step == 100000:
                break
        
        PATH = 'qnet.pth'
        torch.save(self.Q, PATH)

        return step

    
    def run(self, filename='pendulum.gif', writer='imagemagick'):
        PATH = 'qnet.pth'
        Q = torch.load(PATH)
        S = []
        A = []
        R = []
        s = self.env.reset()
        S.append(s)
        A.append(0)
        R.append(0)
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        dt = 0.1
        s_traj = [s]
        done = False
        steps = 0
        while not done:
            a_ = Q(s)
            a = torch.argmax(a_, dim=1)[0].detach().item()
            A.append(a)
            (s, r, done) = self.env.step(a)
            s_traj.append(s)
            S.append(s)
            R.append(r)
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            steps += 1

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set_aspect('equal')
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        text = ax.set_title('')

        def animate(i):
            theta = s_traj[i][0]
            line.set_data([0, -np.sin(theta)], [0, np.cos(theta)])
            text.set_text(f'time = {i * dt:3.1f}')
            return line, text

        anim = animation.FuncAnimation(fig, animate, len(s_traj), interval=(1000 * dt), blit=True, repeat=False)
        anim.save(filename, writer=writer, fps=10)

        plt.close()

        return S, A, R, steps

       


   