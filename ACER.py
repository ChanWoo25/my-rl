# Implementation of ACER Algorithm
# in paper "Sample Efficient Actor-Critic with Experience Replay."
# https://arxiv.org/abs/1611.01224
import gym
from numpy import ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

import random
import collections

lr              = 0.00015   # Learning rate
gamma           = 0.98      # Discount factor
bf_lim          = 2000      # Limit of Buffer
roll            = 10        # Rollout length
batch_size      = 5         # 4 Sequences per mini-batch (4 * rollout_len = 40 samples total)
c               = 10.0      # For truncating importance sampling ratio

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=bf_lim)

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def sample(self, on_policy=False):
        """
            return torch.Tensor Classes that have following informations
            return : state, action, reward, prob, done_mask, is_first
        """
        if on_policy:
            mini_batch = [self.buffer[-1]] # Latest one
        else:
            # randomly choice batch_size's sampels from deque
            mini_batch = random.sample(self.buffer, batch_size)

        state_list = []
        action_list = []
        reward_list = []
        prob_list = []
        done_list = []
        is_first_list = []

        for seq in mini_batch:
            is_first = True
            for transition in seq:
                # State, Action, Reward, Probability, Done
                s, a, r, p, d = transition
                state_list.append(s)
                action_list.append([a])
                reward_list.append(r)
                prob_list.append(p)
                done_list.append(0.0 if d else 1.0)
                is_first_list.append(is_first)
                is_first = False

        state, action, reward, prob, done_mask, is_first = \
            torch.tensor(state_list, dtype=torch.float), \
            torch.tensor(action_list), \
            reward_list, \
            torch.tensor(prob_list, dtype=torch.float),\
            done_list,\
            is_first_list

        return state, action, reward, prob, done_mask, is_first

    def size(self):
        return len(self.buffer)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        pi = F.softmax(x, dim=softmax_dim)
        return pi

    def q(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc_q(x)
        return q


def train(model, optimizer, memory, on_policy=False):
    s, a, r, prob, done_mask, is_first = memory.sample(on_policy)

    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, softmax_dim=1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()

    rho = pi.detach()/prob
    rho_a = rho.gather(1,a)
    rho_bar = rho_a.clamp(max=c)
    correction_coeff = (1-c/rho).clamp(min=0)

    q_ret = v[-1] * done_mask[-1]
    q_ret_list = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + gamma * q_ret
        q_ret_list.append(q_ret.item())
        q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

        if is_first[i] and i != 0:
            # When a new seqence begins, q_ret is Re-initialized
            q_ret = v[i-1] * done_mask[i-1]

    q_ret_list.reverse()
    q_ret = torch.tensor(q_ret_list, dtype=torch.float).unsqueeze(1)

    loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v)
    loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v) # Bias correction term
    loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)

    optimizer.zero_grad()
    loss.mean().backward() # Use average loss For updating weights
    optimizer.step()

def main():
    # Construct
    env = gym.make('CartPole-v1')
    memory = ReplayBuffer()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    score = 0.0
    print_interval = 20

    epi_axes = []
    rwd_axes = []

    for n_epi in range(2000):
        s = env.reset()
        done = False

        while not done:
            seq_data = []
            for t in range(roll):
                prob = model.pi(torch.from_numpy(s).float())
                a = Categorical(prob).sample().item()
                s_prime, r, done, info = env.step(a)
                seq_data.append((s, a, r/100.0, prob.detach().numpy(), done))

                score += r
                s = s_prime
                if done:
                    break

            memory.put(seq_data)
            if memory.size() > 500:
                train(model, optimizer, memory, on_policy=True)
                train(model, optimizer, memory, on_policy=False)

        if n_epi % print_interval == 0 and n_epi != 0:
            reward_avg = score / print_interval
            print("# of episode: {}, avg score: {:.1f}, buffer size: {}".format(n_epi, reward_avg, memory.size()))
            epi_axes.append(n_epi)
            rwd_axes.append(reward_avg)
            score = 0.0

    env.close()

    ### Plotting & Save result
    plt.plot(epi_axes, rwd_axes, label='ACER')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title("ACER Plot")
    plt.legend()
    plt.savefig("plot/ACER_0.png")
    plt.show()

if __name__ == '__main__':
    main()
