import gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

replay_buffer_limit = 50000
sample_batch_size = 32
gamma = 0.98
lr = 0.001

class ReplayBuffer():

    def __init__(self):
        self.buffer = collections.deque(maxlen=replay_buffer_limit)

    def put(self, transition):
        # append to the right, pop from the left
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, sp_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, sp, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            sp_list.append(sp)
            done_mask_list.append([done_mask])

        return \
            torch.tensor(s_list, dtype=torch.float), \
            torch.tensor(a_list), \
            torch.tensor(r_list), \
            torch.tensor(sp_list, dtype=torch.float), \
            torch.tensor(done_mask_list)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Note that the final output could be negative value
        # So, last layer does not include ReLU activate function!!
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # out -> Tensor
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,sp,done_mask = memory.sample(sample_batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(sp).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    score = 0.0
    print_interval = 20

    # q_targt은 업데이트 안함!
    optimizer = optim.Adam(q.parameters(), lr = lr)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            sp, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, sp, done_mask))
            s = sp
            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % 20 == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode :{}, Avg timestep : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                n_epi, score/20.0, memory.size(), epsilon*100
            ))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()

