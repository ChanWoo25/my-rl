import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

########################## REVIEW ##########################
# REINFORCE에 비하여 훨씬 큰 Learning rate로 학습이 가능하며 훨씬 빨랐고,
# 오히려 lr이 작으면 학습이 잘 되지 않았다
# - 아직도 학습은 많이 불안정하여서 수렴 속도가 매번 크게 달라진다.
############################################################


# Hyper Parmeters
lr          = 0.01
gamma       = 0.99
class ActorCritic(nn.Module):
  def __init__(self):
    super(ActorCritic, self).__init__()
    self.loss_list = []

    self.fc1 = nn.Linear(4, 128)
    self.fc_pi = nn.Linear(128, 2) # Policy Estimation
    self.fc_v = nn.Linear(128, 1)  # Value Estimation
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    pol = self.fc_pi(x)
    pi = F.softmax(pol, dim=0)
    v = self.fc_v(x)
    return pi, v

  def gather_loss(self, loss):
    self.loss_list.append(loss.unsqueeze(0))

  def train(self):
    loss = torch.cat(self.loss_list).sum()
    loss = loss/len(self.loss_list)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.loss_list = []

def main():
  env = gym.make('CartPole-v1')
  model = ActorCritic()
  print_interval = 20
  score = 0.0
  vis = False

  for n_epi in range(10000):
    done = False
    s = env.reset()
    while not done:
      s = torch.from_numpy(s).float()
      pi, v = model(s)

      m = Categorical(pi)
      a = m.sample()

      if vis:
        env.render()

      sp, r, done, info = env.step(a.item())
      _, next_v = model(torch.from_numpy(sp).float())
      delta = r + gamma * next_v - v
      loss = - torch.log(pi[a]) * delta.item() + delta * delta
      model.gather_loss(loss)
      score += r

      s = sp

      if done:
        break

    model.train()

    if score / print_interval > 490:
      vis = True

    if n_epi%print_interval==0 and n_epi!=0:
      print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
      score = 0.0
  env.close()

if __name__ == '__main__':
    main()
