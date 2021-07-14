import gym
from numpy import savez_compressed
import torch
from torch._C import TracingState
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Categorical : torch가 지원하는 확률분포 모델 패키지
from torch.distributions import Categorical

# Decaying factor gamma
gamma = 0.99
# Learning rate
lr = 0.0008

class Policy(nn.Module):
  def __init__(self):
    super(Policy, self).__init__()

    self.data = []
    self.fc1 = nn.Linear(4, 128)
    self.fc2 = nn.Linear(128, 2) # only 2 Action class
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    # fc2 거친 후엔 아직 확률값은 아님 => softmax로 확률로 변환
    x = F.relu(self.fc1(x))
    x = F.softmax(self.fc2(x), dim=0)
    return x

  def put_data(self, item):
    """REINFORCE는 에피소드가 끝난 후 학습하기 때문에 데이터를 모은다."""
    self.data.append(item)

  def train(self):
    R = 0
    # 학습하고자 하는 Parameters에 대해서 매 스텝 가중치 초기화
    self.optimizer.zero_grad()

    for r, prob in self.data[::-1]:
      R = r + R * gamma
      loss = -torch.log(prob) * R
      # because PyTorch accumulates the gradients on subsequent backward passes.
      loss.backward()

    self.optimizer.step()
    self.data = [] # Reset


def main():
  # State : only (4,1) vector
  env = gym.make("CartPole-v1")
  pi = Policy()
  score = 0.0
  print_interval = 20

  for n_epi in range(10000):
    s = env.reset()
    done = False

    while not done: # CartPole-v1 forced to terminates at 500 step.
      prob = pi(torch.from_numpy(s).float())
      m = Categorical(prob)
      a = m.sample()
      s_prime, r, done, info = env.step(a.item())
      pi.put_data((r,prob[a]))
      s = s_prime
      score += r

    pi.train()

    if n_epi%print_interval==0 and n_epi!=0:
      print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
      score = 0.0
  env.close()



if __name__ == '__main__':
  main()

