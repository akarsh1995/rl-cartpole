from numpy.matrixlib import defmatrix
import torch.nn as nn
from torch.optim.adam import Adam
import torch

class DQNNetwork(nn.Module):
    def __init__(self, n_states, n_actions, lr=0.01):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states, 120)  # 6*6 from image dimension
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(120, n_actions)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, observation):
        x = self.fc1(observation)
        x = self.fc2(x)
        action_probs = self.fc3(x)
        return action_probs

class ConvNet(nn.Module):

    def __init__(self, obs_shape, num_actions, lr=0.001):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions =num_actions
        self.lr = lr
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, (8, 8), stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=(2,2)),
            nn.ReLU()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(self.first_linear, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.optimizer = Adam(self.parameters(), lr=lr)

    @property
    def first_linear(self):
        # calculating final layer output size
        s = self.net(torch.zeros(1, *self.obs_shape))
        return s.flatten().shape[0]

    def forward(self, x):
        conv_latent = self.net(x/255.0)
        return self.fc_net(conv_latent.flatten(1))

if __name__ == '__main__':
    w = 84
    h = 84
    stack = 4
    m = ConvNet((stack, h, w), 4)
    fake_tensor = torch.zeros((4, *(stack, h, w)))
    print(m(fake_tensor))
