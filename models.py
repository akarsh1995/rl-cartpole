import torch.nn as nn
from torch.optim.adam import Adam


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
