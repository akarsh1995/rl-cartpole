import gym
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Any, List
from random import sample
from torch.optim.adam import Adam



@dataclass
class StateTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: int
    done: bool
    info: Any

    @property
    def state_tensor(self):
        return torch.Tensor(self.state)

    @property
    def next_state_tensor(self):
        return torch.Tensor(self.next_state)

    @property
    def action_tensor(self):
        return torch.Tensor([self.action])

    @property
    def not_done_tensor(self):
        return torch.Tensor([int(not self.done)])

    @property
    def action_tensor(self):
        return torch.Tensor([int(self.action)])

    @property
    def reward_tensor(self):
        return torch.Tensor([self.reward])


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = [None] * buffer_size
        self.idx = 0
        self.buffer_size = buffer_size

    def insert(self, state_transition: StateTransition):
        self.buffer[self.idx % self.buffer_size] = state_transition
        self.idx += 1

    def sample(self, size: int):
        assert size <= self.idx + 1, "Cant sample for more than buffer size"
        if self.idx < self.buffer_size:
            return sample(self.buffer[: self.idx], size)
        return sample(self.buffer, size)


class CartpoleEnv:
    env = gym.make("CartPole-v0")

    def __init__(self, maintain_buffer=True):
        self.current_state = self.env.reset()
        self.replay_buffer = ReplayBuffer()
        self.maintain_buffer = maintain_buffer
        self.episode_finished = False

    def __enter__(self, *args, **kwargs):
        return self

    @classmethod
    def n_actions(cls):
        return cls.env.action_space.n

    @classmethod
    def n_states(cls):
        return cls.env.observation_space.shape[0]

    def __exit__(self, *args, **kwargs):
        self.env.close()

    def step(self, action):
        step = self.env.step(action)
        st_tr = StateTransition(self.current_state, action, *step)
        if self.maintain_buffer:
            self.replay_buffer.insert(st_tr)
        self.current_state = step[0]
        self.episode_finished = st_tr.done
        return st_tr

    def random_step(self):
        sample_action = self.env.action_space.sample()
        return self.step(sample_action)


class DQNNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states, 120)  # 6*6 from image dimension
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(120, n_actions)
        self.optimizer = Adam(self.parameters(), lr=0.01)
       
    def forward(self, observation):
        x = self.fc1(observation)
        x = self.fc2(x)
        action_probs = self.fc3(x)
        return action_probs

def train_step(
        model,
        target_model,
        transition_states: List[StateTransition],
        n_actions = 2,
        discount_factor=0.99):
    states = torch.stack([trans.state_tensor for trans in transition_states])
    next_states = torch.stack([trans.next_state_tensor for trans in transition_states])
    not_done = torch.stack([trans.not_done_tensor for trans in transition_states])
    actions = [trans.action for trans in transition_states]
    rewards = torch.stack([trans.reward_tensor for trans in transition_states])

    with torch.no_grad():
        qvals_predicted = target_model(next_states).max(-1)

    model.optimizer.zero_grad()
    qvals_current = model(states)
    one_hot_actions = torch.nn.functional.one_hot(
        torch.LongTensor(actions), n_actions
    )
    loss = (
        (
            rewards
            + (not_done * qvals_predicted.values)
            - torch.sum(qvals_current * one_hot_actions, -1)
        ) ** 2
    ).mean()
    loss.backward()
    model.optimizer.step()
    return loss

def main():
    model = DQNNetwork(CartpoleEnv.n_states(), CartpoleEnv.n_actions())
    target_model = DQNNetwork(CartpoleEnv.n_states(), CartpoleEnv.n_actions())
    # take a random action
    with CartpoleEnv() as env:
        while not env.episode_finished:
            env.random_step()
    train_step(model, target_model, env.replay_buffer.sample(15))

if __name__ == "__main__":
    main()
