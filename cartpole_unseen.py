import gym
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Any, List
from random import sample
from torch.optim.adam import Adam
from copy import deepcopy
from tqdm import tqdm
from collections import deque



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
        return torch.Tensor([self.reward/100])


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

class DQNModelsHandler:

    def __init__(self, env_class: CartpoleEnv, buffer_size):
        self.environment_class = env_class
        self.n_states = env_class.n_states()
        self.n_actions = env_class.n_actions()
        self.model = DQNNetwork(self.n_states, self.n_actions)
        self.target_model = DQNNetwork(self.n_states, self.n_actions)
        self._loss = None
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.episode_count = 0
        self.rolling_reward = deque(maxlen=12)
       
    def train_step(self, dis_fact=0.99):
        trans_sts = self.replay_buffer.sample(self._sampling_size)
        states = torch.stack([trans.state_tensor for trans in trans_sts])
        next_states = torch.stack([trans.next_state_tensor for trans in trans_sts])
        not_done = torch.Tensor([trans.not_done_tensor for trans in trans_sts])
        actions = [trans.action for trans in trans_sts]
        rewards = torch.stack([trans.reward_tensor for trans in trans_sts])

        with torch.no_grad():
            qvals_predicted = self.target_model(next_states).max(-1)

        self.model.optimizer.zero_grad()
        qvals_current = self.model(states)
        one_hot_actions = torch.nn.functional.one_hot(
            torch.LongTensor(actions), self.n_actions
        )
        loss = (
            (
                rewards
                + (not_done * qvals_predicted.values)
                - torch.sum(qvals_current * one_hot_actions, -1)
            ) ** 2
        ).mean()
        loss.backward()
        self.model.optimizer.step()
        self._loss = loss

    def update_target_model(self):
        state_dict = deepcopy(self.model.state_dict())
        self.target_model.load_state_dict(state_dict)
        print(self.get_current_loss())
       
    def get_current_loss(self):
        return self._loss.detach().item()

    def play_episode(self, update_model=True):
        with self.environment_class() as env:
            while not env.episode_finished:
                state_trans = env.random_step()
                self.replay_buffer.insert(state_trans)
                if update_model:
                    if self.matches_update_criteria():
                        self.train_step()
                        self.update_target_model()
        self.episode_count += 1

    def check_reward(self):
        reward = 0
        with self.environment_class() as env:
            while not env.episode_finished:
                reward += 1
                state = env.current_state
                predicted_action = self.target_model(torch.Tensor(state))
                env.step(predicted_action.argmax().item())
        self.rolling_reward.append(reward)
        return reward

    def get_rolling_reward(self):
        return sum(self.rolling_reward)/len(self.rolling_reward)
   
    def set_model_updt_criteria(self, min_samples_before_update, update_every, sampling_size):
        self._min_samples_before_update = min_samples_before_update
        self._update_every = update_every
        self._sampling_size = sampling_size

    @property
    def n_steps(self):
        return self.replay_buffer.idx
   
    def matches_update_criteria(self):
        if self.replay_buffer.idx >= self._min_samples_before_update:
            if self.episode_count % self._update_every == 0:
                return True
        return False

   
def main():
    env_class = CartpoleEnv
    buffer_size = 100000
    update_every_nth_episode = 50
    sampling_size = 5000
    minimum_samples_before_update = 10000
    models_handler = DQNModelsHandler(env_class, buffer_size)
    models_handler.set_model_updt_criteria(
        minimum_samples_before_update,
        update_every_nth_episode,
        sampling_size
    )

    progress = tqdm()
    try:
        while True:
            progress.update(1)
            models_handler.play_episode()
            if models_handler.episode_count % 100 == 0:
                models_handler.check_reward()
                print('rolling_reward: ', models_handler.get_rolling_reward())
               
    except KeyboardInterrupt:
        pass
   
if __name__ == "__main__":
    main()
