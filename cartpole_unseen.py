import gym
import numpy as np
from dataclasses import dataclass
from typing import Any
from random import sample
from copy import deepcopy
from tqdm import tqdm
from collections import deque
from models import DQNNetwork
import torch
from utils import exp_decay, online_logger
import datetime

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
        return torch.Tensor([self.reward / 100])


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

    def __init__(self, maintain_buffer=True):
        self.env = gym.make("CartPole-v0")
        self.current_state = self.env.reset()
        self.episode_finished = False
        self.reward = 0

    def __enter__(self, *args, **kwargs):
        return self

    @property
    def n_actions(self):
        return self.env.action_space.n

    @property
    def n_states(self):
        return self.env.observation_space.shape[0]

    def __exit__(self, *args, **kwargs):
        self.env.close()

    def step(self, action):
        step = self.env.step(action)
        st_tr = StateTransition(self.current_state, action, *step)
        self.current_state = step[0]
        self.reward += 1
        self.episode_finished = st_tr.done
        return st_tr

    def random_step(self):
        sample_action = self.env.action_space.sample()
        return self.step(sample_action)


class DQNModelsHandler:
    def __init__(self, env_class: CartpoleEnv, buffer_size):
        self.environment_class = env_class
        with env_class() as env:
            self.n_states = env.n_states
            self.n_actions = env.n_actions
        self.model = DQNNetwork(self.n_states, self.n_actions, lr=0.001)
        self.target_model = DQNNetwork(self.n_states, self.n_actions, lr=0.001)
        self.rolling_loss = deque(maxlen=12)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.episode_count = 0
        self.rolling_reward = deque(maxlen=12)
        self.model_update_count = 0

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
            )
            ** 2
        ).mean()
        loss.backward()
        self.model.optimizer.step()
        self.rolling_loss.append(loss.detach().item())
        self.model_update_count += 1

    def update_target_model(self):
        state_dict = deepcopy(self.model.state_dict())
        self.target_model.load_state_dict(state_dict)

    def play_episode(self, update_model=True):
        with self.environment_class() as env:
            while not env.episode_finished:
                if (exp_decay(self.n_steps) > np.random.random()
                     and self.n_steps > self._min_samples_before_update):
                    state_trans = env.random_step()
                else:
                    state = env.current_state
                    predicted_action = self.target_model(torch.Tensor(state))
                    state_trans = env.step(predicted_action.argmax().item())

                self.replay_buffer.insert(state_trans)
                if update_model:
                    if self.matches_update_criteria():
                        self.train_step()
                        self.update_target_model()
                        self.check_reward()
                        self.verbose_training()
            self.episode_count += 1

    def check_reward(self):
        with self.environment_class() as reward_env:
            while not reward_env.episode_finished:
                state = reward_env.current_state
                predicted_action = self.target_model(torch.Tensor(state))
                reward_env.step(predicted_action.argmax().item())
        self.rolling_reward.append(reward_env.reward)

    def get_rolling_reward(self):
        return sum(self.rolling_reward) / len(self.rolling_reward)

    def get_rolling_loss(self):
        return sum(self.rolling_loss) / len(self.rolling_loss)

    def set_model_updt_criteria(
        self, min_samples_before_update, update_every, sampling_size
    ):
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

    def verbose_training(self, online_logging=True):
        if online_logging:
            online_logger.log({
                "Rolling_reward: ": self.get_rolling_reward(),
                "Rolling Loss:": self.get_rolling_loss()
            })
        print("Rolling_reward: ", self.get_rolling_reward())
        print("Rolling Loss:", self.get_rolling_loss())


def main():
    env_class = CartpoleEnv
    buffer_size = 100000
    update_every_nth_episode = 500
    sampling_size = 5000
    minimum_samples_before_update = 10000
    models_handler = DQNModelsHandler(env_class, buffer_size)
    models_handler.set_model_updt_criteria(
        minimum_samples_before_update, update_every_nth_episode, sampling_size
    )
    model_save_at_nth_update = 50
    # progress = tqdm()
    try:
        while True:
            # progress.update(1)
            models_handler.play_episode()
            update_count = models_handler.model_update_count
            if update_count % model_save_at_nth_update == 0 and update_count >= model_save_at_nth_update:
                model_save_name = f"{datetime.datetime.now().strftime('%H:%M:%S')}.pth"
                torch.save(models_handler.target_model.state_dict(), model_save_name)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
