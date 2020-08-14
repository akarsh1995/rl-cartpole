from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from random import sample
from typing import Any, List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

    @property
    def state_tensor(self):
        return torch.Tensor(self.state)

    @property
    def next_state_tensor(self):
        return torch.Tensor(self.next_state)

    @property
    def reward_tensor(self):
        return torch.Tensor([self.reward])

    @property
    def action_tensor(self):
        return torch.Tensor([self.action])

    @property
    def done_tensor(self):
        return torch.Tensor([int(self.done)])


class CartPoleMove:
    env = gym.make('CartPole-v0')

    def __init__(self, render=False):
        self.current_observation = self.env.reset()
        self.action = None
        self.done = False
        self.render = render
        self.reward = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()

    def move_left(self):
        self.take_move(0)

    def move_right(self):
        self.take_move(1)

    def random_move(self):
        action = self.env.action_space.sample()
        self.take_move(action)

    def take_move(self, move_id):
        if not self.done:
            if self.render:
                self.env.render()
            self.action = move_id
            self.current_observation, self.reward, self.done, __ = self.env.step(
                self.action
            )


class DQNModel(nn.Module):

    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(256, self.num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_step(model: DQNModel, target_model: DQNModel, state_transitions: List[Sarsd], discount_factor=.95,
               num_actions=2):
    actions = [s.action for s in state_transitions]
    current_states = torch.stack([s.state_tensor for s in state_transitions])
    next_states = torch.stack([s.next_state_tensor for s in state_transitions])
    done = torch.Tensor([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])
    rewards = torch.stack([s.reward_tensor for s in state_transitions])

    with torch.no_grad():
        q_vals_next = target_model(next_states).max(-1)

    model.optimizer.zero_grad()
    qvals_current = model(current_states)
    one_hot_actions = torch.nn.functional.one_hot(
         torch.LongTensor(actions), num_actions
    )

    loss = (
            (
                    rewards
                    + done * q_vals_next.values
                    - torch.sum(qvals_current * one_hot_actions, -1)
            )**2
    ).mean()

    loss.backward()
    model.optimizer.step()
    return loss


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def insert(self, sars: Sarsd):
        self.queue.append(sars)

    def sample(self, size: int):
        assert size <= len(self.queue)
        return sample(self.queue, size)


def update_target_model(model, target):
    target.load_state_dict(deepcopy(model.state_dict()))


def main(should_log=False):
    from tqdm import tqdm
    if should_log:
        wandb.init(project='dqn-learning', name='dqn-cartpole')
    min_rb_size = 10000
    sample_size = 5000
    env_steps_before_train_step = 1000
    target_model_update = 50
    num_actions = CartPoleMove.env.action_space.n
    observation_space = CartPoleMove.env.observation_space.shape[0]
    model = DQNModel(observation_space, num_actions)
    target = DQNModel(observation_space, num_actions)

    replay_buffer = ReplayBuffer(100000)

    steps_since_train_update = 0
    epochs_since_target = 0
    step_num = -1 * min_rb_size
    progress = tqdm()
    try:
        while True:
            progress.update(1)
            with CartPoleMove() as cartpole:
                while not cartpole.done:
                    obs = cartpole.current_observation
                    cartpole.random_move()
                    action = cartpole.action
                    reward = cartpole.reward
                    next_observation = cartpole.current_observation
                    s = Sarsd(obs, action, reward, next_observation, cartpole.done)
                    replay_buffer.insert(s)
                    # when the train_step should be performed
                    if len(replay_buffer.queue) > min_rb_size and steps_since_train_update > env_steps_before_train_step:
                        loss = train_step(model, target, replay_buffer.sample(sample_size), num_actions)
                        if should_log:
                            wandb.log({'loss': loss.detach().item(), 'step': step_num})
                        print(step_num, loss.detach().item())
                        steps_since_train_update = 0
                        epochs_since_target += 1
                        if epochs_since_target > target_model_update:
                            print('updating target model.')
                            update_target_model(model, target)
                            epochs_since_target = 0
                    steps_since_train_update += 1
                    step_num += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt.')


if __name__ == '__main__':
    main(should_log=False)
