from dataclasses import dataclass
import torch
from random import sample
import numpy as np
import os
from pathlib import Path

def get_temp_dir(*paths_to_join):
    return Path(os.environ["TMPDIR"]).joinpath(*paths_to_join)

def exp_decay(epoch):
    k = 0.999998
    return k ** (epoch)


@dataclass
class StateTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict

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


device = "cuda" if torch.cuda.is_available() else "cpu"
