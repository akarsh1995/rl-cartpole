import os
from datetime import datetime
from copy import deepcopy
from dataclasses import field, dataclass
from atari_frames_wrapper import Frame
from utils import ReplayBuffer, StateTransition
import gym
from models import ConvNet
import torch
from typing import Callable
import wandb

@dataclass
class ModelsHandler:
    input_shape: tuple
    num_actions: int
    lr: float = field(default=0.001)

    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ConvNet(self.input_shape, self.num_actions, self.lr).to(self.device)
        self.tgt_model = ConvNet(self.input_shape, self.num_actions, self.lr).to(self.device)
        self.model_update_count = 0
        self.current_loss = 0

    def train_step(self, rb: ReplayBuffer, sample_size=300):
        # loss calcualation
        trans_sts = rb.sample(sample_size)
        states = torch.stack([trans.state_tensor for trans in trans_sts]).to(self.device)
        next_states = torch.stack([trans.next_state_tensor for trans in trans_sts]).to(self.device)
        not_done = torch.Tensor([trans.not_done_tensor for trans in trans_sts]).to(self.device)
        actions = [trans.action for trans in trans_sts]
        rewards = torch.stack([trans.reward_tensor for trans in trans_sts]).to(self.device)

        with torch.no_grad():
            qvals_predicted = self.tgt_model(next_states).max(-1)

        self.model.optimizer.zero_grad()
        qvals_current = self.model(states)
        one_hot_actions = torch.nn.functional.one_hot(
            torch.LongTensor(actions), self.num_actions
        ).to(self.device)
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
        return loss.detach().item()


    def update_target_model(self):
        state_dict = deepcopy(self.model.state_dict())
        self.tgt_model.load_state_dict(state_dict)
        self.model_update_count += 1

    def save_target_model(self):
        file_name = f"{datetime.now().strftime('%H:%M:%S')}.pth"
        temp_dir = os.environ.get('TMPDIR', '/tmp')
        file_name = os.path.join(temp_dir, file_name)
        torch.save(self.model, file_name)
        wandb.save(file_name)

   
@dataclass
class EpisodeManager:
    env_reg_name: str
    buffer_size: int

    def __post_init__(self):
        self.env = gym.make(self.env_reg_name)
        self.num_steps = 0
        self.episodes = 0
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def play_episode(self, frame_wrapper: Frame, render=False, model: Callable = None):
        prev_obs = frame_wrapper.reset(self.env)
        done = False
        while not done:
            if render:
                self.env.render()
            if model is not None:
                action = model(prev_obs)
            else:
                action = self.env.action_space.sample()
            obs, reward, done, info = self.step(action, frame_wrapper)
            self.replay_buffer.insert(StateTransition(prev_obs, action, obs, reward, done, info))
            prev_obs = obs
            self.num_steps += 1
        self.episodes += 1

    def step(self, action: int, frame_wrapper: Frame):
        if frame_wrapper is not None:
            out = frame_wrapper.step(self.env, action=action)
        else:
            out = self.env.step(action)
        return out


def train_model(
        height = 84,
        width = 84,
        num_stacks = 4,
        lr = 0.001,
        num_actions = 4,
        buffer_size = 100000,
        min_steps_bef_first_update=10000,
        train_step_every_nth_episode=30,
        sample_size=500,
        target_update_every=50,
):
    wandb.init()
    model_manager = ModelsHandler((num_stacks, height, width), num_actions, lr)
    ep_mgr = EpisodeManager("Breakout-v0", buffer_size)
    f = Frame(height, width, num_stacks)
    try:
        while ep_mgr.episodes < 1000:
            ep_mgr.play_episode(f)
            if ep_mgr.num_steps > min_steps_bef_first_update:
                if ep_mgr.episodes % train_step_every_nth_episode == 0:
                    loss = model_manager.train_step(ep_mgr.replay_buffer, sample_size)
                    wandb.log({'loss': loss, 'episodes': ep_mgr.episodes})
                if ep_mgr.episodes % target_update_every == 0:
                    model_manager.update_target_model()
                    model_manager.save_target_model()

    except Exception as e:
        print(e)


if __name__ == '__main__':
    import argh
    argh.dispatch_command(train_model)
