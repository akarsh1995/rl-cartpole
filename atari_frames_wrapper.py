from dataclasses import dataclass
from datetime import datetime
from models import ConvNet

from torchvision.utils import save_image
from utils import get_temp_dir
import torchvision.utils as utils
import torch
from gym.core import Env
import numpy as np
import cv2
import gym


@dataclass
class Frame:
    height: int
    width: int
    num_stacks: int

    def __post_init__(self):
        self.buffer = np.zeros((self.num_stacks, self.height, self.width))

    def _preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def stack_frames(self, image):
        self.buffer[0:self.num_stacks-1, :, :] = self.buffer[ 1:, :, :]
        self.buffer[-1, :, :] = image

    @property
    def observation_shape(self):
        return self.num_stacks, self.height, self.width

    def step(self, env, action=None):
        if action is not None:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        image_processed = self._preprocess_frame(obs)
        self.stack_frames(image_processed)
        return self.buffer.copy(), reward, done, info

    def reset(self, env):
        obs = env.reset()
        image_processed = self._preprocess_frame(obs)
        for i in range(self.num_stacks):
            self.stack_frames(image_processed)
        return self.buffer.copy()

    def get_grid(self):
        stacked = np.expand_dims(self.buffer, 2)
        stacked = stacked.transpose((3, 2, 0, 1))
        imgs_tensor = torch.tensor(stacked)
        grid_image = utils.make_grid(imgs_tensor, 1)
        return grid_image.numpy().transpose((1, 2, 0))

if __name__ == '__main__':
    save_images = False
    env = gym.make("Breakout-v0")
    obs = env.reset()
    f = Frame(640, 480, 4)
    for i in range(40):
        if i == 0:
            f.step(env, 1)
        obs, reward, done, info = f.step(env)
        if save_images:
            if i % 4 == 0:
                cur_date = datetime.now().isoformat()
                # save in temp directory
                file_save_path = f'{get_temp_dir("image {:0>2}.jpg".format(i))}'
                cv2.imwrite(file_save_path, f.get_grid())
                print("file saved at:", file_save_path)

    input_buffer = torch.unsqueeze(torch.Tensor(obs), dim=0)
    model = ConvNet(f.observation_shape, 4)
    print(model(input_buffer))
