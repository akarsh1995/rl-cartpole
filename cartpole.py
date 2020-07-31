import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns


class CartPoleMove:
    env = gym.make('CartPole-v0')

    def __init__(self, render=True):
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


class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_actions = CartPoleMove.env.action_space.n
        self.state_dim = CartPoleMove.env.observation_space.shape[0]
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


restore = True

if restore and os.path.isfile("policy.pt"):
    policy = torch.load("policy.pt")
else:
    policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=0.001)


def update_policy(states, actions, rewards, log_probs, gamma=.99):
    """
    Calculate loss, compute gradients, backpropogate and update policy network params.

    :param states: a list of states in an episode
    :param actions: a list of actions taken in an episode
    :param rewards: a list of rewareds earned at each time step
    :param log_probs: a list of log probabliities of actions taken
    :param gamma: reward discount factor
    :return:
    """
    loss = []
    dis_rewards = rewards[:]
    for i in range(len(dis_rewards) - 2, -1, -1):
        dis_rewards[i] = dis_rewards[i] + gamma * dis_rewards[i+1]

    dis_rewards = torch.tensor(dis_rewards)

    for log_prob, reward in zip(log_probs, dis_rewards):
        loss.append(-log_prob * reward)

    loss = torch.cat(loss).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_policy_values(state):
    state = Variable(torch.from_numpy(state)).type(torch.FloatTensor).unsqueeze(0)
    policy_values = policy(state)
    return policy_values


def generate_episode(t_max=1000):
    """
    Generate and episode. Save states, actions, rewards and log probablilities.
    Update policy
    :param t_max: maximum timesteps in an episode
    :return: undiscounted rewards in the episode
    """
    states, actions, rewards, log_probs = [], [], [], []

    with CartPoleMove(render=False) as cartpole:
        for _ in range(t_max):
            action_probs = F.softmax(
                get_policy_values(cartpole.current_observation), dim=-1)
            sampler = Categorical(action_probs)
            a = sampler.sample()
            log_prob = sampler.log_prob(a)
            cartpole.take_move(a.item())

            # collect information from move
            states.append(cartpole.current_observation)
            actions.append(cartpole.action)
            rewards.append(cartpole.reward)
            log_probs.append(log_prob)

            if cartpole.done:
                break
        update_policy(states, actions, rewards, log_probs)
        return sum(rewards)


def play_episodes(num_episodes=10, render=False):
    """
    Play some episodes using trained policy.
    :param num_episodes: num of episodes to play
    :param render: whther to render a video
    :return:
    """

    for i in range(num_episodes):
        rewards = []
        with CartPoleMove(render=render) as cartpole:
            for _ in range(1000):
                action_probs = F.softmax(
                    get_policy_values(cartpole.current_observation),
                    dim=-1
                )
                sampler = Categorical(action_probs)
                a = sampler.sample()
                log_prob = sampler.log_prob(a)
                cartpole.take_move(a.item())
                rewards.append(cartpole.reward)
                if cartpole.done:
                    print("Episode {} finished with reward {}".
                          format(i+1, np.sum(rewards)))
                    break


num_episodes = 1500
verbose = True
print_every = 50
target_avg_reward_100ep = 195
running_reward = None
rewards = []
running_rewards = []
restore_model = True

if restore_model and os.path.isfile("policy.pt"):
    policy = torch.load('policy.pt')
else:
    policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=0.001)

# Generate episodes 'num_episodes' times
# and update policy after every episode.

for i in range(num_episodes):
    reward = generate_episode()
    rewards.append(reward)
    running_reward = np.mean(rewards[-100:])
    running_rewards.append(running_reward)

    if verbose:
        should_print = not i % print_every
        if should_print:
            print(f"Episode: {i+1}. Running reward: {running_reward}")

    if i >= 99 and running_reward >= target_avg_reward_100ep:
        print(f"Episode: {i+1}. Running reward: {running_reward}")
        print(f"Ran {i+1} episodes. Solved after {i-100+1} episodes.")
        break
    elif i == num_episodes-1:
        print("Couldn't solve after {}".format(num_episodes))

torch.save(policy, "policy.pt")
