import os
import numpy as np
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import tqdm
import random


class QNetwork(nn.Module):
    """
    Deep Q-network
    """

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class ReplayMemory:
    """
    Experience replay
    This class stores trials and shuffles them around such that the model
    will not easily get stuck in a local optimum
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_epsilon(it):

    # random actions at first,
    if it >= 1000:
        epsilon = 0.05
    else:
        epsilon = 1 - 0.95 * it * (1 / 1000)

        # after 1000 iterations, e-greedy with epsilon being 0.5
    return epsilon


def select_action(model, state, epsilon):

    # feed state to model to extract action
    actions = model(torch.FloatTensor(state))
    values, indices = actions.max(0)

    # determine if the model explores or not
    if random.random() < epsilon:
        a = np.random.choice(range(len(actions))).item()
    else:
        a = indices.item()

    return a


def compute_q_val(model, state, action):

    # get the actions given a batch of states
    actions = model(state)

    # the output of the model represents the q-values
    q_val = torch.gather(actions, 1, action.view(-1, 1))

    return q_val


def compute_target(model, reward, next_state, done, discount_factor):
    """
    Method that uses the next state to compute the target
    """

    # get action
    actions = model(next_state)
    _, indices = actions.max(1)

    # convert to target
    indices = torch.gather(actions, 1, indices.view(-1, 1))
    target = reward.view(indices.shape) + (discount_factor * indices)

    # set target to just the reward if next_state is terminal
    target[done] = reward[done].view(target[done].shape)
    return target


def train(model, memory, optimizer, batch_size, discount_factor):
    """
    Method to train the model
    """

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.bool)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()