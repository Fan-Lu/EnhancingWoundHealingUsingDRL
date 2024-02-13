####################################################
# Description: Momories
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-17
####################################################
import random
from collections import namedtuple, deque

import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, n_observations, n_actions, buffer_size, batch_size, device, seed):
        '''
        Initialize a replayBuffer object

        :param n_observations: dimension of each state
        :param n_actions:  dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: GPU or CPU
        :param seed: random seed
        '''

        self.state_size = n_observations
        self.action_size = n_actions
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        # self.experience = namedtuple('Experience', field_names=["state",
        #                                                         "action",
        #                                                         "reward",
        #                                                         "next_state",
        #                                                         "done"])
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory

        :param experience: (tuple)
        :return:
        '''
        # upack experience
        # exp = self.experience(state, action, reward, next_state, done)
        # add to deque
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        '''
        Ramdomly sample a batch of experience from memory

        :return:
        '''

        exps = random.sample(self.memory, k=self.batch_size)

        # states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(self.device)

        states = torch.from_numpy(np.vstack([e[0] for e in exps if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in exps if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in exps if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in exps if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in exps if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        Return the current size of internal memory

        :return:
        '''

        return len(self.memory)


class TrajectoryBuffer(object):

    def __init__(self, n_observations, n_actions, buffer_size, device):
        '''
        Initialize a replayBuffer object

        :param n_observations: dimension of each state
        :param n_actions:  dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: GPU or CPU
        :param seed: random seed
        '''

        self.state_size = n_observations
        self.action_size = n_actions
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        # self.experience = namedtuple('Experience', field_names=["state",
        #                                                         "action",
        #                                                         "reward",
        #                                                         "next_state",
        #                                                         "done"])

    def push(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory

        :param experience: (tuple)
        :return:
        '''
        # upack experience
        # exp = self.experience(state, action, reward, next_state, done)
        # add to deque
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        '''
        Ramdomly sample a batch of experience from memory

        :return:
        '''

        exps = self.memory

        # states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(self.device)

        states = torch.from_numpy(np.vstack([e[0] for e in exps if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in exps if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in exps if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in exps if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in exps if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def reset(self):
        '''
        Clear memory
        '''
        self.memory.clear()

    def __len__(self):
        '''
        Return the current size of internal memory

        :return:
        '''

        return len(self.memory)


class TransCache(object):

    def __init__(self, batch_size, device):
        '''
        Initialize a replayBuffer object

        :param n_observations: dimension of each state
        :param n_actions:  dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: GPU or CPU
        :param seed: random seed
        '''

        self.device = device
        self.memory = deque(maxlen=batch_size)

    def push(self, state, action):
        '''
        Add a new experience to memory

        :param experience: (tuple)
        :return:
        '''
        self.memory.append((state, action))

    def sample(self):
        '''
        Ramdomly sample a batch of experience from memory

        :return:
        '''

        exps = self.memory

        states = torch.from_numpy(np.vstack([e[0] for e in exps if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in exps if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e[2] for e in exps if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e[3] for e in exps if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e[4] for e in exps if e is not None]).astype(np.uint8)).float().to(self.device)

        # return (states, actions, rewards, next_states, dones)
        return (states, actions)

    def reset(self):
        '''
        Clear memory
        '''
        self.memory.clear()

    def __len__(self):
        '''
        Return the current size of internal memory

        :return:
        '''

        return len(self.memory)
