##################################################################
# Description: Twin Delayed Deep Deterministic Policy Gradient
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-08-01
##################################################################

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from utils.memories import ReplayBuffer


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        torch.manual_seed(0)

        self.fc1a = nn.Linear(self.action_size + self.state_size, 128)
        self.fc2a = nn.Linear(128, 64)
        self.fc3a = nn.Linear(64, 1)

        self.fc1b = nn.Linear(self.action_size + self.state_size, 128)
        self.fc2b = nn.Linear(128, 64)
        self.fc3b = nn.Linear(64, 1)

    def forward(self, state, action):
        inputs = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1a(inputs))
        x = F.relu(self.fc2a(x))
        Q1 = self.fc3a(x)

        x = F.relu(self.fc1b(inputs))
        x = F.relu(self.fc2b(x))
        Q2 = self.fc3b(x)

        return Q1, Q2

    def Q1(self, state, action):
        inputs = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1a(inputs))
        x = F.relu(self.fc2a(x))
        Q1 = self.fc3a(x)

        return Q1


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        torch.manual_seed(0)

        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


class Agent_TD3(object):

    def __init__(self, env, args):
        '''
        Advantage Actor Critic
        '''
        super(Agent_TD3, self).__init__()
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.batch_size = args.batch_size
        self.LR = args.LR
        self.TAU = args.TAU
        self.GAMMA = args.GAMMA
        random.seed(0)
        torch.manual_seed(0)

        self.noise_std = 0.1
        self.noise_clip = 0.5

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and False else "cpu")

        # Actor Network (w/ Target Network)
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR)

        # n_observations, n_actions, buffer_size, batch_size, device, seed)
        # Replay memory
        self.memory = ReplayBuffer(self.state_size, self.action_size, args.buffer_size, self.batch_size, self.device, seed=0)
        self.t_step = 0

        self.last_diff = 0
        self.UPDATE_EVERY = 2
        self.cnter = 0

    def step(self, state, action, reward, next_state, done):
        '''

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        # save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if self.memory.__len__() > self.batch_size:
                for i in range(1):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, state, eps=1.):
        '''
        Returns actions for given state as per current policy

        :param state:
        :param eps: (float) epsilon, for epsilon-greedy action selection
        :return:
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return action.cpu().data.numpy()[0]
        else:
            # Generate a random noise
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            # Add noise to the action for exploration
            action = (action + noise).clip(self.env.action_space.min(), self.env.action_space.max())
            return action.cpu().data.numpy()[0]

    def soft_update(self, local_model, target_model):
        '''
        Soft update model parameters
        theta_target = tau * theta_local + (1 - tau) * theta_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :return:
        '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1 - self.TAU) * target_param.data)

    def learn(self, experiences):
        '''
        Update value parameters using given batch of experiences tuples

        :param experiences:
        :return:
        '''
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)
        actions_ = actions.cpu().numpy()
        noise = torch.FloatTensor(actions_).data.normal_(0, self.noise_std).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_actions = (next_actions + noise).clamp(self.env.action_space.min(), self.env.action_space.max())

        # compute the target Q value
        Q1_target, Q2_target = self.critic_target(next_states, next_actions)
        # target_Q = min(Q1_target, Q2_target)
        adjust_diff = (self.last_diff + torch.abs(Q1_target - Q2_target)) / 2
        Q_target = torch.max(Q1_target, Q2_target) - adjust_diff

        self.last_diff = torch.abs(Q1_target - Q2_target)

        Q_target = (rewards + self.GAMMA * Q_target * (1 - dones)).detach()

        # Get current Q estimates
        Q1_current, Q2_current = self.critic(states, actions)

        # compute critic loss
        critic_loss = F.mse_loss(Q1_current, Q_target) + F.mse_loss(Q2_current, Q_target)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.cnter = (self.cnter + 1) % 2
        if self.cnter == 0:
            # compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)