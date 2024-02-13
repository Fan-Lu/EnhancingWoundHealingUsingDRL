####################################################
# Description: Proximal Policy Optimization
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-25
####################################################
import random

# In my experience, A2C works better than A3C and ACKTR is better than both of them.
# Moreover, PPO is a great algorithm for continuous control.
# Thus, I recommend to try A2C/PPO/ACKTR first and use A3C only if you need it specifically for some reasons.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from utils.memories import TrajectoryBuffer

class PPO_Net(nn.Module):

    def __init__(self, action_low, action_high, state_size, action_size):
        super(PPO_Net, self).__init__()

        self.action_low = action_low
        self.action_high = action_high
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        self.valueFunc = nn.Linear(64, 1)
        self.policyFunc = nn.Linear(64, self.action_size)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        values = self.valueFunc(x)
        # TODO: Do we need activation on the last policy layer?
        policies = F.sigmoid(self.policyFunc(x)) * self.action_high

        return policies, values


class Agent_PPO(object):

    def __init__(self, env, args):
        self.args = args
        self.env = env

        self.state_size = env.state_size
        self.action_size = env.action_space.shape[0]
        self.action_low = torch.from_numpy(env.action_space.low)
        self.action_high = torch.from_numpy(env.action_space.high)

        self.batch_size = args.batch_size
        self.LR = args.LR
        self.TAU = args.TAU
        self.GAMMA = args.GAMMA
        self.K_epochs = args.K_epochs
        self.eps_clip = args.eps_clip
        self.UPDATE_EVERY_PPO = args.UPDATE_EVERY_PPO
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        # TODO
        self.action_std = args.action_std_init
        self.action_std_decay_rate = args.action_std_decay_rate
        self.min_action_std = args.min_action_std
        self.action_std_decay_freq = args.action_std_decay_freq

        self.action_var = torch.full((self.action_size,),
                                     self.action_std * self.action_std).to(self.device)

        self.model = PPO_Net(self.action_low, self.action_high, self.state_size, self.action_size)
        self.model_old = PPO_Net(self.action_low, self.action_high, self.state_size, self.action_size)
        self.model_old.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.LR)

        self.mse_loss = nn.MSELoss()

        # Replay memory
        self.memory = TrajectoryBuffer(self.state_size, self.action_size, args.buffer_size, self.device)
        # Initialize time step (for updating every UPDATE_EVERY step)
        self.t_step = 0
        # Initialize time step (for updating every action_std step)
        self.t_step_std = 0

    def act(self, state, eps=1.):
        '''
        Returns actions for given state as per current policy

        :param state:
        :param eps: (float) epsilon, for epsilon-greedy action selection
        :return:
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model_old.eval()
        with torch.no_grad():
            action_mean, value = self.model_old(state)
        self.model_old.train()

        if random.random() > eps:
            action = action_mean.cpu().data.numpy()[0][:]
        else:
            # calculate covariance matrix
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)

            # TODO: Round Action
            action = torch.clamp(dist.sample(), min=self.action_low, max=self.action_high)
            action = action.cpu().data.numpy()[0][:]
            # The position should be integer
            if self.args.spt:
                action[1] = int(action[1])
        # TODO
        return action

    def decay_action_std(self):
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = max(round(self.action_std, 4), self.min_action_std)
        # Creates a tensor of size self.action_size filled with fill_value.
        # The tensor’s dtype is inferred from fill_value.
        # Action variance
        self.action_var = torch.full((self.action_size,), self.action_std * self.action_std).to(self.device)

    def evaluate(self, model, states, actions):
        states_means, states_values = model(states)

        cov_mats = torch.diag(self.action_var).unsqueeze(dim=0)
        dists = MultivariateNormal(states_means, cov_mats)

        action_logprobs = dists.log_prob(actions)
        dist_entropies = dists.entropy()

        return states_values, action_logprobs, dist_entropies

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
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY_PPO
        if self.t_step == 0:
            # If enough samples are available in memory,
            # get random subset and learn every UPDATE_EVERY steps
            # if self.memory.__len__() > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states_policies_old, states_values_old = self.model_old(states)
        _, next_values_old = self.model_old(next_states)

        # rollout experiences and calculate expected value for each step
        target_vales_old = []
        R = next_values_old[-1]
        for idx in reversed(range(len(rewards))):
            R = rewards[idx] + self.GAMMA * R * (1.0 - dones[idx])
            # always add to the list at the first pos since we are going backwards
            target_vales_old.insert(0, R)

        target_vales_old = torch.cat(target_vales_old).view(-1, 1)
        # Normalizing the rewards
        target_vales_old = (target_vales_old - target_vales_old.mean()) / (target_vales_old.std() + 1e-7)

        # calculate advantages: Q(s, a) - V(s)
        advantages = target_vales_old.detach() - states_values_old.detach()

        _, action_logprobs_old, dist_entropies_old = self.evaluate(self.model_old, states, actions)

        # Optimize PPO model for K epochs
        for _ in range(self.K_epochs):
            # Evaluationg old actions and values
            states_values, action_logprobs, dist_entropies = self.evaluate(self.model, states, actions)

            # find the ratio： (pi_theta / pi_theta_old)
            # TOOD: Check detach
            ratios = torch.exp(action_logprobs - action_logprobs_old.detach()).view(-1, 1)

            # Find the surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # TODO: MSE
            loss = -torch.min(surr1, surr2).mean() \
                   + 0.5 * self.mse_loss(states_values, target_vales_old) \
                   - 0.01 * dist_entropies.mean()

            # take gradient step
            self.optimizer.zero_grad()
            # TODO: Check retain_graph
            loss.backward(retain_graph=True)
            self.optimizer.step()

        self.t_step_std = (self.t_step_std + 1) % self.action_std_decay_freq
        if self.t_step_std == 0:
            self.decay_action_std()

        # update old model
        self.model_old.load_state_dict(self.model.state_dict())
        # reset memory
        self.memory.reset()