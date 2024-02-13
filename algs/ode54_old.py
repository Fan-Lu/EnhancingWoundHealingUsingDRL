####################################################
# Description: Non-linear to linear dynamics
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-10-03
####################################################

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd.functional import jacobian

from scipy.integrate import odeint

from control import lqr

import matplotlib.pyplot as plt

class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, A_dim):
        '''
        Find mapping between non-linear and linear models
        '''
        super(Transformer, self).__init__()

        torch.manual_seed(0)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc11 = nn.Linear(self.in_dim, 64)
        self.fc12 = nn.Linear(64, 64)
        self.fc13 = nn.Linear(64, self.out_dim)

        self.fc21 = nn.Linear(self.out_dim, 64)
        self.fc22 = nn.Linear(64, 64)
        self.fc23 = nn.Linear(64, self.in_dim)

        self.Amat = nn.Linear(A_dim[0], A_dim[1], bias=False)

    def map524(self, x):
        x2z = F.elu(self.fc11(x))
        x2z = F.elu(self.fc12(x2z))
        x2z = self.fc13(x2z)

        return x2z

    def matmult(self, x2z):

        return self.Amat(x2z)

    def map425(self, x2z):

        z2x = F.elu(self.fc21(x2z))
        z2x = F.elu(self.fc22(z2x))
        z2x = self.fc23(z2x)

        return z2x

    def forward(self, x):

        x2z = self.map524(x)
        z2x = self.map425(x2z)
        AN = self.matmult(x2z)

        return x2z, AN, z2x


def dynamics(y, t, u, mu, lam):
    dx1dt = mu * y[0]
    dx2dt = lam * (y[1] - y[0] ** 2) + u
    dxdt = np.array([dx1dt, dx2dt])
    return dxdt


class Environment(object):

    def __init__(self, t_nums, t_days):

        self.mu = -0.1
        self.lam = 1.0
        self.Aopt = np.array([[self.mu, 0, 0],
                              [0, self.lam, -self.lam],
                              [0, 0, 2.0 * self.mu]])
        self.Bopt = np.array([0, 1, 0]).reshape(-1, 1)

        Q = np.eye(3)
        R = 1

        self.K, S, E = lqr(self.Aopt, self.Bopt, Q, R)

        self.t_nums = t_nums
        self.t_days = t_days
        self.t_span = np.linspace(0, self.t_days, self.t_nums)
        self.cnter = 0
        self.state_init = np.array([[5.],
                                    [5.]
                                    ]).reshape(-1)
        self.state = self.state_init

    def ode_solver(self, u):
        tspan_tmp = [self.t_span[self.cnter], self.t_span[self.cnter + 1]]
        y_tmp = odeint(dynamics, self.state, tspan_tmp, args=(u, self.mu, self.lam,))
        self.y0 = y_tmp[1]
        X = np.array(y_tmp[1, :]).reshape(2, -1)

        return X

    def step(self, u):
        X = self.ode_solver(u)
        self.cnter += 1

        next_state = X.reshape(-1)
        self.state = next_state

        return self.state

    def reset(self):
        self.cnter = 0
        self.state = self.state_init

        return self.state


def train(env, max_ep=500):
    gamma = 1.0 + 5e-4
    lam = 0.9
    min_lam = 0.01
    model = Transformer(in_dim=2, out_dim=3, A_dim=[3, 3])
    optimizer = optim.Adam(model.parameters())
    mse = nn.MSELoss()

    mse1_buf, mse2_buf, mse_total_buf = [], [], []

    for ep in range(max_ep):
        state_non_linear = env.reset()
        loss_mean = 0.0

        mse1_mean = 0.0
        mse2_mean = 0.0
        for t in range(env.t_nums - 1):
            u = -env.K @ np.array([state_non_linear[0], state_non_linear[1], state_non_linear[0] ** 2]).reshape(-1, 1)
            state_non_linear_tensor = torch.from_numpy(state_non_linear).float().view(1, -1)
            state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
            jac = jacobian(model.map524, state_non_linear_tensor, create_graph=True)
            dxdt = dynamics(state_non_linear, None, u[0][0], env.mu, env.lam)
            Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt).float().view(-1, 1))

            B = torch.from_numpy(np.array([[0, 0], [0, 1]])).float()
            u_tensor = torch.from_numpy(np.array([0, u[0][0]])).float().view(-1, 1)
            Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor))

            lam = max(lam / gamma, min_lam)

            mse1 = mse(Jn_dxdt, state_linear_AN.view(-1, 1) + Jn_bu)
            mse2 = mse(state_non_linear_tensor, state_non_linear_aprx)

            mse1_mean += mse1.data.numpy().mean()
            mse2_mean += mse2.data.numpy().mean()

            loss = mse1 + lam * mse2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_mean += loss.data.numpy().mean()

            state_next_non_linear = env.step(u[0][0])
            state_non_linear = state_next_non_linear
        mse1_buf.append(mse1_mean / env.t_nums)
        mse2_buf.append(mse2_mean / env.t_nums)
        mse_total_buf.append(loss_mean / env.t_nums)
        print('ep: {} lam: {:4f} loss1: {:.4f} loss2: {:.4f} loss_total: {:.4f}'.format(
            ep, lam, mse1_mean / env.t_nums, mse2_mean / env.t_nums, loss_mean / env.t_nums))

    return model, mse1_buf, mse2_buf, mse_total_buf


def test(env, model):
    model, mse1_buf, mse2_buf, mse_total_buf = model

    state_non_linear = env.reset()
    state_linear_chainrule_buf, state_linear_linearapx_buf, state_non_linear_buf, state_non_linear_est_buf = [], [], [], []
    A_est = model.Amat.weight.data.numpy()
    print(A_est)
    for t in range(env.t_nums - 1):
        state_non_linear_buf.append(state_non_linear)
        u = -env.K @ np.array([state_non_linear[0], state_non_linear[1], state_non_linear[0] ** 2]).reshape(-1, 1)
        state_non_linear_tensor = torch.from_numpy(state_non_linear).float().view(1, -1)
        state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
        state_non_linear_est_buf.append(state_non_linear_aprx.data.numpy().squeeze())

        # state_linear_buf.append(state_linear.data.numpy())
        jac = jacobian(model.map524, state_non_linear_tensor)
        dxdt = dynamics(state_non_linear, None, u[0][0], env.mu, env.lam)
        Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt).float().view(-1, 1))
        u_tensor = torch.from_numpy(np.array([0, u[0][0]])).float().view(-1, 1)
        B = torch.from_numpy(np.array([[0, 0], [0, 1]])).float()
        Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor))

        state_linear_chainrule_buf.append(Jn_dxdt.data.numpy())
        state_linear_linearapx = state_linear_AN + Jn_bu
        state_linear_linearapx_buf.append(state_linear_linearapx.data.numpy())

        state_next_non_linear = env.step(u[0][0])
        state_non_linear = state_next_non_linear

    state_linear_chainrule_buf = np.array(state_linear_chainrule_buf).squeeze()
    state_linear_linearapx_buf = np.array(state_linear_linearapx_buf).squeeze()
    state_non_linear_buf = np.array(state_non_linear_buf).squeeze()
    state_non_linear_est_buf = np.array(state_non_linear_est_buf).squeeze()

    fig = plt.figure(num=3)

    ax = fig.add_subplot(311)
    ax.plot(state_linear_chainrule_buf[:, 0], color='r', linestyle='--', label='z0-chain')
    ax.plot(state_linear_chainrule_buf[:, 1], color='g', linestyle='--', label='z1-chain')
    ax.plot(state_linear_chainrule_buf[:, 2], color='b', linestyle='--', label='z2-chain')

    ax.plot(state_linear_linearapx_buf[:, 0], color='r', linestyle='-', label='z0-laprx')
    ax.plot(state_linear_linearapx_buf[:, 1], color='g', linestyle='-', label='z1-laprx')
    ax.plot(state_linear_linearapx_buf[:, 2], color='b', linestyle='-', label='z2-laprx')

    ax.legend()

    ax = fig.add_subplot(312)
    ax.plot(state_non_linear_buf[:, 0], color='r', linestyle='--', label='x0')
    ax.plot(state_non_linear_buf[:, 1], color='b', linestyle='--', label='x1')

    ax.plot(state_non_linear_est_buf[:, 0], color='r', linestyle='-', label='x0')
    ax.plot(state_non_linear_est_buf[:, 1], color='b', linestyle='-', label='x1')

    ax.legend()

    ax = fig.add_subplot(313)
    ax.plot(mse1_buf, label='MSE1')
    ax.plot(mse2_buf, label='Trans Err')
    ax.plot(mse_total_buf, label='Total Err')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    env = Environment(t_nums=201, t_days=50)
    model = train(env, max_ep=100)
    test(env, model)

    # inp = torch.eye(4, 5, requires_grad=True)
    # out = (inp + 1).pow(2).t()
    # out.backward(torch.ones_like(out), retain_graph=True)
    # print(f"First call\n{inp.grad}")
    # out.backward(torch.ones_like(out), retain_graph=True)
    # print(f"\nSecond call\n{inp.grad}")
    # inp.grad.zero_()
    # out.backward(torch.ones_like(out), retain_graph=True)
    # print(f"\nCall after zeroing gradients\n{inp.grad}")