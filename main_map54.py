####################################################
# Description: trainer of different algorithm
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-10-04
###################################################


import os
import random
import time
import sys

import numpy as np
import pandas as pd
from collections import deque

import matplotlib
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from algs.ode54 import Transformer
from algs.a2c import Agent_A2C
from algs.dqn import Agent_DQN
from envs.env import WoundEnv, SimpleEnv, LinearEnv, dynamics, linear_dynamic
from cfgs.config import GetParameters


def train(colab_dir, max_ep=500, args=None):
    alg_name = 'map524'
    args.model_dir = colab_dir + '/res_map524/models_{}/'.format(alg_name)
    args.data_dir = colab_dir + '/res_map524/data_{}/'.format(alg_name)
    args.figs_dir = colab_dir + '/res_map524/figs_{}/'.format(alg_name)

    dirs = [args.model_dir, args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + '_alg_' + alg_name
    runs_dir = args.model_dir + '../../runs_{}/{}'.format(alg_name, runs_dir)
    os.makedirs(runs_dir)

    writer = SummaryWriter(log_dir=runs_dir)

    gamma = 1.0 + 5e-4
    lam = 0.9
    min_lam = 0.01

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    model = Transformer(in_dim=5, out_dim=4).to(device)
    model.AMask = model.AMask.to(device)
    model.outMap = model.outMap.to(device)
    optimizer = optim.Adam(model.parameters())
    mse = nn.MSELoss()

    eps = args.eps_start
    lenv = LinearEnv(args)
    nenv = WoundEnv(args)


    # agent = Agent_DQN(nenv, args)

    agent = Agent_A2C(nenv, args)

    for i_episode in range(max_ep):
        state_non_linear = nenv.reset()
        state_linear_env = lenv.reset()

        loss_mean, mse1_mean, mse2_mean, reward_mean = 0.0, 0.0, 0.0, 0.0
        lam = max(lam / gamma, min_lam)
        for t in range(nenv.t_nums - 1):

            state_non_linear_5 = state_non_linear.reshape(5, args.n_cells)[:, 0]
            state_non_linear_tensor = torch.from_numpy(state_non_linear_5).float().to(device).view(1, -1)
            # state_non_linear_tensor[:, 3] /= 3.0

            # selection action based using DRL
            agent_state = state_non_linear_tensor.cpu().data.numpy().squeeze()
            u = agent.act(agent_state, eps) if args.ctr else 0

            state_next_non_linear, reward_2_abandon, done_2_abandon, info = nenv.step(u)

            state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
            jac = jacobian(model.map524, state_non_linear_tensor, create_graph=True)

            n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = args.n_cells, args.spt, args.X_pump, args.beta, args.gamma1, args.gamma2, args.rho, args.mu, args.alphaTilt, args.power, args.kapa, args.Lam, args.DTilt, args.DTilt_n
            arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
            dxdt = dynamics(state_non_linear, None, u, arrgs)

            dxdt_5 = dxdt.reshape(5, args.n_cells)[:, 0]
            Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt_5).float().to(device).view(-1, 1)).view(1, -1)

            # m1 macrophage
            m1 = state_non_linear_tensor.cpu().data.numpy()[:, 1][0]
            B = torch.from_numpy(np.array([[0,  0, 0, 0, 0],
                                           [0, -m1, 0, 0, 0],
                                           [0,  m1, 0, 0, 0],
                                           [0,  0, 0, 0, 0],
                                           [0,  0, 0, 0, 0]])).float().to(device)
            u_tensor = torch.from_numpy(np.array([u, u, u, u, u])).float().to(device).view(-1, 1)
            Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor)).view(1, -1)

            AA = model.Amat_masked.cpu().data.numpy()
            BB = torch.matmul(jac[0, :, 0, :], B).cpu().data.numpy()

            state_next_linear_env = lenv.step((AA, BB))
            state_next_non_linear_ref = model.map425(torch.from_numpy(state_next_linear_env).float().to(device).view(1, -1))
            state_next_non_linear_ref = state_next_non_linear_ref.cpu().data.numpy()

            agent_state_next = state_next_non_linear.reshape(5, args.n_cells)[:, 0]
            # agent_state_next[3] /= 3.0
            reward = np.exp(-np.linalg.norm(state_next_non_linear_ref - agent_state_next))
            agent.step(agent_state, u, reward, agent_state_next, False)

            react = nenv.theta_space[u]

            mse1 = mse(Jn_dxdt, state_linear_AN + Jn_bu)
            mse2 = mse(state_non_linear_tensor, state_non_linear_aprx)
            mse3 = mse(state_linear[:, :3], state_non_linear_tensor[:, :3])
            # loss = mse1 + lam * (mse2)
            loss = mse1 + lam * (mse2 + 0.9 * mse3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse1_mean += mse1.cpu().data.numpy().mean()
            mse2_mean += mse2.cpu().data.numpy().mean()
            loss_mean += loss.cpu().data.numpy().mean()
            reward_mean += reward

            # state_next_non_linear, reward, done, info = env.step(u)
            state_non_linear = state_next_non_linear
        eps = max(args.eps_end, args.eps_decay * eps)  # decrease epsilon
        writer.add_scalar('Loss/train_mse1', mse1_mean / nenv.t_nums, i_episode)
        writer.add_scalar('Loss/train_mse2', mse2_mean / nenv.t_nums, i_episode)
        writer.add_scalar('Loss/train_mse_total', loss_mean / nenv.t_nums, i_episode)
        writer.add_scalar('DRL/Reward', reward_mean / nenv.t_nums, i_episode)
        writer.add_scalar('DRL/Eps', eps, i_episode)
        writer.add_scalar('Loss/Lamda', lam, i_episode)

        print('\r TrainEp: {} \t lam: {:4f} loss1: {:.4f} loss2: {:.4f} loss_total: {:.4f}'.format(
            i_episode, lam, mse1_mean / nenv.t_nums, mse2_mean / nenv.t_nums, loss_mean / nenv.t_nums), end="")
        if (i_episode+1) % 5 == 0:
            test(colab_dir, device, (model, agent), writer, i_episode, args)
            torch.save(model.state_dict(),
                       args.model_dir + 'checkpoint_ctr_{}_ep_{}.pth'.format(args.ctr, i_episode))
            torch.save(agent.model.state_dict(),
                       args.model_dir + 'checkpoint_ctr_{}_ep_{}.pth'.format(args.ctr, i_episode))

    return model, agent


def test(colab_dir, device, models, writer, i_episode, args):

    nenv = WoundEnv(args)

    state_non_linear = nenv.reset()
    state_linear_chainrule_buf, state_linear_linearapx_buf, state_non_linear_buf, state_non_linear_est_buf = [], [], [], []
    cstate_linear_buf = []
    action_buf = []

    model, agent = models
    # print(model.Amat_masked.cpu().data.numpy())

    for t in range(nenv.t_nums - 1):
        state_non_linear_buf.append(state_non_linear.reshape(5, args.n_cells))
        state_non_linear_5 = state_non_linear.reshape(5, args.n_cells)[:, 0]
        state_non_linear_tensor = torch.from_numpy(state_non_linear_5).float().to(device).view(1, -1)
        # state_non_linear_tensor[:, 3] /= 3.0

        # selection action based using DRL
        u = agent.act(state_non_linear_tensor.cpu().data.numpy().squeeze()) if args.ctr else 0
        state_next_non_linear, reward_2_abandon, done_2_abandon, info = nenv.step(u)

        state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
        state_non_linear_est_buf.append(state_non_linear_aprx.cpu().data.numpy().reshape(5, -1))

        jac = jacobian(model.map524, state_non_linear_tensor, create_graph=True)

        n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = args.n_cells, args.spt, args.X_pump, args.beta, args.gamma1, args.gamma2, args.rho, args.mu, args.alphaTilt, args.power, args.kapa, args.Lam, args.DTilt, args.DTilt_n
        arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
        dxdt = dynamics(state_non_linear, None, u, arrgs)
        dxdt_5 = dxdt.reshape(5, args.n_cells)[:, 0]
        Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt_5).float().to(device).view(-1, 1)).view(1, -1)

        # m1 macrophage
        v = state_non_linear_tensor.cpu().data.numpy()[:, 1][0]
        B = torch.from_numpy(np.array([[0, 0, 0, 0, 0],
                                       [0, -v, 0, 0, 0],
                                       [0, v, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]])).float().to(device)
        u_tensor = torch.from_numpy(np.array([u, u, u, u, u])).float().to(device).view(-1, 1)
        Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor)).view(1, -1)

        react = nenv.theta_space[u]
        action_buf.append(react)

        cstate_linear_buf.append(state_linear.cpu().data.numpy())
        state_linear_chainrule_buf.append(Jn_dxdt.cpu().data.numpy())
        state_linear_linearapx = state_linear_AN + Jn_bu
        state_linear_linearapx_buf.append(state_linear_linearapx.cpu().data.numpy())

        state_non_linear = state_next_non_linear

    cstate_linear_buf = np.array(cstate_linear_buf).squeeze()
    state_linear_chainrule_buf = np.array(state_linear_chainrule_buf).squeeze()
    state_linear_linearapx_buf = np.array(state_linear_linearapx_buf).squeeze()
    state_non_linear_buf = np.array(state_non_linear_buf).squeeze()
    state_non_linear_est_buf = np.array(state_non_linear_est_buf).squeeze()
    action_buf = np.array(action_buf).squeeze()
    trange = (nenv.t_span / 3.0)[:-1]

    im_scal = 0.8
    leg_pos = (1, 0.5)
    fig = plt.figure(figsize=(8, 12), num=4)
    plt.tight_layout()

    ax = fig.add_subplot(411)
    ax.plot(trange, action_buf, label='Actualtion')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(412)
    ax.plot(trange, cstate_linear_buf[:, 0], color='r', linestyle='--', label=r'$H-trans$')
    ax.plot(trange, cstate_linear_buf[:, 1], color='g', linestyle='--', label=r'$I-trans$')
    ax.plot(trange, cstate_linear_buf[:, 2], color='b', linestyle='--', label=r'$P-trans$')
    ax.plot(trange, cstate_linear_buf[:, 3], color='y', linestyle='--', label=r'$M-trans$')
    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(413)
    ax.plot(trange, state_linear_chainrule_buf[:, 0], color='r', linestyle='--', label=r'$\dot{H}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 1], color='g', linestyle='--', label=r'$\dot{I}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 2], color='b', linestyle='--', label=r'$\dot{P}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 3], color='y', linestyle='--', label=r'$\dot{M}-chain$')

    ax.plot(trange, state_linear_linearapx_buf[:, 0], color='r', linestyle='-', label=r'$\dot{H}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 1], color='g', linestyle='-', label=r'$\dot{I}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 2], color='b', linestyle='-', label=r'$\dot{P}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 3], color='y', linestyle='-', label=r'$\dot{M}-laprx$')

    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(414)
    ax.plot(trange, state_non_linear_buf[:, 0, 0], color='r', linestyle='--', label='Debris-ref')
    ax.plot(trange, state_non_linear_buf[:, 1, 0], color='g', linestyle='--', label='M1-ref')
    ax.plot(trange, state_non_linear_buf[:, 2, 0], color='b', linestyle='--', label='M2-ref')
    ax.plot(trange, state_non_linear_buf[:, 3, 0] / 3.0, color='y', linestyle='--', label='Temp-ref')
    ax.plot(trange, state_non_linear_buf[:, 4, 0], color='c', linestyle='--', label='New-ref')

    ax.plot(trange, state_non_linear_est_buf[:, 0], color='r', linestyle='-', label='Debris-est')
    ax.plot(trange, state_non_linear_est_buf[:, 1], color='g', linestyle='-', label='M1-est')
    ax.plot(trange, state_non_linear_est_buf[:, 2], color='b', linestyle='-', label='M2-est')
    ax.plot(trange, state_non_linear_est_buf[:, 3] / 3.0, color='y', linestyle='-', label='Temp-est')
    ax.plot(trange, state_non_linear_est_buf[:, 4], color='c', linestyle='-', label='New-est')

    ax.set_xlabel('time, day')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    # plt.show()
    writer.add_figure('Wound/Test_ctr_{}'.format(args.ctr), fig, i_episode)
    plt.close()


if __name__ == "__main__":
    args = GetParameters()
    args.gpu = False
    colab_dir = "../../../ExpDataDARPA/"
    train(colab_dir, max_ep=10000, args=args)






