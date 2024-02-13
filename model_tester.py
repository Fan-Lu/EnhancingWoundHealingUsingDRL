####################################################
# Description: model tester
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-12
####################################################
import os
import time

import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch

from envs.env import WoundEnv
from algs.dqn import Agent_DQN
from algs.a2c import Agent_A2C

from cfgs.config import GetParameters


def mesh_2d_mat(t, r, tdays, radius, vector):
    @np.vectorize
    def value(tval, rval):
        tidx = np.where(t == tval)[0][0]
        ridx = np.where(r == rval)[0][0]
        val = vector[tidx][ridx]
        return val

    valuefun = value(tdays, radius)
    return valuefun


def healing_time(new_tissues):
    wound_radius_list = []
    r = np.linspace(0, 3, 100)
    for t_idx in range(601):
        new_tissue_t_day = new_tissues[t_idx, :]
        rmin = 99
        for ridx in range(100):
            if new_tissue_t_day[ridx] >= 0.95:
                rmin = ridx
                break
        wound_radius = r[rmin]
        wound_radius_list.append(wound_radius)
    return wound_radius_list


def plotsRes(args, model_idx):
    t = np.linspace(0, 60, 601)
    t_ode = t / 6.0 + 1.2

    r = np.linspace(0, 3, 100)
    # states_act0_noRL, states_act1_noRL, states_RL = vectors

    states_act0_noRL_DF = pd.read_csv('./res/data/state_a0_info.csv')
    states_act1_noRL_DF = pd.read_csv('./res/data/state_a1_info.csv')
    states_RL_DF = pd.read_csv('./res/data/state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))
    actions_RL_DF = pd.read_csv('./res/data/action_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    states_act0_noRL_YY1 = states_act0_noRL_DF.values[:, 100 * 1 + 1]
    states_act1_noRL_YY1 = states_act1_noRL_DF.values[:, 100 * 1 + 1]
    states_RL_YY1 = states_RL_DF.values[:, 100 * 1 + 1]

    states_act0_noRL_YY2 = states_act0_noRL_DF.values[:, 100 * 2 + 1]
    states_act1_noRL_YY2 = states_act1_noRL_DF.values[:, 100 * 2 + 1]
    states_RL_YY2 = states_RL_DF.values[:, 100 * 2 + 1]

    states_act0_noRL = states_act0_noRL_DF.values[:, 100 * 4 + 1:]
    states_act1_noRL = states_act1_noRL_DF.values[:, 100 * 4 + 1:]
    states_RL = states_RL_DF.values[:, 100 * 4 + 1:]
    actions_RL = actions_RL_DF.values[:, 1:]

    wound_radius_list_a0 = healing_time(states_act0_noRL)
    wound_radius_list_a1 = healing_time(states_act1_noRL)
    wound_radius_list_rl = healing_time(states_RL)

    fig = plt.figure(figsize=(12, 12), num=4)
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t[1:] / 3.0, actions_RL)
    ax.set_title('RL Policies')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t / 3.0, wound_radius_list_a0, color='g', linestyle='--', label='No Treatment')
    ax.plot(t / 3.0, wound_radius_list_a1, color='r', linestyle='-', label='ODE Treatment')
    ax.plot(t / 3.0, wound_radius_list_rl, color='b', linestyle='-', label='RL Treatment')
    ax.plot(t_ode, wound_radius_list_a0, color='m', linestyle='--', label='Twice of No Treatment')
    ax.set_xlabel(r't, days')
    ax.set_ylabel(r'wound radius, mm')
    ax.set_title('Wound Radius')
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(states_act0_noRL_YY1, states_act0_noRL_YY2, label='No Treatment')
    ax.plot(states_act1_noRL_YY1, states_act1_noRL_YY2, label='ODE Treatment')
    ax.plot(states_RL_YY1, states_RL_YY2, label='RL Treatment')
    ax.set_xlabel('M1')
    ax.set_ylabel('M2')
    ax.legend()
    ax.set_title('Infomation Flow')

    fig.tight_layout()
    plt.savefig('./res/figs/test_r_all_rl_{}_anum_{}_ep{}.pdf'.format(args.alg_rl, args.action_size, model_idx))
    plt.close()


def apply_noRL_treatment(args, action):
    args.check_opt = True
    env = WoundEnv(args)
    state = env.reset()
    state_buf, action_buf = [], []
    k = 0
    state_buf.append(state)
    for k in range(600):
        next_state, reward, done, info = env.step(action)
        print('Time: {} Action: {} New Tissue: {:.2f}'.format(k + 1, action, info[-1]))
        state = next_state
        state_buf.append(state)
    state_buf = np.array(state_buf)
    state_buf_df = pd.DataFrame(state_buf)
    state_buf_df.to_csv('./res/data/state_a{}_info.csv'.format(action))

    return env.t_span[k + 1] / 3.0


def apply_RL_treatment(args, model_idx):
    args.check_opt = False
    env = WoundEnv(args)
    # agent = AgentDQN(env, args)
    agent = Agent_A2C(env, args)

    model_dir = './res/models_a2c/'
    # agent.qnet_local.load_state_dict(torch.load(model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(args.action_size, model_idx)))
    agent.actor_critic.load_state_dict(torch.load(model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(args.action_size, model_idx)))
    state = env.reset()
    state_buf, action_buf = [], []
    state_buf.append(state)
    k = 0
    for k in range(600):
        action = agent.act(state, 0)
        next_state, reward, done, info = env.step(action)
        state = next_state
        state_buf.append(state)
        action_buf.append(action)
        print('Episode: {} Time: {} Action: {:.2f} New Tissue: {:.2f}'.format(model_idx, k + 1, action, info[-1]))

    state_buf = np.array(state_buf)
    state_buf_df = pd.DataFrame(state_buf)
    state_buf_df.to_csv('./res/data/state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    action_buf = np.array(action_buf)
    action_buf_df = pd.DataFrame(action_buf)
    action_buf_df.to_csv('./res/data/action_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    return env.t_span[k + 1] / 3.0


if __name__ == "__main__":
    args = GetParameters()
    # apply_noRL_treatment(args, action=0)
    # apply_noRL_treatment(args, action=1)

    model_visited = []
    model_idx = 5
    stop_cnt = 0
    while model_idx <= args.n_episodes:
        if stop_cnt > 10:
            break
        if model_idx not in model_visited:
            if os.path.isfile('./res/models_a2c/checkpoint_anum_{}_ep_{}.pth'.format(args.action_size, model_idx)):
                model_visited.append(model_idx)
                apply_RL_treatment(args, model_idx)
                plotsRes(args, model_idx)
                model_idx += 5
                stop_cnt = 0
            else:
                time.sleep(1400)
                stop_cnt += 1


    # model_idx = 20
    # apply_RL_treatment(args, model_idx)
    # plots3D(model_idx)

    # for i in range(3, 10):
    #     model_idx = i * 5
    #     apply_RL_treatment(args, model_idx)
    #     plots3D(model_idx)