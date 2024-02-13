####################################################
# Description: Hyper-Parameters Initial Settings
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-21
####################################################

import time
import argparse
from distutils.util import strtobool


def GetParameters():
    tm = time.localtime()
    month = tm.tm_mon
    day = tm.tm_mday

    parser = argparse.ArgumentParser(description='DARPA RL Configuration')
    # Randomness Related
    parser.add_argument('--seed', default=0, type=int, help='Random Seed Initialization')

    ########################################################################################################
    # Wound Related
    parser.add_argument('--Ra', default=3.0, type=float, help='R: wound size measured in mm')
    parser.add_argument('--Lam', default=0.03, type=float, help='L: in mm')
    parser.add_argument('--Tday', default=1.0, type=float, help='T: in day')
    parser.add_argument('--k1', default=3.0, type=float, help=' ')
    parser.add_argument('--delta', default=1.0 / 3.0, type=float, help=' in day')
    parser.add_argument('--beta', default=1.0, type=float, help=' ')
    parser.add_argument('--rho', default=0.1, type=float, help='k4')
    parser.add_argument('--kapa', default=0.05, type=float, help='k')
    parser.add_argument('--power', default=5.0, type=float, help='q: power')
    parser.add_argument('--gamma1', default=0.1, type=float, help=' ')
    parser.add_argument('--gamma2', default=0.1, type=float, help=' ')
    parser.add_argument('--mu', default=0.2, type=float, help='k4')
    parser.add_argument('--DTilt', default=0.32, type=float, help=' ')
    parser.add_argument('--DTilt_n', default=0.0003, type=float, help=' ')
    parser.add_argument('--alphaTilt', default=1.8, type=float, help=' ')
    # TODO: Check
    parser.add_argument('--k5', default=1.0, type=float, help=' ')

    # Actuator
    parser.add_argument('--t1', default=2.0, type=float, help=' ')
    parser.add_argument('--dt', default=2.0, type=float, help=' ')

    parser.add_argument('--spt', default=False, type=bool, help='whether to do spatio and temporal control')
    parser.add_argument('--cont', default=False, type=bool, help='whether or not to continuous actions')

    parser.add_argument('--action_size', default=501, type=int, help=' ')

    parser.add_argument('--check_opt', default=False, type=bool,
                        help='Whether to use policy from ODE paper, if set to False, use RL ')
    parser.add_argument('--is_train', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='to train or test?')

    parser.add_argument('--n_cells', default=100, type=int, help='Random Seed Initialization')
    parser.add_argument('--X_pump', default=80, type=int,
                        help='Position of ion pump that creates EF, attracting macrophages')
    parser.add_argument('--t_days', default=60, type=int, help='Total number of days')
    parser.add_argument('--t_nums', default=121, type=int, help='Dived t_days into t_nums')
    parser.add_argument('--t_nums_sim', default=601, type=int, help='Dived t_days into t_nums')
    parser.add_argument('--r', default=60, type=int, help='wound position where we exam')
    parser.add_argument('--Tc', default=3.0, type=float, help='Final time needs to divided by Tc')
    #########################################################################################################

    ########################################################################################################
    # In-vivo Device Parameter setting
    parser.add_argument('--n_hour', default=6.0, type=float,
                        help='experimental time')
    parser.add_argument('--n_delay', default=1.2, type=float,
                        help='delay')
    parser.add_argument('--freq_act', default=1.0, type=float,
                        help='frequency of applying voltage')
    parser.add_argument('--n_ch', default=8, type=int,
                        help='number of channels on device to test')
    parser.add_argument('--x_target', default=20, type=int,
                        help='pre-defined target current')

    parser.add_argument('--pre_healnet_dir', default='../../../ExpDataDARPA/res_device/healnet/', type=str,
                        help='directory for loading healnet values')
    parser.add_argument('--cur_device_dir', default='../../../ExpDataDARPA/res_device/Output/', type=str,
                        help='directory for loading current values')
    parser.add_argument('--cur_invio_dir', default='../../../ExpDataDARPA/res_device/data_save/', type=str,
                        help='directory for saving current values')
    parser.add_argument('--vol_invio_dir', default='../../../ExpDataDARPA/res_device/', type=str,
                        help='directory for saving voltage values')
    parser.add_argument('--err_invio_dir', default='../../../ExpDataDARPA/res_device/data_save/', type=str,
                        help='directory for saving error values')
    ########################################################################################################

    # RL Related
    parser.add_argument('--alg_rl', default='a2c', type=str, help='dqn, a2c, ppo, td3')

    parser.add_argument('--nscale', default=3.0, type=float)

    parser.add_argument('--model_dir', default='./res/models/', type=str, help='dqn, a2c, ppo, td3')
    parser.add_argument('--data_dir', default='./res/data/', type=str, help='dqn, a2c, ppo')
    parser.add_argument('--figs_dir', default='./res/figs/', type=str, help='dqn, a2c, ppo')

    parser.add_argument('--n_episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--eps_start', default=1.0, type=float, help='Epsilon Greedy, epsilon start value')
    parser.add_argument('--eps_end', default=0.1, type=float, help='Epsilon Greedy, epsilon end value')
    # eps_decay 0.995
    parser.add_argument('--eps_decay', default=0.999, type=float, help='Epsilon Greedy, epsilon decay rate')

    # close loop related
    parser.add_argument('--ctr', default=False, type=lambda x: bool(strtobool(x)),
                        help='close loop to decide action size')
    parser.add_argument('--cloose_loop', default=True, type=lambda x: bool(strtobool(x)), help='close loop to decide action size')
    parser.add_argument('--decoder_size', default=5, type=int, help='state size of decoder')

    # DQN Related
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--buffer_size', default=int(1e3), type=int, help='replay buffer size')
    parser.add_argument('--TAU', default=1e-3, type=float, help='soft update')
    parser.add_argument('--UPDATE_EVERY', default=4, type=int, help='UPDATE_EVERY')
    parser.add_argument('--GAMMA', default=0.995, type=float, help='discount factor')
    # LR 5e-4
    parser.add_argument('--LR', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--gpu', default=False, type=bool, help='whether to use GPU')

    # A3C Related
    parser.add_argument('--num_worker', default=1, type=int, help='batch size')
    parser.add_argument('--actor_ratio', default=0.5, type=float, help=' ')
    parser.add_argument('--entropy_beta', default=1e-2, type=float, help=' ')

    # PPO related
    parser.add_argument('--eps_clip', default=0.1, type=float, help='clip parameter for PPO: default is 0.2')
    parser.add_argument('--K_epochs', default=5, type=int, help='update NN model for K epochs in one PPO update')
    parser.add_argument('--UPDATE_EVERY_PPO', default=241, type=int, help='Should smaller than 601 ')
    parser.add_argument('--action_std_init', default=0.8, type=float,
                        help='starting std for action distribution (Multivariate Normal), 1.0 good for tracking')
    parser.add_argument('--action_std_decay_rate', default=1e-2, type=float,
                        help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
    parser.add_argument('--min_action_std', default=0.01, type=float,
                        help='minimum action_std (stop decay after action_std <= min_action_std)')
    parser.add_argument('--action_std_decay_freq', default=int(10), type=int,
                        help='action_std decay frequency (in num timesteps)')

    # args = parser.parse_args()
    # a dummy argument to fool ipython
    args, unknown = parser.parse_known_args()

    return args