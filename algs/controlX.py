####################################################
# Description: Sliding Mode Controller
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-09-07
####################################################
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from matplotlib.lines import Line2D

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
def animate(i, xs, ys):
    ax.clear()
    ax.plot(xs, ys)
    # plt.xticks(rotation=45, ha='right')
    # plt.subplots_adjust(bottom=0.30)
    # plt.title('TMP102 Temperature over Time')
    # plt.ylabel('Temperature (deg C)')


class Agent_Control(object):

    def __init__(self, x_target, n_ch=8):
        '''

        @param x_target: predefined target current
        @param n_ch: number of channels
        '''

        # bio params
        # number of channels
        self.n_ch = n_ch
        # predefined target current
        self.x_target = np.array([x_target for _ in range(n_ch)])

        # hyer parameters for smc for each channle
        self.k_gain = 0.4
        self.r_gain = 0.008
        self.sign = -1.0

        # max voltage constraints
        self.vol_max = 3.3
        # min voltage constraints
        self.vol_min = 0.0

        # sampling time: one sec
        self.ts = 1.0

        # initialization
        self.x_pre = np.zeros(self.n_ch)
        self.x_target_pre = self.x_target * np.ones(self.n_ch)
        self.init_input_ch_pre = np.zeros(self.n_ch)
        self.init_artificail_ch_pre = np.zeros(self.n_ch)

    def dynamics(self, u):
        '''
        Fake dynamics for testing
        '''
        # x_cur = self.x_pre + u * self.ts
        # TODO: what are the maximum resistent
        x_cur = u / 0.1

        return x_cur

    def smc(self, x_curr, x_target):
        '''
        Sliding Mode Controller

        :param x_curr: current state
        :param x_target: target state
        :return: u_app_ch: control input
                 error:    states error from target
        '''
        error = x_curr - x_target
        s_ch = self.k_gain * error + ((x_curr - self.x_pre) / self.ts
                                      - ((x_target - self.x_target_pre) / self.ts))
        init_artificail_ch = self.sign * self.r_gain * np.sign(s_ch * self.vol_max * np.cos(self.init_input_ch_pre))
        init_input_ch = self.init_input_ch_pre + (self.init_artificail_ch_pre + init_artificail_ch) / 2.0 * self.ts
        u_sat_ch = self.vol_max * np.sin(init_input_ch)
        u_app_ch = np.clip(u_sat_ch, self.vol_min, self.vol_max)

        print('CurC: {:.2f} VolO: {:.2f} Err: {:.2f}'.format(x_curr[0], u_app_ch[0], error[0]))

        self.x_target_pre = x_target
        self.init_input_ch_pre = init_input_ch
        self.init_artificail_ch_pre = init_artificail_ch
        self.x_pre = x_curr

        return u_app_ch, error

    def karman(self):
        '''
        karman filtering
        '''

        return

    def cpid(self, x_cur, x_target):
        '''
        Cascaded PID
        '''
        return


class Agent_Actuator(object):

    def __init__(self, args):
        '''

        @param args: arguments that contains all the hyperparameters
        '''

        self.args = args

        runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + 'exp_on_device'
        runs_dir = '../../../ExpDataDARPA/runs_ondevice/{}'.format(runs_dir)
        os.makedirs(runs_dir)
        self.writer = SummaryWriter(log_dir=runs_dir)

        # delay
        self.n_delay = args.n_delay
        # total number of acts to take in n_hour range
        self.n_act = int(self.args.n_hour * 3600 / self.args.freq_act)
        self.controller = Agent_Control(self.args.x_target, self.args.n_ch)

        # TODO: check in calibration
        self.vol_max = np.zeros(self.args.n_ch)

        self.fail_ch = []

        # Time buffer
        self.time_vector_sec = [0.]   # seonds
        self.time_vector_min = [0.]   # minutes
        self.time_vector_hro = [0.]   # hours

        self.x_curr_buf = np.zeros([1, self.args.n_ch])
        self.x_targ_buf = np.zeros([1, self.args.n_ch])
        self.x_targ_buf[0, :] = self.args.x_target
        self.u_volt_buf = np.zeros([1, self.args.n_ch])
        self.e_eror_buf = np.zeros([1, self.args.n_ch])
        self.p_woud_buf = np.zeros([1, 4])
        self.p_woud_buf[0, 0] = 1.0

        self.flx_total = [0.0]
        self.flag_check = False

        # saving the voltages on the file to be read by Prabhat
        table_voltage = {'t(s)': [0],
                         'V1(v)': [0], 'V2(v)': [0], 'V3(v)': [0], 'V4(v)': [0],
                         'V5(v)': [0], 'V6(v)': [0], 'V7(v)': [0], 'V8(v)': [0]
                         }
        # loading the current from the file to be used for controller
        table_current = {'t(s)': [0],
                         'I1(v)': [0], 'I2(v)': [0], 'I3(v)': [0], 'I4(v)': [0],
                         'I5(v)': [0], 'I6(v)': [0], 'I7(v)': [0], 'I8(v)': [0]
                         }
        # saving errors of all the channales
        table_currerr = {'t(s)': [0],
                         'error1(A)': [0], 'error2(A)': [0], 'error3(A)': [0], 'error4(A)': [0],
                         'error5(A)': [0], 'error6(A)': [0], 'error7(A)': [0], 'error8(A)': [0]
                        }
        # create pandas data frame for salving and loading data
        df_voltage = pd.DataFrame(table_voltage)
        df_current = pd.DataFrame(table_current)
        df_currerr = pd.DataFrame(table_currerr)

        dirs = [self.args.cur_invio_dir, self.args.vol_invio_dir, self.args.err_invio_dir]
        for dir_tmp in dirs:
            if not os.path.exists(dir_tmp):
                os.makedirs(dir_tmp)
        df_voltage.to_csv(self.args.vol_invio_dir + 'Wound_1.csv')
        df_current.to_csv(self.args.cur_invio_dir + 'Wound_1.csv')
        df_currerr.to_csv(self.args.err_invio_dir + 'error_1.csv')

    def loader(self, name):
        '''
        read curr from csv file
        @param name: file name to load
        @return: data frame of name values
        '''
        read_flag = False
        df_tmp = None
        while not read_flag:
            # try if we can read the file that contains current values
            try:
                if name == 'current':
                    df_tmp = pd.read_csv(self.args.cur_device_dir + 'Wound_1.csv')
                elif name == 'healnet':
                    df_tmp = pd.read_csv(self.args.pre_healnet_dir + 'prob_table.csv')
                else:
                    assert False, 'Please specify the name of files needed to load!!!'
                read_flag = True
            # non-readable? wait for 0.1 sec
            except:
                time.sleep(0.1)
                # TODO: What if file keeps being unavailable?
                print("The file was not readable, try again to read the file")
        return df_tmp

    def saver(self, x_curr, u_app_ch, x_error):
        '''

        @param x_curr:
        @param u_app_ch:
        @param x_error:
        @param start_time:
        @return:
        '''
        # total time in hours
        time_vector_hour = (time.time() - self.start_time) / 3600

        # TODO: Check here
        time_table = 5
        table_voltage = {'t(s)': [time_table],
                         'V1(v)': [u_app_ch[0]], 'V2(v)': [u_app_ch[1]], 'V3(v)': [u_app_ch[2]], 'V4(v)': [u_app_ch[3]],
                         'V5(v)': [u_app_ch[4]], 'V6(v)': [u_app_ch[5]], 'V7(v)': [u_app_ch[6]], 'V8(v)': [u_app_ch[7]]
                         }
        time_table = time_vector_hour
        table_current = {'t(s)': [time_table],
                         'I1(v)': [x_curr[0]], 'I2(v)': [x_curr[1]], 'I3(v)': [x_curr[2]], 'I4(v)': [x_curr[3]],
                         'I5(v)': [x_curr[4]], 'I6(v)': [x_curr[5]], 'I7(v)': [x_curr[6]], 'I8(v)': [x_curr[7]]
                         }
        time_table = time_vector_hour
        table_currerr = {'t(s)': [time_table],
                         'I1(v)': [x_error[0]], 'I2(v)': [x_error[1]], 'I3(v)': [x_error[2]], 'I4(v)': [x_error[3]],
                         'I5(v)': [x_error[4]], 'I6(v)': [x_error[5]], 'I7(v)': [x_error[6]], 'I8(v)': [x_error[7]]
                         }
        df_voltage = pd.DataFrame(table_voltage)
        df_current = pd.DataFrame(table_current)
        df_currerr = pd.DataFrame(table_currerr)

        # print('Voltage: {} \n Current: {} \n Err: {} \n'.format(u_app_ch, x_curr, x_error))

        df_voltage.to_csv(self.args.vol_invio_dir + 'Wound_1.csv', mode='a', index=True, header=False)
        df_current.to_csv(self.args.cur_invio_dir + 'Wound_1.csv', mode='a', index=True, header=False)
        df_currerr.to_csv(self.args.err_invio_dir + 'error_1.csv', mode='a', index=True, header=False)

    def calibration(self, u_app_ch, error):
        '''

        @param u_app_ch:
        @param error:
        @return:
        '''
        for i_ch in range(self.args.n_ch):
            # TODO: diferent channels has different gains
            if abs(error[i_ch]) <= 0.5:
                self.controller.r_gain = 0.008
            elif error[i_ch] > 0.5:
                self.controller.r_gain = 0.08
            else:
                self.controller.r_gain = 0.008

            # TODO: diferent channels has voltage constraints
            # voltage constraints
            if self.vol_max[i_ch] < 1 and u_app_ch[i_ch] > 3.25:
                self.vol_max[i_ch] += 1.0
            elif self.vol_max[i_ch] >= 0 and u_app_ch[i_ch] > 3.25:
                # time_table = 40
                u_app_ch[i_ch] = 1.5
                self.vol_max[i_ch] += 1
            else:
                self.vol_max[i_ch] = 0

        return u_app_ch

    def failure_check(self, current, voltage, u_app_ch, x_target, timer):
        # assign False to those channel that not work
        # senario 1: we need current in channels, but current small even though we apply voltage

        if timer % 20 > 19:
            s1 = np.multiply(np.multiply(current < 1e-5, voltage > 0.1), x_target > 0)
            ch_didi = 1 - s1
            fail_tmp = list(np.where(ch_didi == 0)[0])
            self.fail_ch = fail_tmp
        else:
            s1 = np.multiply(np.multiply(current < 1e-5, voltage > 0.1), x_target > 0)
            ch_didi = 1 - s1
            fail_tmp = list(np.where(ch_didi == 0)[0])
            self.fail_ch.extend(fail_tmp)
            self.fail_ch = list(set(self.fail_ch))
            # self.ref_ch = list(range(self.args.n_ch)).remove(self.fail_ch)
            if len(self.fail_ch) > 5:
                self.fail_ch = fail_tmp
        self.ref_ch = list(range(self.args.n_ch))
        for fch in self.fail_ch:
            self.ref_ch.remove(fch)
        self.ref_ch = []
        for lp in range(2):
            for idx in range(self.args.n_ch):
                if idx not in self.fail_ch and idx not in self.ref_ch:
                    self.ref_ch.append(idx)
                    break

        # assign new references
        # self.new_ref = [0, 1]
        # self.ch_didi[self.new_ref] = 0
        print(self.fail_ch, self.ref_ch)
        # voltage = np.multiply(voltage, self.ch_didi)
        for idx in self.fail_ch:
            x_target[idx] = 0
            u_app_ch[idx] = 0
        for idx in self.ref_ch:
            x_target[idx] = 0
            u_app_ch[idx] = 0

        return u_app_ch, x_target

    def plot(self, t_idx):
        '''

        @param t_idx:
        @return:
        '''
        fig = plt.figure(figsize=(14, 10), num=6)
        ax6 = fig.add_subplot(321)
        ax5 = fig.add_subplot(322)
        ax1 = fig.add_subplot(323)
        ax2 = fig.add_subplot(324)
        ax3 = fig.add_subplot(325)
        ax4 = fig.add_subplot(326)
        chs = {0: [3.5, 3], 1: [3.5, 7], 2: [2.5, 5], 3: [5, 2],
               4: [7.5, 5], 5: [5, 8], 6: [6.5, 3], 7: [6.5, 7], 8: [5, 5]}
        for i in range(8):
            j, k = chs[i]
            if i in self.fail_ch:
                cc = 'k'
            elif i in self.ref_ch:
                cc = 'b'
            else:
                cc = 'r'
            ax6.scatter(j, k, marker="o", s=300, c=cc, alpha=1.0)
        ax6.scatter(chs[8][0], chs[8][1], marker="o", s=300, c='b', alpha=1.0)
        custom_lines = [Line2D([0], [0], color='r', lw=4),
                        Line2D([0], [0], color='k', lw=4),
                        Line2D([0], [0], color='b', lw=4)]
        ax6.legend(custom_lines, ['Working', 'Failed', 'Refer'], loc='center left', bbox_to_anchor=(1, 0.5))

        ax1.plot(self.x_targ_buf[:, 0], linestyle='-.', label='Target')
        for c_idx in range(self.args.n_ch):
            ax1.plot(self.x_curr_buf[:, c_idx], linestyle='-', label='Ch_{}'.format(c_idx))
            ax2.plot(self.u_volt_buf[:, c_idx], label='Ch_{}'.format(c_idx))
            ax3.plot(self.e_eror_buf[:, c_idx], label='Ch_{}'.format(c_idx))

        ax5.plot(self.flx_total, c='r', label='CuCu')
        ax5.set_ylabel('cumulated current', color='r')
        ax5.fill_between(range(len(self.flx_total)), 0, self.flx_total, where= 0 <= np.array(self.flx_total),
                         facecolor='red', interpolate=True, alpha=0.1)
        ax5.tick_params(axis='y', labelcolor='r')

        ax4.plot(self.p_woud_buf[:, 0], label='Hemostasis')
        ax4.plot(self.p_woud_buf[:, 1], label='Inflammation')
        ax4.plot(self.p_woud_buf[:, 2], label='Proliferation')
        ax4.plot(self.p_woud_buf[:, 3], label='Maturation')

        ax1.set_ylabel('Current, mA')
        ax2.set_ylabel('Voltage, V')
        ax3.set_ylabel('Error from target current')
        ax4.set_ylabel('Variables values')

        # Shrink current axis by 20%
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        for idx, ax in enumerate(axes):
            if idx < 5:
                ax.set_xlabel('time, sec')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis

        plt.savefig('../../../ExpDataDARPA/figs/smc_{}.png'.format(t_idx), format='png')
        self.writer.add_figure('SMC Controller', fig, t_idx)
        if not os.path.exists('../../../ExpDataDARPA/figs/'):
            os.makedirs('../../../ExpDataDARPA/figs/')

        # plt.savefig('../../../ExpDataDARPA/figs/smc_{}.pdf'.format(t_idx), format='pdf')
        # plt.savefig('../../../ExpDataDARPA/figs/smc_{}.png'.format(t_idx), format='png')
        plt.close()

    def act(self):
        '''

        @return:
        '''
        self.start_time = time.time()
        x_target = self.controller.x_target
        # we first read the predidtions of healnet
        df_healnet_old = self.loader(name='healnet')
        healnet_len_old = len(df_healnet_old.index)

        for t_idx in range(1, self.n_act):
            run_time = (time.time() - self.start_time) / 3600
            if run_time > self.args.n_hour:
                print("The code has been runing for {:.4f} hrs...".format(run_time))
                break

            time.sleep(self.args.freq_act - 0.5)
            # TODO: what if file reading time is two long
            df_current = self.loader(name='current')
            df_healnet = self.loader(name='healnet')

            # TODO: Add HealNet to Tensorboard
            # current length of healnet predictions
            healnet_len = len(df_healnet.index)
            # flag for wheather proliferation has reached certain value
            pro_flag = False
            # if we have new data coming from healnet
            if healnet_len > healnet_len_old:
                for j in range(healnet_len - healnet_len_old):
                    title_df = df_healnet.iat[healnet_len_old + j, 0]
                    # get prediction of maturation
                    mat = df_healnet.iat[healnet_len_old + j, 4]
                    # get the proliferation prediction
                    pro = df_healnet.iat[healnet_len_old + j, 5]
                    if "wound_1" in title_df.lower():
                        # TODO: np.any(r == 5)?
                        if pro > 0.75:
                            x_target = 30.0 * np.ones(self.args.n_ch)
                            pro_flag = True
                        if pro > 0.95:
                            x_target = 0.0 * np.ones(self.args.n_ch)
                            pro_flag = True

            # this was the hardware team request to change the value of target after 3 hours
            cur_time_dur = (time.time() - self.start_time) / 3600.0
            if cur_time_dur > 3 and not pro_flag:
                # TODO: This is suspicious
                # new refrence if the healnet value passes 0.75
                x_target = 20.0 * np.ones(self.args.n_ch)
                for idx in self.fail_ch:
                    x_target[idx] = 0

            # update length of healnet predictions
            healnet_len_old = healnet_len

            # read current of all channels
            x_curr = df_current.values[-1, 1:9]
            # read corresponding voltage of all channels
            v_curr = df_current.values[-1, 9:17]
            # cumulate the current
            self.flx_total.append(self.flx_total[-1] + np.sum(x_curr))

            # SMC control
            u_app_ch, x_error = self.controller.smc(x_curr, x_target)
            # check failure
            u_app_ch, x_target = self.failure_check(x_curr, v_curr, u_app_ch, x_target, cur_time_dur * 3600)
            # self.controller.x_pre = x_curr
            # u_app_ch = self.calibration(u_app_ch, x_error)
            x_target_c = x_target * np.ones(self.args.n_ch)

            # print('CurC: {:.2f} CurT: {:.2f} VolO: {:.2f} Err: {:.2f}'.format(x_curr[0], x_target_c[0], u_app_ch[0], x_error[0]))

            wound_p = df_healnet.values[-1, 3:]
            self.p_woud_buf = np.vstack((self.p_woud_buf, wound_p.reshape(1, 4)))
            self.x_curr_buf = np.vstack((self.x_curr_buf, x_curr.reshape(1, self.args.n_ch)))
            self.x_targ_buf = np.vstack((self.x_targ_buf, x_target_c.reshape(1, self.args.n_ch)))
            self.u_volt_buf = np.vstack((self.u_volt_buf, u_app_ch.reshape(1, self.args.n_ch)))
            self.e_eror_buf = np.vstack((self.e_eror_buf, x_error.reshape(1, self.args.n_ch)))

            # save all the data to csv
            self.saver(x_curr, u_app_ch, x_error)
            self.time_vector_sec.append((time.time() - self.start_time))
            self.time_vector_min.append((time.time() - self.start_time) / 60)
            self.time_vector_hro.append((time.time() - self.start_time) / 3600)
            self.plot(t_idx)


if __name__ == "__main__":
    x_buf = np.zeros((0, 8))
    u_buf = np.zeros((0, 8))
    x_t_buf = []

    cont = Agent_Control(x_target=4.0)
    target = 4.0
    x_cur = np.zeros(8)

    for t in range(500):
        if t < 100:
            target = 2.0
        elif t < 200:
            target = 4.0
        elif t < 300:
            target = 6.0
        else:
            target = 4.0
        x_t_buf.append(target)
        u_app, err = cont.smc(x_cur, target)
        # read state after applying input
        x_next = cont.dynamics(u_app)

        cont.x_pre = x_cur
        x_cur = x_next
        x_buf = np.vstack((x_buf, cont.x_pre.reshape(1, -1)))
        u_buf = np.vstack((u_buf, u_app.reshape(1, -1)))

        # print('State: {:.2f} Input: {:.2f} err: {:.2f}'.format(cont.x_pre, u_app, err))

    fig = plt.figure(figsize=(12, 4), num=2)

    ax = fig.add_subplot(121)
    ax.plot(u_buf)
    ax.set_xlabel('time, sec')
    ax.set_ylabel('input voltage')

    ax = fig.add_subplot(122)
    ax.plot(x_t_buf, label='target x')
    ax.plot(x_buf[:, 0], label='real x')
    ax.legend()
    ax.set_xlabel('time, sec')
    ax.set_ylabel('current')

    plt.tight_layout()
    plt.show()

