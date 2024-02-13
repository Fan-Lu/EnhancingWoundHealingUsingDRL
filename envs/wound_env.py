import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import metrics


class WoundEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def model_act(z, t, beta, g1, g2, kr, alpha, k, k4, D, D1, h, act_on=0, power=5, n_cells=100, t1=2, dt=4):

        def ion_concentration(loc_x):
            amplitude = 1
            n = 1.25
            x_z = loc_x / 10
            return amplitude * np.exp(-((x_z ** 2) / n))

        def actuator(t, t1, dt):
            if t > t1 and t < t1 + dt:
                return 1
            elif t > t1 and t < t1 + 1:
                return t - t1
            elif t > t1 + dt - 1 and t < t1 + dt:
                return t1 + dt - t
            else:
                return 0

        amplitude = 1
        # act_on=1 actuator on (1) or off
        X_pump = 80
        t0 = 2
        a_l = np.empty(n_cells)
        m_l = np.empty(n_cells)
        u_l = np.empty(n_cells)
        c_1 = np.empty(n_cells)
        n_l = np.empty(n_cells)
        tetta = np.zeros(n_cells)
        for i in range(n_cells):
            #a_f = actuator(t, t1, dt)
            i_f = ion_concentration(i - X_pump)
            tetta[i] = i_f
        for i in range(n_cells):
            a, m, u, c, n = z[i], z[i + n_cells], z[i + 2 * n_cells], z[i + 3 * n_cells], z[
                i + 4 * n_cells]  # m1=m , m2=u
            dadt = -a * m
            dmdt = beta * a - a * m - g1 * m - k4 * ((m ** power) / ((k ** power) + (m ** power))) - act_on * (
                    tetta[i] * m)  # k4=
            # TODO: Sign Error gamma2
            dudt = k4 * ((m ** power) / ((k ** power) + (m ** power))) + g2 * u + act_on * (tetta[i] * m)
            dcdt = u - kr * c
            dndt = c * (alpha * n * (1 - n))
            a_l[i] = dadt
            m_l[i] = dmdt
            u_l[i] = dudt
            c_1[i] = dcdt
            n_l[i] = dndt

        m_l[0] = m_l[0] + D * (z[1 + n_cells] - z[n_cells]) / (h * h)
        u_l[0] = u_l[0] + D * (z[1 + 2 * n_cells] - z[2 * n_cells]) / (h * h)

        n_l[0] = n_l[0] + D1 * (z[3 * n_cells] * (z[1 + 4 * n_cells] - z[4 * n_cells])) / (h * h)

        for i in range(1, n_cells - 1):
            m_l[i] = m_l[i] + D * (z[i - 1 + n_cells] - 2 * z[i + n_cells] + z[i + 1 + n_cells]) / (h * h)
            u_l[i] = u_l[i] + D * (z[i - 1 + 2 * n_cells] - 2 * z[i + 2 * n_cells] + z[i + 1 + 2 * n_cells]) / (
                    h * h)

            n_l[i] = n_l[i] + D1 * (z[i + 3 * n_cells] * (
                    z[i - 1 + 4 * n_cells] - 2 * z[i + 4 * n_cells] + z[i + 1 + 4 * n_cells])) / (h * h)
        n_c = n_cells - 1
        m_l[n_c] = m_l[n_c] + D * (z[n_c - 1 + n_cells] - z[n_c + n_cells]) / (h * h)
        u_l[n_c] = u_l[n_c] + D * (z[n_c - 1 + 2 * n_cells] - z[n_c + 2 * n_cells]) / (h * h)

        n_l[n_c] = n_l[n_c] + D1 * (
                z[n_c + 3 * n_cells] * (z[n_c - 1 + 4 * n_cells] - 2 * z[n_c + 4 * n_cells] + 1)) / (h * h)
        dzdt = np.concatenate([a_l, m_l, u_l, c_1, n_l], axis=0)
        return dzdt

    def __init__(self):
        global np
        from sklearn.svm import SVR
        self.svr_rbf = SVR(kernel="rbf", C=100, gamma=0.001, epsilon=0.05)
        self.state = 0
        self.counter = 0
        self.done = False
        self.y_list = np.zeros(201)
        # We have 2 actions, corresponding to "turn on", "turn off"
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(201)
        self.action_list = []
        self.reward_list=[]
        self.reward_counter=0
        self.reward = 0
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: 0,
            1: 1
        }
        beta = 1
        g1 = 0.1
        g2 = 0.1
        kr = 0.2
        alpha = 1.8
        k = 0.05
        k4 = 0.1
        D = 0.32
        D1 = 0.0003
        h = 0.03
        n_cells = 100
        number = 601

        t = np.linspace(0, 60, number)
        z0 = [0 for i in range(n_cells * 5)]
        for j in range(n_cells):
            z0[j] = 1
        a = np.empty(n_cells * number)
        m1 = np.empty(n_cells * number)
        m2 = np.empty(n_cells * number)
        c = np.empty(n_cells * number)
        n = np.empty(n_cells * number)
        for j in range(n_cells):
            a[j] = z0[j]
            m1[j] = z0[j + n_cells]
            m2[j] = z0[j + 2 * n_cells]
            c[j] = z0[j + 3 * n_cells]
            n[j] = z0[j + 4 * n_cells]

        for i in range(1, number):
            tspan = [t[i - 1], t[i]]
            # import pdb; pdb.set_trace()
            act_on = 0
            power = 5
            n_cells = 100
            z = odeint(WoundEnv.model_act, z0, tspan,
                       args=(beta, g1, g2, kr, alpha, k, k4, D, D1, h, act_on, power, n_cells))
            for j in range(n_cells):
                a[i * n_cells + j] = z[1][j]
                m1[i * n_cells + j] = z[1][j + n_cells]
                m2[i * n_cells + j] = z[1][j + 2 * n_cells]
                c[i * n_cells + j] = z[1][j + 3 * n_cells]
                n[i * n_cells + j] = z[1][j + 4 * n_cells]
            z0 = z[1]
        import pandas as pd

        import numpy as np

        from sklearn import metrics
        from sklearn.svm import SVR

        y = t.reshape(601, 1)
        r = 60  # radius
        X = np.zeros((number, 5))
        for i in range(number):
            X[i, :] = [a[i * n_cells + r], m1[i * n_cells + r], m2[i * n_cells + r], c[i * n_cells + r],
                       n[i * n_cells + r]]

        self.svr_rbf.fit(X, y)
        print(1)
    def step(self, action):
        # self.state = random.randint(0, 20)
        # self.reward = random.randint(-1, 1)
        # self.done = [False, True][random.randint(0, 1)]
        # return int(self.state*10), self.reward, self.done, {}
        global np
        '''if self.done:
            print("Wound Closed")
            return self.state, self.reward, self.done, {}'''
        self.action_list.append(action)
        beta = 1
        g1 = 0.1
        g2 = 0.1
        kr = 0.2
        alpha = 1.8
        k = 0.05
        k4 = 0.1
        D = 0.32
        D1 = 0.0003
        h = 0.03
        n_cells = 100
        number = 2  # since we are solving for just two time point

        t = np.round(np.linspace(0, 60, 601), 1)
        if self.counter == 0:
            z0 = [0 for i in range(n_cells * 5)]
            for j in range(n_cells):
                z0[j] = 1
        else:
            z0 = self.z0

        self.a_con = np.empty(n_cells * number)
        self.m1_con = np.empty(n_cells * number)
        self.m2_con = np.empty(n_cells * number)
        self.c_con = np.empty(n_cells * number)
        self.n_con = np.empty(n_cells * number)
        for j in range(n_cells):
            self.a_con[j] = z0[j]
            self.m1_con[j] = z0[j + n_cells]
            self.m2_con[j] = z0[j + 2 * n_cells]
            self.c_con[j] = z0[j + 3 * n_cells]
            self.n_con[j] = z0[j + 4 * n_cells]

        # print(np.where(t == self.state)[0])
        i = int(self.counter * 10)

        timing_step = i

        tspan = [t[i], t[i + 1]]

        act_on = action
        # act_on=0
        power = 5
        n_cells = 100
        z = odeint(WoundEnv.model_act, z0, tspan,
                   args=(beta, g1, g2, kr, alpha, k, k4, D, D1, h, act_on, power, n_cells))
        for j in range(n_cells):
            self.a_con[1 * n_cells + j] = z[1][j]
            self.m1_con[1 * n_cells + j] = z[1][j + n_cells]
            self.m2_con[1 * n_cells + j] = z[1][j + 2 * n_cells]
            self.c_con[1 * n_cells + j] = z[1][j + 3 * n_cells]
            self.n_con[1 * n_cells + j] = z[1][j + 4 * n_cells]
        z0 = z[1]
        self.z0 = z[1]
        # y = t.reshape(601, 1)
        r = 60  # radius
        X = np.zeros((number, 5))
        for i in range(number):
            X[i, :] = [self.a_con[i * n_cells + r], self.m1_con[i * n_cells + r], self.m2_con[i * n_cells + r],
                       self.c_con[i * n_cells + r],
                       self.n_con[i * n_cells + r]]

        y_act_con = self.svr_rbf.predict(X[1, :].reshape(1, 5))[0]
        # print('y_act', y_act_con)
        next_state = y_act_con
        # print('state', next_state)
        self.state = next_state
        self.counter += 0.1
        # self.observation_space=int(self.state * 10)

        # if next_state < self.counter:
        #     reward = -1
        # else:
        #     reward = (next_state - self.counter)
        if next_state >= 20:
            self.done = True
            self.reward += (next_state - self.counter) * 400
            import csv
            with open('Reward_saver.csv', mode='w') as csv_file:
                fieldnames = ['episode', 'reward']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                writer.writerow({'episode': self.reward_counter, 'reward': self.reward})

            with open('Action_saver.csv', mode='w') as csv_file:
                fieldnames = ['counter', 'action']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writeheader()
                k = 0
                for i in range(len(self.action_list)):
                    l = k + i * 0.1
                    writer.writerow({'counter': l, 'action': self.action_list[i]})
            self.reward_counter += 1
        if action > 0:
            self.reward += -0.1
        # print('reward: ', reward)
        # print('state_actuator', self.state,'state_without',self.counter)
        return int(self.state * 10), self.reward, self.done, {'state_actuator': self.state, 'state_without': self.counter,
                                                         'action': action}

    def reset(self):

        self.reward=0
        self.action_list = []
        self.reward_list = []
        self.state = 0

        self.done = False
        self.counter = 0
        self.y_list = np.zeros(201)
        # We have 2 actions, corresponding to "turn on", "turn off"
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(201)
        return self.state

    def render(self, mode='human'):
        pass

        # plt.plot(self.counter, self.state)
        # plt.hold(True)


if __name__ == "__main__":
    env = WoundEnv()

    env.step(0)
