#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Giorgio Angelotti
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import gym
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import math
from gym import logger
from scipy import stats
import pandas as pd
import os

wd = os.getcwd()
tfk = tf.keras
tfkl = tfk.layers


class StochasticCartPole(gym.Wrapper):
    def __init__(self, env, std=1.0):
        self.std = std
        super().__init__(env)
        self.env = env

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = np.random.normal(loc=self.force_mag, scale=self.std) if action == 1 else np.random.normal(loc=-self.force_mag, scale=self.std)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * theta_dot ** 2 * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}


def collect(STEPS, mode):
    if mode == 'det':
        env = gym.make("CartPole-v1")
    elif mode == 'stoch':
        detenv = gym.make("CartPole-v1")
        env = StochasticCartPole(detenv, std=2.0)
    else:
        raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")

    X = np.zeros((STEPS, 4))
    Y = np.zeros((STEPS, 4))
    A = np.zeros((STEPS, 1))
    R = np.zeros((STEPS, 1))

    m = 0
    for k in range(STEPS):
        done = False
        X[k + m] = env.reset()
        while not done and k + m < STEPS:
            a = env.action_space.sample()
            A[k + m] = a
            Y[k + m], R[k + m], done, _ = env.step(a)
            if not done and k + m + 1 < STEPS:
                X[k + m + 1] = Y[k + m]
                m += 1

        if k + m + 1 == STEPS:
            break
    return X, A, R, Y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', type=str, action='store', dest='e',
                        help='Select the environment type between "det" and "stoch"')

    parser.add_argument('-b', type=int, action='store', dest='N',
                        help='Number of batches')

    parser.add_argument('-s', type=int, action='store', dest='s',
                        help='Number of batch sizes')

    parser.add_argument('-t', type=int, action='store', dest='t',
                        help='Number of steps')

    parser.add_argument('-k', type=str, action='store', dest='k',
                        help='Symmetry')

    params = parser.parse_args()
    delta = np.zeros((params.s, params.N))
    pv = np.zeros((params.s, params.N))

    for m in range(params.s):
        for k in range(params.N):
            X, A, R, Y = collect((m+1)*params.t, params.e)
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            A = A.astype(np.float32)

            # Preprocesing
            state_scaler = MaxAbsScaler()
            action_scaler = MinMaxScaler(feature_range=(-1, 1))

            state_scaler.fit(np.vstack((X, Y)))

            scale_factor = 1. #put it to 1 and not 1.5

            X = state_scaler.transform(X)*scale_factor
            Y = state_scaler.transform(Y)*scale_factor
            A = action_scaler.fit_transform(A)*scale_factor

            ### Evaluation Dataset (10 **6 is a big number of steps)
            evX, evA, evR, evY = collect(10 ** 5, params.e)

            evX = np.array(evX).astype(np.float32)
            evY = np.array(evY).astype(np.float32)
            evA = np.array(evA).astype(np.float32)

            evX = state_scaler.transform(evX) * scale_factor
            evY = state_scaler.transform(evY) * scale_factor
            evA = action_scaler.transform(evA) * scale_factor

            evQa = np.hstack((evX, evA))

            Qa = np.hstack((X, A))
            Q = np.hstack((Qa, Y))

            ### Dynamical model
            dynamics = tfk.Sequential([
                tfkl.InputLayer(input_shape=5),
                tfkl.Dense(128, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(128, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(128, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(4, use_bias=True, kernel_initializer='he_uniform', activation='tanh')
            ])

            es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

            dynamics.compile(optimizer='adam', loss='mse')
            dynamics.fit(Qa, Y, batch_size=64, epochs=5000, shuffle=True, validation_split=0.1, callbacks=[es],
                         verbose=0)

            if params.k == 'SAR':
                Xsym = -np.copy(X)
                Asym = -np.copy(A)
                Ysym = -np.copy(Y)

            elif params.k == 'ISR':
                Xsym = -np.copy(X)
                Asym = np.copy(A)
                Ysym = np.copy(Y)

            elif params.k == 'AI':
                Xsym = np.copy(X)
                Asym = -np.copy(A)
                Ysym = np.copy(Y)

            elif params.k == 'SFI':
                Xsym = np.copy(X)
                Xsym[:, 0] *= -1
                Asym = np.copy(A)
                Ysym = np.copy(Y)

            elif params.k == 'TI':
                Xsym = np.copy(X)
                Xsym[:, 0] += 0.3 #because not scaling
                Asym = np.copy(A)
                Ysym = np.copy(Y)
                Ysym[:, 0] += 0.3 #because not scaling

            else:
                raise ValueError("Select a symmetry between SAR, ISR, AI, SFI and TI")

            XAsym = np.hstack((Xsym, Asym))

            # augmenting the data set
            QaSym = np.vstack((Qa, XAsym))
            Ysym = np.vstack((Y, Ysym))

            # Dynamical model with symmetry
            dynamich = tfk.Sequential([
                tfkl.InputLayer(input_shape=5),
                tfkl.Dense(128, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(128, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(128, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(4, use_bias=True, kernel_initializer='he_uniform', activation='tanh')
            ])

            es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

            dynamich.compile(optimizer='adam', loss='mse')
            dynamich.fit(QaSym, Ysym, batch_size=64, epochs=5000, shuffle=True, validation_split=0.1, callbacks=[es],
                         verbose=0)


            # Evaluation
            dists = dynamics.predict(evQa, verbose=0)
            disth = dynamich.predict(evQa, verbose=0)
            dists = np.sum((evY - dists)**2., axis=1).reshape(-1)
            disth = np.sum((evY - disth)** 2., axis=1).reshape(-1)
            stat, pval = stats.wilcoxon(dists, disth, correction=True, alternative='less')
            delta[m, k] = stat
            pv[m, k] = pval
            print(delta[m, k], pval)

    df1 = pd.DataFrame(delta)
    df2 = pd.DataFrame(pv)
    df1.to_csv(wd+'/continuous/cartpole/results/augment_stat_'+params.e+'_'+params.k+'_'+str(params.t)+'_'+str(params.N)+'_'+str(params.s)+'.csv')
    df2.to_csv(
        wd+'/continuous/cartpole/results/augment_pval_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(params.s) + '.csv')
    print(np.median(delta, axis=1), np.median(pv, axis=1))