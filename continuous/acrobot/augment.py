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
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pandas as pd
import os

wd = os.getcwd()

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def collect(STEPS, mode):
    if mode == 'det':
        env = gym.make("Acrobot-v1")
    elif mode == 'stoch':
        env = gym.make("Acrobot-v1")
        env.torque_noise_max = 0.5
    else:
        raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")
    X = np.zeros((STEPS, 6))
    Y = np.zeros((STEPS, 6))
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

            ### Evaluation Dataset (10 **6 is a big number of steps)
            evX, evA, evR, evY = collect(10 ** 5, params.e)

            evX = np.array(evX).astype(np.float32)
            evY = np.array(evY).astype(np.float32)
            evA = np.array(evA).astype(np.float32)

            evX[:, -2] /= 4 * np.pi
            evX[:, -1] /= 9 * np.pi
            evY[:, -2] /= 4 * np.pi
            evY[:, -1] /= 9 * np.pi

            # Preprocessing
            action_scaler = MinMaxScaler(feature_range=(-1, 1))
            state_scaler = StandardScaler()

            scale_factor = 1. #put it to 1 and not 3

            X[:, -2] /= 4*np.pi
            X[:, -1] /= 9*np.pi
            Y[:, -2] /= 4*np.pi
            Y[:, -1] /= 9*np.pi

            state_scaler.fit(np.vstack((X, Y)))
            X = state_scaler.transform(X) * scale_factor
            Y = state_scaler.transform(Y) * scale_factor

            A = action_scaler.fit_transform(A) * scale_factor

            evX = state_scaler.transform(evX) * scale_factor
            evY = state_scaler.transform(evY) * scale_factor
            evA = action_scaler.transform(evA) * scale_factor

            evQa = np.hstack((evX, evA))

            Qa = np.hstack((X, A))
            Q = np.hstack((Qa, Y))

            # Dynamical model
            dynamics = tfk.Sequential([
                tfkl.InputLayer(input_shape=7),
                tfkl.Dense(256, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dropout(0.2),
                tfkl.Dense(256, use_bias=False, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                # tfkl.BatchNormalization(),
                tfkl.Dense(256, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(6, use_bias=True, kernel_initializer='he_uniform', activation=None)
            ])

            es = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)

            dynamics.compile(optimizer='adam', loss='log_cosh')
            dynamics.fit(Qa, Y, batch_size=64, epochs=5000, shuffle=True, validation_split=0.1, callbacks=[es],
                         verbose=0)

            # action summary: 0 LEFT, 1 NOTHING, 2 RIGHT

            if params.k == 'AAVI':
                Xsym = np.copy(X)
                Xsym[:, 1] = - Xsym[:, 1]
                Xsym[:, 3] = - Xsym[:, 3]
                Xsym[:, 4] = - Xsym[:, 4]
                Xsym[:, 5] = - Xsym[:, 5]
                Asym = -np.copy(A) # 0 LEFT -> 2 RIGHT, 1 NOTHING -> 1 NOTHING, 2 RIGHT -> 0 LEFT
                Ysym = np.copy(Y)
                Ysym[:, 1] = - Ysym[:, 1]
                Ysym[:, 3] = - Ysym[:, 3]
                Ysym[:, 4] = - Ysym[:, 4]
                Ysym[:, 5] = - Ysym[:, 5]

            elif params.k == 'CAVI':
                Xsym = np.copy(X)
                Xsym[:, 0] = - Xsym[:, 0]
                Xsym[:, 2] = - Xsym[:, 2]
                Xsym[:, 4] = - Xsym[:, 4]
                Xsym[:, 5] = - Xsym[:, 5]
                Asym = -np.copy(A) # 0 LEFT -> 2 RIGHT, 1 NOTHING -> 1 NOTHING, 2 RIGHT -> 0 LEFT
                Ysym = np.copy(Y)
                Ysym[:, 0] = - Ysym[:, 0]
                Ysym[:, 2] = - Ysym[:, 2]
                Ysym[:, 4] = - Ysym[:, 4]
                Ysym[:, 5] = - Ysym[:, 5]

            elif params.k == 'AI':
                Xsym = np.copy(X)
                Asym = -np.copy(A) # 0 LEFT -> 2 RIGHT, 1 NOTHING -> 1 NOTHING, 2 RIGHT -> 0 LEFT
                Ysym = np.copy(Y)

            elif params.k == 'SSI':
                Xsym = -np.copy(X)
                Asym = np.copy(A)
                Ysym = np.copy(Y)

            else:
                raise ValueError("Select a symmetry between AAVI, CAVI, AI and SSI")

            XAsym = np.hstack((Xsym, Asym))

            # augmenting the data set
            QaSym = np.vstack((Qa, XAsym))
            Ysym = np.vstack((Y, Ysym))

            # Dynamical model for symmetry
            dynamich = tfk.Sequential([
                tfkl.InputLayer(input_shape=7),
                tfkl.Dense(256, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dropout(0.2),
                tfkl.Dense(256, use_bias=False, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.BatchNormalization(),
                tfkl.Dense(256, use_bias=True, kernel_initializer='he_uniform', activation=None),
                tfkl.LeakyReLU(0.2),
                tfkl.Dense(6, use_bias=True, kernel_initializer='he_uniform', activation=None)
            ])

            es = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)

            dynamich.compile(optimizer='adam', loss='log_cosh')
            dynamich.fit(QaSym, Ysym, batch_size=64, epochs=5000, shuffle=True, validation_split=0.1, callbacks=[es],
                         verbose=0)
            # Evaluation
            dists = dynamics.predict(evQa, verbose=0)
            disth = dynamich.predict(evQa, verbose=0)
            dists = np.sum(np.log(np.cosh(evY - dists)), axis=1).reshape(-1)
            disth = np.sum(np.log(np.cosh(evY - disth)), axis=1).reshape(-1)
            stat, pval = stats.wilcoxon(dists, disth, correction=True, alternative='less')
            delta[m, k] = stat
            pv[m, k] = pval
            print(delta[m, k], pval)

    df1 = pd.DataFrame(delta)
    df2 = pd.DataFrame(pv)
    df1.to_csv(wd+'/continuous/acrobot/results/augment_stat_'+params.e+'_'+params.k+'_'+str(params.t)+'_'+str(params.N)+'_'+str(params.s)+'.csv')
    df2.to_csv(
        wd+'/continuous/acrobot/results/augment_pval_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(params.s) + '.csv')
    print(np.median(delta, axis=1), np.median(pv, axis=1))