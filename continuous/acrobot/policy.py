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

from augment import collect
import numpy as np
import gym
from gym.utils import seeding
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from math import cos, acos, asin, sin
import os

wd = os.getcwd()
tfk = tf.keras
tfkl = tfk.layers


class ModelBasedAcrobot(gym.Wrapper):
    def __init__(self, env, model, state_scaler, action_scaler):
        super().__init__(env)
        self.env = env
        self.model = model
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,)).astype(
            np.float32
        )
        self.state = self._get_ob()
        return self.state

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.state = self.state.reshape((1, -1))
        action = np.array([[action]])

        self.state = self.state_scaler.transform(self.state)
        action = self.action_scaler.transform(action)

        model_input = np.hstack((self.state, action))

        self.state = self.model.predict(model_input)
        self.state = self.state_scaler.inverse_transform(self.state)

        self.state = self.state.reshape(-1)
        terminal = self._terminal()

        reward = -1.0 if not terminal else 0.0

        return self.state, reward, terminal, {}

    def _get_ob(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        )

    def _terminal(self):
        s = self.state
        return bool(-s[0] - s[0]*s[2] + s[1]*s[3] > 1.0)


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
    perf = np.zeros((params.s, params.N))
    perfstd = np.zeros((params.s, params.N))

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

            if params.e == 'det':
                cp = gym.make("Acrobot-v1")
            elif params.e == 'stoch':
                cp = gym.make("Acrobot-v1")
                cp.torque_noise_max = 0.5
            else:
                raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")

            BaseModel = ModelBasedAcrobot(cp, dynamics, state_scaler, action_scaler)
            #BaseModel = Monitor(BaseModel)
            env = DummyVecEnv([lambda: BaseModel])
            env = VecMonitor(env)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            # Instantiate the agent
            model = PPO('MlpPolicy', env, verbose=0, device='cpu')
            # Train the agent
            model.learn(total_timesteps=int(5e4))
            # Save the agent
            model.save(wd+'/continuous/acrobot/results/'+params.e+'ppo_base'+params.k)

            env2 = DummyVecEnv([lambda: cp])
            env2 = VecMonitor(env2)
            env2 = VecNormalize(env2, norm_obs=True, norm_reward=True)
            del model
            model = PPO.load(wd+'/continuous/acrobot/results/'+params.e+'ppo_base'+params.k, env2, verbose=False, device='cpu')
            env2.norm_reward = False
            rwdbt, stdbt = evaluate_policy(model, env2, n_eval_episodes=100, deterministic=True)
            #print('Base True', rwdbt, stdbt)
            del model, BaseModel, env, env2

            SymModel = ModelBasedAcrobot(cp, dynamich, state_scaler, action_scaler)
            #SymModel = Monitor(SymModel)
            env = DummyVecEnv([lambda: SymModel])
            env = VecMonitor(env)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            # Instantiate the agent
            model = PPO('MlpPolicy', env, verbose=0, device='cpu')
            # Train the agent
            model.learn(total_timesteps=int(5e4))
            # Save the agent
            model.save(wd+'/continuous/acrobot/results/'+params.e+'ppo_sym'+params.k)
            #rwds, stds = evaluate_policy(model, env, n_eval_episodes=30, deterministic=True)


            env2 = DummyVecEnv([lambda: cp])
            env2 = VecMonitor(env2)
            env2 = VecNormalize(env2, norm_obs=True, norm_reward=True)
            del model
            model = PPO.load(wd+'/continuous/acrobot/results/'+params.e+'ppo_sym'+params.k, env2, verbose=False, device='cpu')
            env2.norm_reward = False
            rwdst, stdst = evaluate_policy(model, env2, n_eval_episodes=100, deterministic=True)

            # perf[m, k, 0] = rwdt
            perf[m, k] = rwdbt - rwdst
            # perfstd[m, k, 0] = stdt
            # perfstd[m, k, 0] = stdbt
            # perfstd[m, k, 1] = stdst
            df1 = pd.DataFrame(perf)

            df1.to_csv(
                wd+'/continuous/acrobot/results/polmean_' + params.e + '_' + params.k + '_' + str(
                    params.t) + '_' + str(params.N) + '_' + str(
                    params.s) + '.csv')
            del df1