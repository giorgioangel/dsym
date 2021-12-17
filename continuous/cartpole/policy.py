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
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from gym import logger
from check import StochasticCartPole
import os

wd = os.getcwd()
tfk = tf.keras
tfkl = tfk.layers


class ModelBasedCartPole(gym.Wrapper):
    def __init__(self, env, model, state_scaler, action_scaler):
        super().__init__(env)
        self.env = env
        self.model = model
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

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
        done = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
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

        return self.state, reward, done, {}


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
    #perfstd = np.zeros((params.s, params.N, 2))

    for m in range(params.s):
        for k in range(params.N):
            print(m, k)
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

            es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)

            dynamics.compile(optimizer='adam', loss='mse')
            dynamics.fit(Qa, Y, batch_size=128, epochs=10000, shuffle=True, validation_split=0.1, callbacks=[es],
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

            es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)

            dynamich.compile(optimizer='adam', loss='mse')
            dynamich.fit(QaSym, Ysym, batch_size=128, epochs=10000, shuffle=True, validation_split=0.2, callbacks=[es],
                         verbose=0)

            if params.e == 'det':
                cp = gym.make("CartPole-v1")
            elif params.e == 'stoch':
                detenv = gym.make("CartPole-v1")
                cp = StochasticCartPole(detenv, std=2.0)
            else:
                raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")
            # Automatically normalize the input features and reward
            '''
            env = DummyVecEnv([lambda: cp])
            env = VecMonitor(env)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            # Instantiate the agent
            model = PPO('MlpPolicy', env, verbose=0, device='cpu')
            # Train the agent
            model.learn(total_timesteps=int(1e5))
            #env.training = False
            env.norm_reward = False
            rwdt, stdt = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
            del model, env
            '''
            BaseModel = ModelBasedCartPole(cp, dynamics, state_scaler, action_scaler)
            #BaseModel = Monitor(BaseModel)
            env = DummyVecEnv([lambda: BaseModel])
            env = VecMonitor(env)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            # Instantiate the agent
            model = PPO('MlpPolicy', env, verbose=0, device='cpu')
            # Train the agent
            model.learn(total_timesteps=int(5e4))
            # Save the agent
            model.save(wd+'/continuous/cartpole/results/'+params.e+'ppo_base'+params.k)

            env2 = DummyVecEnv([lambda: cp])
            env2 = VecMonitor(env2)
            env2 = VecNormalize(env2, norm_obs=True, norm_reward=True)
            del model
            model = PPO.load(wd+'/continuous/cartpole/results/'+params.e+'ppo_base'+params.k, env2, verbose=False, device='cpu')
            env2.norm_reward = False
            rwdbt, stdbt = evaluate_policy(model, env2, n_eval_episodes=100, deterministic=True)
            #print('Base True', rwdbt, stdbt)
            del model, BaseModel, env, env2

            SymModel = ModelBasedCartPole(cp, dynamich, state_scaler, action_scaler)
            #SymModel = Monitor(SymModel)
            env = DummyVecEnv([lambda: SymModel])
            env = VecMonitor(env)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            # Instantiate the agent
            model = PPO('MlpPolicy', env, verbose=0, device='cpu')
            # Train the agent
            model.learn(total_timesteps=int(5e4))
            # Save the agent
            model.save(wd+'/continuous/cartpole/results/'+params.e+'ppo_sym'+params.k)
            #rwds, stds = evaluate_policy(model, env, n_eval_episodes=30, deterministic=True)


            env2 = DummyVecEnv([lambda: cp])
            env2 = VecMonitor(env2)
            env2 = VecNormalize(env2, norm_obs=True, norm_reward=True)
            del model
            model = PPO.load(wd+'/continuous/cartpole/results/'+params.e+'ppo_sym'+params.k, env2, verbose=False, device='cpu')
            env2.norm_reward = False
            rwdst, stdst = evaluate_policy(model, env2, n_eval_episodes=100, deterministic=True)

            #perf[m, k, 0] = rwdt
            perf[m, k] = rwdbt-rwdst
            #perfstd[m, k, 0] = stdt
            #perfstd[m, k, 0] = stdbt
            #perfstd[m, k, 1] = stdst
            df1 = pd.DataFrame(perf)
            df1.to_csv(
                wd + '/continuous/cartpole/results/polmean_' + params.e + '_' + params.k + '_' + str(
                    params.t) + '_' + str(params.N) + '_' + str(
                    params.s) + '.csv')
            del df1