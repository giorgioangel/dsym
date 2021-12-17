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
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import math
from gym import logger
import pandas as pd
import os

wd = os.getcwd()
tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfb = tfp.bijectors


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


def init_once(x, name):
  return tf.compat.v1.get_variable(name, initializer=x, trainable=False)


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


def make_degrees(p, hidden_dims):
    m = [tf.constant(range(1, p + 1))]
    for dim in hidden_dims:
        n_min = min(np.min(m[-1]), p - 1)
        degrees = (np.arange(dim) % max(1, p - 1) + min(1, p - 1))
        degrees = tf.constant(degrees, dtype="int32")
        m.append(degrees)
    return m


def make_masks(degrees):
    masks = [None] * len(degrees)
    for i, (ind, outd) in enumerate(zip(degrees[:-1], degrees[1:])):
        masks[i] = tf.cast(ind[:, tf.newaxis] <= outd, dtype="float32")
    masks[-1] = tf.cast(degrees[-1][:, np.newaxis] < degrees[0], dtype="float32")
    return masks


def make_constraint(mask):
    def _constraint(x):
        return mask * tf.identity(x)
    return _constraint


def make_init(mask):
    def _init(shape, dtype=None):
        return mask * tf.keras.initializers.GlorotUniform(23)(shape)
    return _init


def make_network(p, hidden_dims, params):
    masks = make_masks(make_degrees(p, hidden_dims))
    masks[-1] = tf.tile(masks[-1][..., tf.newaxis], [1, 1, params])
    masks[-1] = tf.reshape(masks[-1], [masks[-1].shape[0], p * params])

    network = tf.keras.Sequential([
        tf.keras.layers.InputLayer((p,))
    ])
    for dim, mask in zip(hidden_dims + [p * params], masks):
        layer = tf.keras.layers.Dense(
            dim,
            kernel_constraint=make_constraint(mask),
            kernel_initializer=make_init(mask),
            activation=tf.nn.leaky_relu)
        network.add(layer)
    network.add(tf.keras.layers.Reshape([p, params]))

    return network


class MAF(tfb.Bijector):
    def __init__(self, shift_and_log_scale_fn, name="maf"):
        super(MAF, self).__init__(forward_min_event_ndims=1, name=name)
        self._shift_and_log_scale_fn = shift_and_log_scale_fn

    def _shift_and_log_scale(self, y):
        params = self._shift_and_log_scale_fn(y)
        shift, log_scale = tf.unstack(params, num=2, axis=-1)
        return shift, log_scale

    def _forward(self, x):
        y = tf.zeros_like(x, dtype=tf.float32)
        for i in range(x.shape[-1]):
            shift, log_scale = self._shift_and_log_scale(y)
            y = x * tf.math.exp(log_scale) + shift
        return y

    def _inverse(self, y):
        shift, log_scale = self._shift_and_log_scale(y)
        return (y - shift) * tf.math.exp(-log_scale)

    def _inverse_log_det_jacobian(self, y):
        _, log_scale = self._shift_and_log_scale(y)
        return -tf.reduce_sum(log_scale, axis=self.forward_min_event_ndims)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', type=str, action='store', dest='e',
                        help='Select the environment type between "det" and "stoch"')

    parser.add_argument('-b', type=int, action='store', dest='N',
                        help='Number of batches')

    parser.add_argument('-s', type=int, action='store', dest='s',
                        help='Number of batch sizes')

    parser.add_argument('-q', type=float, action='store', dest='q',
                        help='Quantile')

    parser.add_argument('-t', type=int, action='store', dest='t',
                        help='Number of steps')

    parser.add_argument('-k', type=str, action='store', dest='k',
                        help='Symmetry')

    params = parser.parse_args()
    nu = np.zeros((params.s, params.N))

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

            scale_factor = 1.5
            X = state_scaler.transform(X)*scale_factor
            Y = state_scaler.transform(Y)*scale_factor
            A = action_scaler.fit_transform(A)*scale_factor

            ### Evaluation Dataset (10 **6 is a big number of steps)
            evX, evA, evR, evY = collect(10 ** 6, params.e)

            evX = np.array(evX).astype(np.float32)
            evY = np.array(evY).astype(np.float32)
            evA = np.array(evA).astype(np.float32)

            evX = state_scaler.transform(evX) * scale_factor
            evY = state_scaler.transform(evY) * scale_factor
            evA = action_scaler.transform(evA) * scale_factor

            Qa = np.hstack((X, A))
            Q = np.hstack((Qa, Y))

            ### Creating the MAF Normalizing Flow
            dim = 9
            hidden_dim = [32 for i in range(3)]
            layers = len(hidden_dim)
            bijectors = []
            for i in range(0, layers):
                made = make_network(dim, hidden_dim, 2)
                bijectors.append(MAF(made))
                bijectors.append(tfb.Permute(init_once(
                        np.random.permutation(dim).astype('int32'),
                        name='permute_%d' % i)))

            bijectors = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))

            distribution = tfd.TransformedDistribution(
                distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(dim)),
                bijector=bijectors
            )

            x_ = tfkl.Input(shape=(dim,), dtype=tf.float32)

            distribution.log_prob(x_)

            log_prob_ = distribution.log_prob(x_)
            model = tfk.Model(x_, log_prob_)

            # Compiling the model
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3, amsgrad=True), loss=lambda _, log_prob: -log_prob)

            # Training the model
            _ = model.fit(x=Q,
                          y=np.zeros((Q.shape[0], 0), dtype=np.float32),
                          batch_size= Q.shape[0],
                          epochs=3000,
                          steps_per_epoch=1,
                          verbose=0,
                          shuffle=True)

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
                Xsym[:, 0] += 0.45
                Asym = np.copy(A)
                Ysym = np.copy(Y)
                Ysym[:, 0] += 0.45

            else:
                raise ValueError("Select a symmetry between SAR, ISR, AI, SFI and TI")

            XAsym = np.hstack((Xsym, Asym))
            Qsym = np.hstack((XAsym, Ysym))

            M = distribution.log_prob(Q)
            U = distribution.log_prob(Qsym)

            mM = np.quantile(M, params.q)

            nu_sim = 0

            for i in range(len(U)):
                if U[i] > mM:
                    nu_sim += 1

            print(mM, nu_sim/len(U))
            print(m, k)
            nu[m, k] = nu_sim/len(U)

    df = pd.DataFrame(nu)
    df.to_csv(
        wd+'/continuous/cartpole/results/check_nu_' + params.e + '_' +str(params.q)+ '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    print(np.mean(nu, axis=1), np.std(nu, axis=1))