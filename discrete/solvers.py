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
import numba as nb


# FUNCTION THAT SOLVES AN UPPER TRIANGULAR LINEAR PROBLEM A x = b
# Input: matrix A, known vector b
# Output: Solution x
@nb.njit((nb.float64[:, :], nb.float64[:]), fastmath=True, nogil=True, cache=True)
def solve_triangular(matrix, vector):
    output = np.zeros_like(vector)
    output[-1] = vector[-1]/matrix[-1, -1]
    for i in range(vector.shape[0]-1):
        index = vector.shape[0]-2-i
        output[index] = (vector[index] - np.dot(matrix[index, index+1:], output[index+1:]))/matrix[index, index]
    return output


# FUNCTION THAT COMPUTES THE MARKOV CHAIN TRANSITION FUNCTION GIVEN A POLICY AND THE EXPECTED REWARD FROM THE
# IMMEDIATE REWARD.
# Input: MDP transition matrix, Immediate Reward matrix, Policy
# Output: MC transition matrix, Expected Reward matrix
@nb.njit((nb.float64[:, :, :], nb.float64[:, :, :], nb.int32[:]), fastmath=True, nogil=True, cache=True)
def pol_model(transition, reward, policy):
    rew = np.zeros(reward.shape[1], dtype=np.float64)
    trans = np.eye(reward.shape[1], dtype=np.float64)
    for s in range(reward.shape[1]):
        index = policy[s]
        trans[s] = transition[index, s]
        rew[s] = np.dot(trans[s], reward[index, s])
    return trans, rew


# FUNCTION THAT COMPUTES THE MARKOV CHAIN TRANSITION FUNCTION GIVEN A STOCHASTIC POLICY AND THE EXPECTED REWARD FROM THE
# IMMEDIATE REWARD.
# Input: MDP transition matrix, Immediate Reward matrix, Policy
# Output: MC transition matrix, Expected Reward matrix
def pol_model_s(transition, reward, policy):
    rew = np.zeros(reward.shape[1], dtype=np.float64)
    trans = np.eye(reward.shape[1], dtype=np.float64)
    for s in range(reward.shape[1]):
        for a in range(transition.shape[0]):
            for sn in range(reward.shape[1]):
                trans[s, sn] += policy[s, a]*transition[a, s, sn]
                rew[s] += policy[s, a]*transition[a, s, sn]*reward[a,s,sn]
    for s in range(reward.shape[1]):
        trans[s] /= np.sum(trans[s])
    return trans, rew


# FUNCTION THAT PERFORMS POLICY EVALUATION AND THEN OUTPUTS THE PERFORMANCE OF THE POLICY:
# THE DOT PRODUCT BETWEEN THE VALUE FUNCTION AND THE INITIAL STATE DISTRIBUTION
# Input: MDP transition matrix, Immediate Reward matrix, Policy, Gamma, Initial State distribution
# Output: Performance of the policy
@nb.njit((nb.float64[:, :, :], nb.float64[:, :, :], nb.int32[:], nb.float64, nb.float64[:]), fastmath=True, nogil=True,
         cache=True)
def policy_eval_i(t, r, policy, gamma, init):
    tre, re = pol_model(t, r, policy)
    z = np.eye(tre.shape[0], dtype=np.float64) - gamma * tre
    q, rr = np.linalg.qr(z)
    y = np.dot(q.T, re)
    v = solve_triangular(rr, y)
    #z = np.linalg.inv(z)
    #v = np.dot(z, re)
    out = np.dot(init, v)
    return out


# FUNCTION THAT PERFORMS POLICY EVALUATION FOR A STOCH. POLICY AND THEN OUTPUTS THE PERFORMANCE OF THE POLICY:
# THE DOT PRODUCT BETWEEN THE VALUE FUNCTION AND THE INITIAL STATE DISTRIBUTION
# Input: MDP transition matrix, Immediate Reward matrix, Policy, Gamma, Initial State distribution
# Output: Performance of the policy
def policy_eval_i_s(t, r, policy, gamma, init):
    tre, re = pol_model_s(t, r, policy)
    z = np.eye(tre.shape[0], dtype=np.float64) - gamma * tre
    q, rr = np.linalg.qr(z)
    y = np.dot(q.T, re)
    v = solve_triangular(rr, y)
    #z = np.linalg.inv(z)
    #v = np.dot(z, re)
    out = np.dot(init, v)
    return out


# FUNCTION THAT PERFORMS POLICY EVALUATION AND THEN OUTPUTS THE VALUE FUNCTION
# Input: MDP transition matrix, Immediate Reward matrix, Policy, Gamma
# Output: Value function
@nb.njit(fastmath=True, nogil=True, cache=True)
def policy_eval(t, r, gamma):
    z = np.eye(t.shape[0], dtype=np.float64) - gamma * t
    q, rr = np.linalg.qr(z)
    y = np.dot(q.T, r)
    v = solve_triangular(rr, y)
    #z = np.linalg.inv(z)
    #v = np.dot(z, r)
    return v


# FUNCTION THAT PERFORMS POLICY ITERATION AND THEN OUTPUTS THE OPTIMAL POLICY
# Input: MDP transition matrix, Immediate Reward matrix, Gamma, Starting Policy
# Output: Optimal Policy
@nb.njit((nb.float64[:, :, :], nb.float64[:, :, :], nb.float64, nb.int32[:]), fastmath=True, nogil=True, cache=True)
def policy_iteration(t, r, gamma, pol):
    policy = np.copy(pol)
    ptest = np.copy(policy)
    reward = np.zeros((t.shape[0], t.shape[1]), dtype=np.float64)
    for s in range(t.shape[1]):
        for a in range(t.shape[0]):
            reward[a, s] = np.dot(r[a, s], t[a, s])
    vtest = np.zeros(t.shape[1], dtype=np.float64)
    tra, rew = pol_model(t, r, policy)
    v = policy_eval(tra, rew, gamma)
    vr = np.zeros_like(reward)
    v2 = np.ones_like(v)
    changed = True
    while changed is True:
        changed = False
        for a in range(t.shape[0]):
            for s in range(t.shape[1]):
                vr[a, s] = np.dot(v, t[a, s])
        for s in range(t.shape[1]):
            ptest[s] = np.argmax(reward[:, s] + gamma*vr[:, s])
            if ptest[s] != policy[s]:
                changed = True
                tra, rew = pol_model(t, r, ptest)
                v2 = policy_eval(tra, rew, gamma)
                policy[s] = ptest[s]
        if np.max(np.abs(v2-v)) < 0.001:
            v = np.copy(v2)
            changed = False
        else:
            v = np.copy(v2)
    return policy


# FUNCTION THAT PERFORMS POLICY EVALUATION AND THEN OUTPUTS THE PERFORMANCE OF THE POLICY:
# THE DOT PRODUCT BETWEEN THE VALUE FUNCTION AND THE INITIAL STATE DISTRIBUTION
# Input: MDP transition matrix, Immediate Reward matrix, Policy, Gamma, Initial State distribution
# Output: Performance of the policy
@nb.njit((nb.float64[:, :, :], nb.float64[:, :, :], nb.int32[:], nb.float64, nb.float64[:]), fastmath=True, nogil=True,
         cache=True)
def iterative_policy_eval_i(t, r, policy, gamma, init):
    threshold = 0.01
    out = np.ones(t.shape[1], dtype=nb.float64)
    tre, re = pol_model(t, r, policy)
    delta = np.inf
    counter = 0
    while delta >= threshold and counter < 1e7:
        delta = 0
        for s in range(tre.shape[1]):
            counter += 1
            v_temp = out[s]
            out[s] = re[s] + gamma*np.dot(tre[s], out)
            delta = max(delta, abs(v_temp-out[s]))
    out = np.dot(init, out)
    return out


@nb.njit((nb.float64[:, :], nb.float64[:], nb.float64[:], nb.float64, nb.float64), fastmath=True, nogil=True,
         cache=True)
def iterative_policy_eval(t, r, v, gamma, threshold):
    out = np.copy(v)
    delta = np.inf
    counter = 0
    while delta >= threshold and counter < 1e7:
        delta = 0
        for s in range(t.shape[1]):
            counter += 1
            v_temp = out[s]
            out[s] = r[s] + gamma*np.dot(t[s], out)
            delta = max(delta, abs(v_temp-out[s]))
    return out


# FUNCTION THAT PERFORMS POLICY ITERATION AND THEN OUTPUTS THE OPTIMAL POLICY
# Input: MDP transition matrix, Immediate Reward matrix, Gamma, Starting Policy
# Output: Optimal Policy
@nb.njit((nb.float64[:, :, :], nb.float64[:, :, :], nb.float64, nb.int32[:]), fastmath=True, nogil=True, cache=True)
def iterative_policy_iteration(t, r, gamma, pol):
    policy = np.copy(pol)
    ptest = np.copy(policy)
    vtest = np.ones(t.shape[1], dtype=np.float64)
    tra, rew = pol_model(t, r, policy)
    v = iterative_policy_eval(tra, rew, vtest, gamma, 0.01)
    changed = True
    antiloop = 0
    while changed is True and antiloop < 10:
        changed = False
        for s in range(t.shape[1]):
            for a in range(t.shape[0]):
                if policy[s] != a:
                    ptest[s] = a
                    tra, rew = pol_model(t, r, ptest)
                    vtest = iterative_policy_eval(tra, rew, vtest, gamma, 0.01)
                    if np.all(vtest >= v) is True:
                        if np.all(vtest == v) is False:
                            v = np.copy(vtest)
                            policy = np.copy(ptest)
                            changed = True
                            antiloop = 0
                        else:
                            policy = np.copy(ptest)
                            changed = True
                            antiloop += 1
                    else:
                        ptest = np.copy(policy)
    return policy
