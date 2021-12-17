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
import multiprocessing as mp
from grids import Grid, StochasticGrid
from check import collect, learn_transition
from solvers import policy_iteration, policy_eval_i
from scipy import stats
import pandas as pd
import os

wd = os.getcwd()


def compare_perf(stoch_type, size, symmetry, b, min_batch, T, Rew):
    if stoch_type == 'det':
        X, A, R, Y = collect(min_batch * (b + 1), size, Grid)
    elif stoch_type == 'stoch':
        X, A, R, Y = collect(min_batch * (b + 1), size, StochasticGrid)
    else:
        raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")

    Ts, N = learn_transition(size, X, Y, A)
    Ts = Ts.astype(np.float64)
    gamma = 0.9
    policy = np.random.randint(0, 4, size=size ** 2, dtype=np.int32)
    init = 1. / (size ** 2 - 1) * np.ones(size ** 2, dtype=np.int32)
    init[-1] = 0

    Asym = np.copy(A)

    if symmetry == 'TRSAI':
        Xsym = np.copy(Y)
        Ysym = np.copy(X)
        for i in range(A.shape[0]):
            if A[i] == 0:
                Asym[i] = 1
            elif A[i] == 1:
                Asym[i] = 0
            elif A[i] == 2:
                Asym[i] = 3
            else:
                Asym[i] = 2

    elif symmetry == 'SDAI':
        Xsym = np.copy(X)
        Ysym = np.copy(Y)
        for i in range(A.shape[0]):
            if A[i] == 0:
                Asym[i] = 1
            elif A[i] == 1:
                Asym[i] = 0
            elif A[i] == 2:
                Asym[i] = 3
            else:
                Asym[i] = 2

    elif symmetry == 'ODAI' or symmetry == 'ODWA':
        Xsym = np.copy(X)
        Ysym = np.copy(Y)
        for i in range(A.shape[0]):
            if A[i] == 0:
                Asym[i] = 1
            elif A[i] == 1:
                Asym[i] = 0
            elif A[i] == 2:
                Asym[i] = 3
            else:
                Asym[i] = 2
        for i in range(A.shape[0]):
            if Asym[i] == 0:
                temp = np.copy(Xsym[i] + size)
                if temp // size < size:
                    Ysym[i] = temp
                else:
                    Ysym[i] = Xsym[i] % size

            elif Asym[i] == 1:
                temp = np.copy(Xsym[i] - size)
                if temp // size >= 0:
                    Ysym[i] = temp
                else:
                    Ysym[i] = Xsym[i] + size * (size - 1)

            elif Asym[i] == 2:
                temp = np.copy(Xsym[i] - 1)
                mod = Xsym[i] // size
                if temp // size == mod:
                    Ysym[i] = temp
                else:
                    Ysym[i] = (size - 1) + Xsym[i]

            elif Asym[i] == 3:
                temp = np.copy(Xsym[i] + 1)
                mod = Xsym[i] // size
                if temp // size == mod:
                    Ysym[i] = temp
                else:
                    Ysym[i] = mod * size

        if symmetry == 'ODWA':
            for i in range(A.shape[0]):
                if A[i] == 0:
                    Asym[i] = 3
                elif A[i] == 1:
                    Asym[i] = 2
                elif A[i] == 2:
                    Asym[i] = 1
                else:
                    Asym[i] = 0

    elif symmetry == 'TI':
        Xsym = np.copy(Y)
        Ysym = np.copy(Y)

        for i in range(A.shape[0]):
            if Asym[i] == 0:
                temp = np.copy(Xsym[i] + size)
                if temp // size < size:
                    Ysym[i] = temp
                else:
                    Ysym[i] = Xsym[i] % size

            elif Asym[i] == 1:
                temp = np.copy(Xsym[i] - size)
                if temp // size >= 0:
                    Ysym[i] = temp
                else:
                    Ysym[i] = Xsym[i] + size * (size - 1)

            elif Asym[i] == 2:
                temp = np.copy(Xsym[i] - 1)
                mod = Xsym[i] // size
                if temp // size == mod:
                    Ysym[i] = temp
                else:
                    Ysym[i] = (size - 1) + Xsym[i]

            elif Asym[i] == 3:
                temp = np.copy(Xsym[i] + 1)
                mod = Xsym[i] // size
                if temp // size == mod:
                    Ysym[i] = temp
                else:
                    Ysym[i] = mod * size

    elif symmetry == 'TIOD':
        Xsym = np.copy(Y)
        Ysym = np.copy(X)

    else:
        raise ValueError("Select a symmetry between TRSAI, SDAI, ODAI, ODWA, TI and TIOD")

    X = np.vstack((X, Xsym))
    A = np.vstack((A, Asym))
    Y = np.vstack((Y, Ysym))

    Th, Nh = learn_transition(size, X, Y, A)
    Th = Th.astype(np.float64)

    optimal_policy = policy_iteration(T, Rew, gamma, policy)
    #perf = policy_eval_i(T, Rew, optimal_policy, gamma, init)
    optimal_policys = policy_iteration(Ts, Rew, gamma, optimal_policy)
    #perfs = policy_eval_i(Ts, Rew, optimal_policys, gamma, init)
    perfstrue = policy_eval_i(T, Rew, optimal_policys, gamma, init)
    optimal_policyh = policy_iteration(Th, Rew, gamma, optimal_policy)
    #perfh = policy_eval_i(Th, Rew, optimal_policyh, gamma, init)
    perfhtrue = policy_eval_i(T, Rew, optimal_policyh, gamma, init)
    return perfstrue, perfhtrue

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', type=str, action='store', dest='e',
                        help='Select the environment type between "det" and "stoch"')

    parser.add_argument('-l', type=int, action='store', dest='size',
                        help='Grid side size')

    parser.add_argument('-b', type=int, action='store', dest='N',
                        help='Number of batches')

    parser.add_argument('-s', type=int, action='store', dest='s',
                        help='Number of batch sizes')

    parser.add_argument('-t', type=int, action='store', dest='t',
                        help='Min number of steps per batch')

    parser.add_argument('-k', type=str, action='store', dest='k',
                        help='Symmetry')

    params = parser.parse_args()

    T = np.zeros((4, params.size ** 2, params.size ** 2), dtype=np.float32)
    Rew = -1 * np.ones((4, params.size ** 2, params.size ** 2), dtype=np.float64)
    Rew[:, :, params.size ** 2 - 1] = 1
    if params.e == 'stoch':
        testenv = Grid(params.size)
        ## creating true matrix T
        for state in range(params.size ** 2):
            testenv.state = state
            testenv.done = False
            for action in range(4):
                if action == 0:
                    for movement in range(4):
                        testenv.state = state
                        testenv.done = False
                        if movement == 0:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.6
                        elif movement == 1:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.2
                        else:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.1
                elif action == 1:
                    for movement in range(4):
                        testenv.state = state
                        testenv.done = False
                        if movement == 1:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.6
                        elif movement == 0:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.2
                        else:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.1
                elif action == 2:
                    for movement in range(4):
                        testenv.state = state
                        testenv.done = False
                        if movement == 2:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.6
                        elif movement == 3:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.2
                        else:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.1
                elif action == 3:
                    for movement in range(4):
                        testenv.state = state
                        testenv.done = False
                        if movement == 3:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.6
                        elif movement == 2:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.2
                        else:
                            testenv.step(movement)
                            next_state = testenv.state
                            testenv.done = False
                            T[action, state, next_state] = 0.1
                else:
                    pass
            # if np.sum(T[:, state, :]) != 4:
            # print(state, np.sum(T[:, state, :]))
    elif params.e == 'det':
        testenv = Grid(params.size)
        ## creating true matrix T
        for state in range(params.size ** 2):
            testenv.state = state
            testenv.done = False
            for action in range(4):
                testenv.step(action)
                next_state = testenv.state
                T[action, state, next_state] = 1
    else:
        raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")
    perf = np.zeros((params.s, params.N))
    pval = np.zeros(perf.shape[0])
    size = params.size
    min_batch = params.t
    T = T.astype(np.float64)
    inputs = [(params.e, size, params.k, b, min_batch, T, Rew) for b in range(params.s) for k in range(params.N)]
    #print(inputs)
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(compare_perf, inputs)
    pool.close()
    pool.join()
    index = 0
    for b in range(perf.shape[0]):
        for k in range(params.N):
            #p1, p2 = compare_perf(params.e, size, params.k, b, min_batch, T, Rew)
            perf[b, k] = results[index][0]-results[index][1]
            #perf[b, k] = p1-p2
            index += 1
        stat, pv = stats.wilcoxon(perf[b], correction=True)
        pval[b] = pv

    df1 = pd.DataFrame(perf)
    df2 = pd.DataFrame(pval)
    df1.to_csv(
        wd+'/discrete/results/perf_difference_' + str(params.size) + '_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    df2.to_csv(
        wd+'/discrete/results/perf_pval_' + str(params.size) + '_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    #print("Delta: mean, std, min, max, median\n")
    #print(np.mean(perf, axis=1), np.std(perf, axis=1), np.min(perf, axis=1), np.max(perf, axis=1), np.median(perf,
    #                                                                                                         axis=1))
    #print("\n")
    #print("Pvalue")
    #print(pval)
