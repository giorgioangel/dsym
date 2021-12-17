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
import multiprocessing as mp
from grids import StochasticGrid, Grid
from check import collect, learn_transition
from statsmodels.stats.power import GofChisquarePower
import pandas as pd
import os

wd = os.getcwd()

@nb.njit((nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :, :], nb.int32),
         nogil=True, fastmath=True, cache=True)
def distsum(A, B, N, samples):
    counter = np.zeros((A.shape[0], A.shape[1]), dtype=nb.float32)
    #for a in range(A.shape[0]):
        #for s in range(A.shape[1]):
            #if np.sum(N[a, s]) >= samples:
                #counter[a, s] = 1
    return np.sum(np.abs(A-B)), np.mean(counter)


def compute_distance(b, T, size, min_batch, symmetry, stoch_type):
    if stoch_type == 'det':
        X, A, R, Y = collect(min_batch * (b + 1), size, Grid)
    elif stoch_type == 'stoch':
        X, A, R, Y = collect(min_batch * (b + 1), size, StochasticGrid)
    else:
        raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")
    # 0 UP 1 DOWN 2 LEFT 3 RIGHT
    Ts, Ns = learn_transition(size, X, Y, A)
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
                    Ysym[i] = Xsym[i] + size*(size - 1)

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
                    Ysym[i] = Xsym[i] + size*(size - 1)

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

    chipower = GofChisquarePower()
    samples = int(chipower.solve_power(effect_size=0.4, power=0.95, alpha=0.05, n_bins=T.shape[1]))

    dists, cs = distsum(T, Ts, Nh, samples)
    disth, cn = distsum(T, Th, Nh, samples)
    dist = dists - disth

    return dist, cn


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
            #if np.sum(T[:, state, :]) != 4:
                #print(state, np.sum(T[:, state, :]))
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
    dist = np.zeros((params.s, params.N))
    count = np.zeros((params.s, params.N))
    size = params.size
    min_batch = params.t
    inputs = [(b, T, size, min_batch, params.k, params.e) for b in range(params.s) for k in range(params.N)]
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(compute_distance, inputs)
    pool.close()
    pool.join()
    index = 0
    for b in range(dist.shape[0]):
        for k in range(params.N):
            dist[b, k] = results[index][0]
            count[b, k] = results[index][1]
            index += 1

    df1 = pd.DataFrame(dist)
    df2 = pd.DataFrame(count)
    df1.to_csv(
        wd+'/discrete/results/augment_delta_' + str(params.size) + '_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    df2.to_csv(
        wd+'/discrete/results/augment_counter_' + str(params.size) + '_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    #print("Delta: mean, std, min, max, median\n")
    #print(np.mean(dist, axis=1), np.std(dist, axis=1), np.min(dist, axis=1), np.max(dist, axis=1), np.median(dist,
    #                                                                                                       axis=1))
    #print("\n")
    #print("Counter: mean, std, min, max, median\n")
    #print(np.mean(count, axis=1), np.std(count, axis=1), np.min(count, axis=1), np.max(count, axis=1), np.median(count,
    #                                                                                                         axis=1))
