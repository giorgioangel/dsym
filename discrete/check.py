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
import numba as nb
from grids import StochasticGrid, Grid
from statsmodels.stats.power import GofChisquarePower
import pandas as pd
import os

wd = os.getcwd()

def collect(STEPS, size, environment):
    env = environment(size)

    X = np.zeros((STEPS, 1), dtype=np.int32)
    Y = np.zeros((STEPS, 1), dtype=np.int32)
    A = np.zeros((STEPS, 1), dtype=np.int32)
    R = np.zeros((STEPS, 1), dtype=np.int32)

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


@nb.njit((nb.int64, nb.int32[:, :], nb.int32[:, :], nb.int32[:, :]), fastmath=True, nogil=True, cache=True)
def learn_transition(size, X, Y, A):
    N = np.zeros((4, size ** 2, size ** 2), dtype=np.float32)
    T = np.zeros((4, size ** 2, size ** 2), dtype=np.float32)
    for i in range(X.shape[0]):
        N[A[i, 0], X[i, 0], Y[i, 0]] += 1

    for a in range(4):
        for s in range(size ** 2):
            somma = np.sum(N[a, s, :])
            if somma == 0:
                T[a, s] = size ** (-2.) * np.ones(size ** 2)
            else:
                T[a, s] = N[a , s]/somma
    return T, N


@nb.njit((nb.float32[:, :, :], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :]), fastmath=True, nogil=True, cache=True)
def calculate_nu_det(T, Xsym, Asym, Ysym):
    U = np.zeros(Xsym.shape[0])
    for i in range(Xsym.shape[0]):
        if T[Asym[i, 0], Xsym[i, 0], Ysym[i, 0]] == 1:
            U[i] = 1
    nu = np.sum(U)/Xsym.shape[0]
    return nu


#@nb.njit((nb.float32[:, :, :], nb.float32[:, :, :], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :],
#          nb.int32[:, :], nb.int32), fastmath=True, nogil=True, cache=True)
def calculate_nu_stoch(T, N, X, A, Y, Xsym, Asym, Ysym, samples):
    '''
    if env == 'det':
        testenv = Grid(size)
        testenv.reset()
    else:
        testenv = Grid(size)
        testenv.reset()
    if symm == 'TRSAI':
        def symmetrizer(states, s, a, ns):
            return [s for i in range(states)]
    elif symm == 'SDAI':
        def symmetrizer(states, s, a, ns):
            return [sp for sp in range(states)]
    elif symm == 'ODAI':
        def symmetrizer(states, s, asymm, ns):
            next_states = []
            for sp in range(states):
                testenv.state = sp
                _, _, _, _, = testenv.step(asymm)
                ts, _, _, _, = testenv.step(asymm)
                next_states.append(ts)
            return next_states
    elif symm == 'ODWA':
        def symmetrizer(states, s, asymm, ns):
            next_states = []
            if asymm == 3:
                a = 1
            elif asymm == 2:
                a = 0
            elif asymm == 0:
                a = 3
            elif asymm == 1:
                a = 2
            for sp in range(states):
                testenv.state = sp
                _, _, _, _, = testenv.step(a)
                ts, _, _, _, = testenv.step(a)
                next_states.append(ts)
            return next_states
    elif symm == 'TI':
        def symmetrizer(states, s, asymm, ns):
            next_states = []
            for sp in range(states):
                testenv.state = sp
                ts, _, _, _, = testenv.step(asymm)
                next_states.append(ts)
            return next_states
    elif symm == 'TIOD':
        def symmetrizer(states, s, asymm, ns):
            return [s for i in range(states)]
    else:
        raise ValueError("Error!!")
    '''
    counter = np.zeros(X.shape[0], dtype=np.int32)
    stat = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(X.shape[0]):
        if np.sum(N[Asym[i, 0] , Xsym[i, 0]]) == 0:
            stat[i] = 0.5
        else:
            zero1 = np.argwhere(T[Asym[i, 0], Xsym[i, 0]] == 0).flatten().tolist()
            zero2 = np.argwhere(T[A[i, 0], X[i, 0]] == 0).flatten().tolist()
            zero1.append(Ysym[i, 0])
            zero2.append(Y[i, 0])
            T1 = np.sort(np.delete(T[Asym[i, 0], Xsym[i, 0]], zero1, None))
            T2 = np.sort(np.delete(T[A[i, 0], X[i, 0]], zero2, None))
            if len(T1) != 0 and len(T2) != 0:
                #if len(zero1) == len(zero2):
                d1 = abs(T1[0]-T2[-1])
                d2 = abs(T1[-1]-T2[0])
                #else:
                #    d1 = abs(T1[0]-T2[0])
                #    d2 = abs(T1[-1]-T2[-1])
                d = max(d1, d2)
            else:
                #d = 1./T.shape[1]
                d=0.5
            stat[i] = max(0, 1-max(abs(T[Asym[i, 0], Xsym[i, 0], Ysym[i, 0]] - T[A[i, 0], X[i, 0], Y[i, 0]]), d))
        #permutazione = symmetrizer(T.shape[1], X[i, 0], Asym[i, 0], Y[i, 0])
        #Tperm = T[Asym[i,0], Xsym[i,0], permutazione]
        #stat[i] = 1- 2*np.max(np.abs(T[A[i, 0], X[i, 0]] - Tperm))
        if np.sum(N[A[i, 0], X[i, 0]]) >= samples and np.sum(N[Asym[i, 0], Xsym[i, 0]]) >= samples:
            counter[i] = 1
    nu = np.mean(stat)
    c = np.mean(counter)
    return nu, c


def transformation_output(b, size, min_batch, symmetry, stoch_type):
    if stoch_type == 'det':
        X, A, R, Y = collect(min_batch * (b + 1), size, Grid)
    elif stoch_type == 'stoch':
        X, A, R, Y = collect(min_batch * (b + 1), size, StochasticGrid)
    else:
        raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")

    # 0 UP 1 DOWN 2 LEFT 3 RIGHT
    T, N = learn_transition(size, X, Y, A)
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

    #if stoch_type == 'det':
    #   nu = calculate_nu_det(T, Xsym, Asym, Ysym)
    #elif  stoch_type == 'stoch':
    chipower = GofChisquarePower()
    samples = chipower.solve_power(effect_size=0.3, power=0.95, alpha=0.05, n_bins=T.shape[1])
    nu, c = calculate_nu_stoch(T, N, X, A, Y, Xsym, Asym, Ysym, samples)
    #else:
    #    raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")
    return nu, c


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
                        help='Number of steps')

    parser.add_argument('-k', type=str, action='store', dest='k',
                        help='Symmetry')

    params = parser.parse_args()

    nu = np.zeros((params.s, params.N))
    c = np.zeros((params.s, params.N))
    size = params.size
    min_batch = params.t
    inputs = [(b, size, min_batch, params.k, params.e) for b in range(params.s) for k in range(params.N)]
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(transformation_output, inputs)
    pool.close()
    pool.join()
    index = 0
    for b in range(params.s):
        for k in range(params.N):
            nu[b, k] = results[index][0]
            c[b, k] = results[index][1]
            index += 1

    df1 = pd.DataFrame(nu)
    df2 = pd.DataFrame(c)
    df1.to_csv(
        wd+'/discrete/results/check_nu_' + str(params.size) + '_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    df2.to_csv(
        wd+'/discrete/results/check_counter_' + str(params.size) + '_' + params.e + '_' + params.k + '_' + str(params.t) + '_' + str(params.N) + '_' + str(
            params.s) + '.csv')
    #print(np.mean(nu, axis=1), np.std(nu, axis=1))
    #print(np.mean(c, axis=1), np.std(c, axis=1))


