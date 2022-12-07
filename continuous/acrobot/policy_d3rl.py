#!/usr/bin/env python
#from augment import collect_d3rl
import pandas as pd
import os
from typing import Any, Callable, List, Optional, Tuple, Union
import d3rlpy
import gym
import numpy as np
from d3rlpy.algos import DiscreteCQL, DQN
from d3rlpy.models.encoders import DefaultEncoderFactory
from typing_extensions import Protocol
from d3rlpy.dataset import Episode
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation


def collect_d3rl(STEPS, mode):
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
    terminals = np.zeros((STEPS, 1))
    ep_terminals = np.zeros((STEPS, 1))
    m = 0
    for k in range(STEPS):
        done = False
        X[k + m] = env.reset()
        while not done and k + m < STEPS:
            a = env.action_space.sample()
            A[k + m] = a
            Y[k + m], R[k + m], done, _ = env.step(a)
            if done:
                terminals[k+m] = 1
            if not done and k + m + 1 < STEPS:
                X[k + m + 1] = Y[k + m]
                m += 1

        if k + m + 1 == STEPS:
            ep_terminals[k+m] = 1
            break
    return X, A, R, Y, terminals, ep_terminals


class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
            self,
            x: Union[np.ndarray, List[Any]],
            action: Union[np.ndarray, List[Any]],
            with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


def my_evaluate_on_environment(
        env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.
    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.
    .. code-block:: python
        import gym
        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment
        env = gym.make('CartPole-v0')
        scorer = evaluate_on_environment(env)
        cql = CQL()
        mean_episode_return = scorer(cql)
    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.
    Returns:
        scoerer function.
    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards)), float(np.std(episode_rewards, ddof=1))

    return scorer


if __name__ == '__main__':
    wd = os.getcwd()
    d3rlpy.seed(181991)

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

    parser.add_argument('-g', type=bool, action='store', dest='gpu',
                        help='GPU')

    params = parser.parse_args()
    perfdqn = np.zeros((params.s, params.N))
    perfcql = np.zeros((params.s, params.N))

    perfstd = np.zeros((params.s, params.N, 2))

    rwddqn = np.zeros((params.s, params.N))
    rwddqn_sym = np.zeros((params.s, params.N))
    rwdcql = np.zeros((params.s, params.N))
    rwdcql_sym = np.zeros((params.s, params.N))

    stddqn = np.zeros((params.s, params.N))
    stddqn_sym = np.zeros((params.s, params.N))
    stdcql = np.zeros((params.s, params.N))
    stdcql_sym = np.zeros((params.s, params.N))

    for m in range(params.s):
        for k in range(params.N):
            print(m, k)
            X, A, R, Y, ter, epter = collect_d3rl((m+1)*params.t, params.e)
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            A = A.astype(np.float32)

            dataset = d3rlpy.dataset.MDPDataset(
                observations=X,
                actions=A,
                rewards=R,
                terminals=ter,
                episode_terminals=epter,
            )

            ### prepare dataset to extend
            dataset_sym = d3rlpy.dataset.MDPDataset(
                observations=X,
                actions=A,
                rewards=R,
                terminals=ter,
                episode_terminals=epter,
            )

            if params.k == 'AAVI':
                Xsym = np.copy(X)
                Xsym[:, 1] = - Xsym[:, 1]
                Xsym[:, 3] = - Xsym[:, 3]
                Xsym[:, 4] = - Xsym[:, 4]
                Xsym[:, 5] = - Xsym[:, 5]
                Asym = -np.copy(A)+2  # 0 LEFT -> 2 RIGHT, 1 NOTHING -> 1 NOTHING, 2 RIGHT -> 0 LEFT
                eptersym = np.copy(epter)

            elif params.k == 'CAVI':
                Xsym = np.copy(X)
                Xsym[:, 0] = - Xsym[:, 0]
                Xsym[:, 2] = - Xsym[:, 2]
                Xsym[:, 4] = - Xsym[:, 4]
                Xsym[:, 5] = - Xsym[:, 5]
                Asym = -np.copy(A)+2  # 0 LEFT -> 2 RIGHT, 1 NOTHING -> 1 NOTHING, 2 RIGHT -> 0 LEFT
                eptersym = np.copy(epter)

            elif params.k == 'AI':
                Xsym = np.copy(X)
                Asym = -np.copy(A)+2  # 0 LEFT -> 2 RIGHT, 1 NOTHING -> 1 NOTHING, 2 RIGHT -> 0 LEFT
                eptersym = np.copy(epter)

            elif params.k == 'SSI':
                Xsym = -np.copy(X)
                Asym = np.copy(A)
                eptersym = -np.copy(epter)+1.

            else:
                raise ValueError("Select a symmetry between AAVI, CAVI, AI and SSI")

            dataset_mirror = d3rlpy.dataset.MDPDataset(
                observations=Xsym,
                actions=Asym,
                rewards=R,
                terminals=ter,
                episode_terminals=epter,
            )
            dataset_sym.extend(dataset_mirror)

            if params.e == 'det':
                cp = gym.make("Acrobot-v1")
            elif params.e == 'stoch':
                cp = gym.make("Acrobot-v1")
                cp.torque_noise_max = 0.5
            else:
                raise ValueError("You did not select a correct environment type: 'stoch' or 'det'")
            # Automatically normalize the input features and reward

            # Scaling
            mean = np.zeros(6)
            std = np.array([1., 1., 1., 1., 4*np.pi, 9*np.pi])
            scaler = d3rlpy.preprocessing.StandardScaler(mean=mean, std=std, eps=0)

            #
            #encoder = DefaultEncoderFactory(use_batch_norm=True)

            # DQN
            dqn = DQN(learning_rate=18e-6, batch_size=256, scaler=scaler, target_update_interval=4000, use_gpu=params.gpu)
            dqn.fit(dataset, n_steps=50*(m+1)*params.t, save_metrics=False)
            dqn_sym = DQN(learning_rate=18e-6, batch_size=256, scaler=scaler, target_update_interval=4000, use_gpu=params.gpu)
            dqn_sym.fit(dataset_sym, n_steps=50*(m+1)*params.t, save_metrics=False)


            # setup CQL algorithm
            cql = DiscreteCQL(learning_rate=18e-6, alpha=0.9, batch_size=256, use_gpu=params.gpu, target_update_interval=4000, scaler=scaler)
            # start training
            cql.fit(dataset, n_steps=50*(m+1)*params.t, save_metrics=False)

            cql_sym = DiscreteCQL(learning_rate=18e-6, alpha=0.9, batch_size=256, encoder=params.gpu, use_gpu=True, target_update_interval=4000, scaler=scaler)
            cql_sym.fit(dataset_sym, n_steps=50*(m+1)*params.t, save_metrics=False)

            rwddqn[m,k], stddqn[m,k] = my_evaluate_on_environment(cp, n_trials=100)(dqn)
            rwddqn_sym[m,k], stddqn_sym[m,k] = my_evaluate_on_environment(cp, n_trials=100)(dqn_sym)
            rwdcql[m,k], stdcql[m,k] = my_evaluate_on_environment(cp, n_trials=100)(cql)
            rwdcql_sym[m,k], stdcql_sym[m,k] = my_evaluate_on_environment(cp, n_trials=100)(cql_sym)

            perfdqn[m, k] = rwddqn_sym[m,k] - rwddqn[m,k]
            perfcql[m, k] = rwdcql_sym[m,k] - rwdcql[m,k]

            df1 = pd.DataFrame(perfdqn)
            df1.to_csv(
                wd + '/continuous/acrobot/results/polmean_dqn_' + params.e + '_' + params.k + '_' + str(
                    params.t) + '_' + str(params.N) + '_' + str(
                    params.s) + '.csv')
            del df1

            df4 = pd.DataFrame(perfcql)
            df4.to_csv(
                wd + '/continuous/acrobot/results/polmean_cql_' + params.e + '_' + params.k + '_' + str(
                    params.t) + '_' + str(params.N) + '_' + str(
                    params.s) + '.csv')
            del df4

            np.savez_compressed(wd + '/continuous/acrobot/results/polmean_' + params.e + '_' + params.k + '_' + str(
                    params.t) + '_' + str(params.N) + '_' + str(
                    params.s) + '.npz', rdqn=rwddqn, sdqn=stddqn, rcql=rwdcql, scql=stdcql, rdqns=rwddqn_sym,
                                sdqns=stddqn_sym, rcqls=rwdcql_sym, scqls=stdcql_sym, allow_pickle=True)