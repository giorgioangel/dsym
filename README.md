# Expert-guided Symmetry Detection in MDPs

This is the official repository of the paper "Expert-guided Symmetry Detection in Markov Decision Processes" accepted to the 14th International Conference on Agents and Artificial Intelligence - ICAART 2022 ( arXiv preprint https://arxiv.org/abs/2111.10297 ).

Some part of the code concerns a preliminary version of an extension of the work submitted to another venue.

## Table of contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [What's included](#whats-included)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)


## Requirements
The code has been tested on the minimal version reported.

```shell script
pip install --user -r requirements.txt
```

1. gym==0.19.0
2. keras==2.7.0
3. numpy==1.20.0
4. numba==0.54.1
5. pandas==1.3.4
6. scikit-learn==1.0.1
7. scipy==1.7.2
8. stable-baselines3==1.3.0
9. tensorflow-gpu==2.7.0
10. tensorflow-probability==0.12.2
11. torch==1.10.0
12. tqdm==4.62.3



## Quick start
To check if a symmetry is present in the categorical environment:
```shell script
python discrete/check.py -e env_type -k transformation -l grid_size -t min_traj -b batches -s different_batch_sizes 
```

Similar reasoning for the other scripts, look at the code.

Example:
```shell script
python discrete/check.py -e det -k TRSAI -l 10 -t 1000 -b 100 -s 10
```
Output:
```shell script
check_counter_10_det_TRSAI_1000_100_10.csv
```

## Parameters
- -e env_type: string - det, stoch
- -b batch: integer - number of different batches of fixed size
- -k transformation: string - read the label in the paper (TRSAI, SDAI, etc.)
- -t min_traj: integer - starting number of different trajectories in a batch
- -s different_batch_sizes: integer - number of different batch sizes (multiples of min_traj)
- -q quantile: float - quantile to compute the threshold in continuous environment
- -l grid_size: integer - size of the meshing of the categorical grid

## What's included

```text
dsym/
├── discrete/
│   ├── results/
│   │   ├── check_counter_10_det_TRSAI_1000_100_10.csv
│   │   ├── (other results)
│   │   └── (other results)
│   ├── check.py
│   ├── grids.py
│   ├── augment.py
│   ├── solvemdp.py
│   └── solvers.py
├── continuous/
│   ├── cartpole/
│   │   ├── results/
│   │   │   ├── (other results)
│   │   │   └── (other results)
│   │   ├── check.py
│   │   ├── augment.py
│   │   └── policy.py
│   └── acrobot/
│       ├── results/
│       │   ├── (other results)
│       │   └── (other results)
│       ├── check.py
│       ├── augment.py
│       └── policy.py
├── requirements.txt
├── README.md  # This file
└── LICENSE # GNU Affero General Public License

```

## Creators

**Creator 1**

Giorgio Angelotti

## Copyright and license

Code and documentation copyright 2021 the authors. Code released under the [GNU-AGPLv3 License]

Enjoy