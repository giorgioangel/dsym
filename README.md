# Exploiting Expert Guided Symmetry detection in Offline Reinforcement Learning

This is the official repository of the paper "Exploiting Expert Guided Symmetry detection in Offline Reinforcement Learning" submitted to NeurIPS 2022.

## Table of contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [What's included](#whats-included)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)


## Requirements
The code has been tested on the minimal version reported.

```shell script
conda install --file requirements.txt
```


## Quick start
To check if a symmetry is present in the categorical environment:
```shell script
python discrete/check.py -e env_type -k transformation -l grid_size -t min_traj -b batches -s different_batch_sizes 
```

Similar reasoning for the other scripts, look at the code.

Example:
```shell script
python continuous/cartpole/policy.py -e stoch -b 30 -s 10 -t 1000 -k SAR
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

Code and documentation copyright 2022 the authors. Code released under the [GNU-AGPLv3 License]

Enjoy