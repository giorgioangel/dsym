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

import gym
import numpy as np


class StochasticGrid(gym.Env):

    def __init__(self, grid_size):
        self.grid_size = grid_size

        self.goal = (grid_size//2)**2

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)
        self.state = 0
        self.observation_space = gym.spaces.Discrete(self.grid_size**2)

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def move(self, a):
        if a == 0:
            temp = self.state + self.grid_size
            if temp // self.grid_size < self.grid_size:
                self.state = temp
            else:
                self.state %= self.grid_size

        elif a == 1:
            temp = self.state - self.grid_size
            if temp // self.grid_size >= 0:
                self.state = temp
            else:
                self.state += self.grid_size * (self.grid_size - 1)

        elif a == 2:
            temp = self.state - 1
            mod = self.state // self.grid_size
            if temp // self.grid_size == mod:
                self.state = temp
            else:
                self.state += (self.grid_size - 1)

        elif a == 3:
            temp = self.state + 1
            mod = self.state // self.grid_size
            if temp // self.grid_size == mod:
                self.state = temp
            else:
                self.state = mod * self.grid_size

    def step(self, a):
        info = {}
        reward = -1.
        random_number = np.random.uniform(0, 1)
        # 0 UP # 1 DOWN # 2 LEFT # 3 RIGHT
        if a == 0:
            if random_number <= 0.6:
                self.move(0)
            elif 0.6 < random_number <= 0.8:
                self.move(1)
            elif 0.8 < random_number <= 0.9:
                self.move(2)
            else:
                self.move(3)

        elif a == 1:
            if random_number <= 0.6:
                self.move(1)
            elif 0.6 < random_number <= 0.8:
                self.move(0)
            elif 0.8 < random_number <= 0.9:
                self.move(2)
            else:
                self.move(3)

        elif a == 2:
            if random_number <= 0.6:
                self.move(2)
            elif 0.6 < random_number <= 0.8:
                self.move(3)
            elif 0.8 < random_number <= 0.9:
                self.move(0)
            else:
                self.move(1)

        elif a == 3:
            if random_number <= 0.6:
                self.move(3)
            elif 0.6 < random_number <= 0.8:
                self.move(2)
            elif 0.8 < random_number <= 0.9:
                self.move(0)
            else:
                self.move(1)

        else:
            raise ValueError("Received invalid action which is not part of the action space")

        if self.state == self.goal:
            reward = 1
            self.done = True

        return self.state, reward, self.done, info

    def close(self):
        pass


class Grid(gym.Env):
    def __init__(self, grid_size):
        self.grid_size = grid_size

        self.goal = (grid_size)**2-1

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)
        self.state = 0
        self.observation_space = gym.spaces.Discrete(self.grid_size**2)

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def step(self, a):
        info = {}
        reward = -1.
        # 0 UP # 1 DOWN # 2 LEFT # 3 RIGHT
        if a == 0:
            temp = self.state + self.grid_size
            if temp // self.grid_size < self.grid_size:
                self.state = temp
            else:
                self.state %= self.grid_size

        elif a == 1:
            temp = self.state - self.grid_size
            if temp // self.grid_size >= 0:
                self.state = temp
            else:
                self.state += self.grid_size*(self.grid_size -1)

        elif a == 2:
            temp = self.state - 1
            mod = self.state // self.grid_size
            if temp // self.grid_size == mod:
                self.state = temp
            else:
                self.state += (self.grid_size - 1)

        elif a == 3:
            temp = self.state + 1
            mod = self.state // self.grid_size
            if temp // self.grid_size == mod:
                self.state = temp
            else:
                self.state = mod*self.grid_size

        else:
            raise ValueError("Received invalid action which is not part of the action space")

        if self.state == self.goal:
            reward = 1.
            self.done = True

        return self.state, reward, self.done, info

    def close(self):
        pass