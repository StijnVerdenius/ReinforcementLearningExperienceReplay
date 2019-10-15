

import numpy as np
import gym


class FrozenLakeExtension:


    def __init__(self, **kwargs):
        self._env = gym.envs.make("FrozenLake8x8-v0")
        self.action_space = self._env.action_space
        self.observation_space = np.zeros((64))

    def _to_one_hot(self, element):
        state = np.zeros(64)
        state[element] = 1
        return state

    def seed(self, seed):
        self._env.seed(seed)

    def reset(self):
        return self._to_one_hot(self._env.reset())

    def step(self, action):
        s_next, a, b, c = self._env.step(action)
        return self._to_one_hot(s_next), a,b,c


