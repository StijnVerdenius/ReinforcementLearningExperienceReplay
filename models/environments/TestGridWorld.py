import random

import numpy as np


class dummy:

    def __init__(self):
        self.n = 4


class TestGridWorld:

    def __init__(self, **kwargs):

        self.target = np.array([3, 9])
        self.env = np.array([[0] * 5] * 10)

        self.observation_space = np.zeros(2)
        self.action_space = dummy()
        self.current_position = np.array([0, 0])

    def seed(self, seed):
        pass

    def reset(self):
        self.current_position = np.array([random.randint(0, x) for x in [len(self.env[0]), len(self.env[1])]])
        return self.current_position

    def step(self, action):

        old = self.current_position


        if (action == 1):
            self.current_position += np.array([1, 0])
        elif (action == 2):
            self.current_position -= np.array([1, 0])
        elif (action == 3):
            self.current_position -= np.array([0, 1])
        else:
            self.current_position += np.array([0, 1])

        base_reward = -1

        if (self.current_position == self.target).all():
            print("done")
            return self.current_position, 10, True, None

        elif (self.current_position[0] < 0 or self.current_position[0] > len(self.env[0])):
            base_reward -= abs(self.current_position[0])
            base_reward += self.step(self.opposite_step(action))[1]
        elif (self.current_position[1] < 0 or self.current_position[1] > len(self.env[1])):
            print()
            base_reward -= abs(self.current_position[1])
            base_reward += self.step(self.opposite_step(action))[1]

        print(action, self.current_position, base_reward)

        return self.current_position, base_reward, False, None

    def opposite_step(self, action):
        if (action == 1):
            return 2
        elif (action == 2):
            return 1
        elif action == 0:
            return 3
        else:
            return 0
