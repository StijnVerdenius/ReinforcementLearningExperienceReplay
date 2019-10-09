# todo
import argparse
from typing import List, Dict

import matplotlib.pyplot as plt

from models.agents import ParentAgent
from models.replays import ParentReplay


class Plotter:

    def __init__(self,
                 environment,
                 replay_memory: ParentReplay,
                 agent: ParentAgent,
                 device,
                 statistics: List[Dict],
                 arguments: argparse.Namespace):
        self.statistics = [{key + "_train": value for key, value in statistic.items()} for statistic in statistics]
        self.arguments = arguments
        self.agent = agent
        self.memory = replay_memory
        self.environment = environment
        self._device = device

    def plot(self):
        first_element = self.statistics.pop(0)

        for key in first_element.keys():
            counter = 0
            values = []
            while True:
                try:

                    stat = self.statistics[counter]

                    values.append(stat[key])

                    counter += 1

                except IndexError:
                    break

            plt.title(key)
            plt.plot(values)
            plt.show()

    def animate(self, n=10):
        self.agent.eval()
        for i in range(n):
            # todo: animate
            pass
