# todo
import argparse
from typing import List, Dict

from models.agents import ParentAgent
from models.replays import ParentReplay

import matplotlib.pyplot as plt

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
                except:
                    break

            plt.title(key)
            plt.plot(values)
            plt.show()


    def animate(self):
        pass
