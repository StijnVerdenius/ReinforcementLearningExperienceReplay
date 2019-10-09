# todo
import argparse
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

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
            batches = []
            for i, element in enumerate(values):
                if (i % self.arguments.eval_freq) == 0:
                    batches.append([])
                if element is not None:
                    batches[-1].append(element)

            try:

                mean = np.array([np.mean(sublist) for sublist in batches])
                std = np.array([np.std(sublist) for sublist in batches])
                upper = mean + std
                lower = mean - std

                plt.title(key)
                plt.fill_between(range(len(mean)), lower, upper, alpha=0.5, color="gray")
                plt.plot(upper, color="black", alpha=0.9, linestyle=":")
                plt.plot(lower, color="black", alpha=0.9, linestyle=":")
                plt.xticks(range(len(mean)), [self.arguments.eval_freq*i for i in range(len(mean))])
                plt.xlabel("Episodes")

                plt.plot(mean, color="black")
                plt.show()
            except Exception as e:
                print(key, e)

    def animate(self, n=10):
        self.agent.eval()
        for i in range(n):
            pass

