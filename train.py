from models.agents import ParentAgent
from models.losses import ParentLoss
from models.replays import ParentReplay

import argparse
import sys
import time
import math
from datetime import datetime
from typing import List, Tuple

import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from models import GeneralModel
from utils.constants import *
from utils.model_utils import save_models, calculate_accuracy
from utils.system_utils import setup_directories, save_codebase_of_run


class Trainer:

    def __init__(self,
                 environment,  # todo: typing
                 replay_memory: ParentReplay,
                 loss: ParentLoss,
                 agent: ParentAgent,
                 optimizer: Optimizer,
                 arguments: argparse.Namespace):

        self.arguments = arguments
        self.optimizer = optimizer
        self.agent = agent
        self.loss = loss
        self.memory = replay_memory
        self.environment = environment

    def train(self):
        pass

