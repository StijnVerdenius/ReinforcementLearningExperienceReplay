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
                 device,
                 patience: int, # todo: typing
                 arguments: argparse.Namespace):

        self.arguments = arguments
        self.optimizer = optimizer
        self.agent = agent
        self.loss = loss
        self.memory = replay_memory
        self.environment = environment

        self._log_header = '  Time Epoch Iteration    Progress (%Epoch) | Train Loss Train Acc. | Valid Loss Valid Acc. | Best | VAE-stuff'
        self._log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>10.6f} {:>10.6f} | {:>10.6f} {:>10.6f} | {:>4s} | {:>4s}'.split(
                ','))
        self._start_time = time.time()
        self._device = device
        self._patience = patience

        # init current runs timestamp
        DATA_MANAGER.set_date_stamp(addition=arguments.run_name)

        # initialize tensorboardx
        self.writer = SummaryWriter(os.path.join(GITIGNORED_DIR, RESULTS_DIR, DATA_MANAGER.stamp, SUMMARY_DIR))

    def train(self):
        """
                 main training function
                """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self.arguments)

        # data gathering
        progress = []

        epoch = 0

        try:

            print(f"{PRINTCOLOR_BOLD}Started training with the following config:{PRINTCOLOR_END}\n{self.arguments}\n\n")
            print(self._log_header)

            best_metrics = (math.inf, 0)
            patience = self._patience
            # run
            for epoch in range(self.arguments.epochs):

                # do epoch
                epoch_progress, best_metrics, patience = self._epoch_iteration(epoch, best_metrics, patience)

                # add progress-list to global progress-list
                progress += epoch_progress

                # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                DATA_MANAGER.save_python_obj(progress,
                                             os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, PROGRESS_DIR,
                                                          "progress_list"),
                                             print_success=False)

                # flush prints
                sys.stdout.flush()

                if patience == 0:
                    break

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            save_models([self.agent], f"KILLED_at_epoch_{epoch}")
            return False
        except Exception as e:
            print(e)
            save_models([self.agent], f"CRASH_at_epoch_{epoch}")
            raise e

        # flush prints
        sys.stdout.flush()

        # example last save
        save_models([self.agent], "finished")
        return True

    def _epoch_iteration(self, epoch, best_metrics, patience):

        # todo: implement stuff
        raise NotImplementedError

