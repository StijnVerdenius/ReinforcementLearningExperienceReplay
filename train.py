import argparse
import math
import sys
import time

import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer

from models.agents import ParentAgent
from models.losses import ParentLoss
from models.replays import ParentReplay, PriorityReplay
from utils.constants import *
from utils.model_utils import save_models
from utils.system_utils import setup_directories, save_codebase_of_run, autodict, report_error


class Trainer:

    def __init__(self,
                 environment,  # todo: typing
                 replay_memory: ParentReplay,
                 loss: ParentLoss,
                 agent: ParentAgent,
                 optimizer: Optimizer,
                 device,
                 patience: int,  # todo: typing
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
        self._global_steps = 0

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

        episode = 0

        try:

            print(f"{PRINTCOLOR_BOLD}Started training with the following config:{PRINTCOLOR_END}\n{self.arguments}\n\n")
            # print(self._log_header)

            best_metrics = (math.inf, 0)
            patience = self._patience
            # run
            for episode in range(self.arguments.episodes):

                # do epoch
                episode_durations, losses, reward = self._episode_iteration()

                # add progress-list to global progress-list

                progress += [{**autodict(episode_durations, losses, episode, reward), **self._collect_metrics(None)}]
                if (episode % self.arguments.eval_freq) == 0:

                    # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                    DATA_MANAGER.save_python_obj(progress,
                                                 os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, PROGRESS_DIR,
                                                              "progress_list"),
                                                 print_success=False)

                    self._log(episode, progress[-1])

                # flush prints
                sys.stdout.flush()

                if patience == 0:  # todo: implement patience and maximum elapsed time
                    break

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
            save_models([self.agent], f"KILLED_at_epoch_{episode}")
        except Exception as e:
            report_error(e, self.agent, episode)

        # flush prints
        sys.stdout.flush()

        # example last save
        save_models([self.agent], "finished")

        return DATA_MANAGER.load_python_obj(os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, PROGRESS_DIR,
                                                         "progress_list"))

    def _step_train(self, epoch=0, best_metrics=[], patience=0):

        # todo: use parameters: elias

        # don't learn without some decent experience
        if len(self.memory) < self.arguments.batch_size:
            return None

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.arguments.batch_size)

        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, reward, next_state, done = zip(*transitions)

        # loss is measured from error between current and newly expected Q values
        loss = self._compute_loss(state, action, reward, next_state, done)

        # backpropagation of loss to Neural Network (PyTorch magic)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _collect_metrics(self, *stuff):
        return {"dummy_metric": 0 }  # todo elias

    def _compute_q_val(self, state, action):
        state = self._restore_state(state)
        q = self.agent.forward(state)
        if isinstance(self.agent.actions, tuple):
            return q
        return q[torch.arange(len(action)), action]

    def _compute_target(self, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
        best_action = self._select_action(next_state, 0)
        if isinstance(self.agent.actions, tuple):
            return torch.mul(
                (self.arguments.discount_factor * self._compute_q_val(next_state, best_action)) + reward.unsqueeze(0).T,
                (1 - done.float()).unsqueeze(0).T)
        else:
            return torch.mul(reward + self.arguments.discount_factor * self._compute_q_val(next_state, best_action),
                             1 - done.float())

    def _restore_state(self, state):
        if (isinstance(state, (int, float, np.int64))):
            state = [state]
        elif (len(state.shape) == 1 and isinstance(state, torch.Tensor)):
            state = state.unsqueeze(0).T.to(self._device)
        return state

    def _select_action(self, state, epsilon):
        with torch.no_grad():
            state = self._restore_state(state)
            # q = self.agent.forward(torch.Tensor(state).float())
            if (not isinstance(state, torch.Tensor)):
                state = torch.Tensor(state).float().to(self._device)
            q = self.agent.forward(state)
            if isinstance(self.agent.actions, tuple):
                return q.tolist()
            else:
                if np.random.random() < epsilon:
                    return np.random.choice(range(self.agent.actions[0]))
                index = torch.argmax(q, -1)
                return index.tolist()

    def _episode_iteration(self):

        print("new")

        step = 0
        s = self.environment.reset()

        summed_reward = 0

        while True:
            action = self._select_action(s, self._get_epsilon())
            s_next, r, done, _ = self.environment.step(action)


            if (not r == 0 ):
                print(r)

            summed_reward += r

            if type(self.memory) is PriorityReplay.PriorityReplay:
                # Priority replay memory push
                error = self._compute_loss([s], [action], [r], [s_next], [done]).item()
                self.memory.push((s, action, r, s_next, done), error)
            else:
                # Default memory push
                self.memory.push((s, action, r, s_next, done))

            step += 1
            self._global_steps += 1

            s = s_next

            loss = self._step_train()

            if done:
                break

        return step, loss, summed_reward

    def _get_epsilon(self):
        if self._global_steps >= 1000:
            return 0.05
        else:
            return np.linspace(1, 0.05, 1000)[self._global_steps]

    def _log(self, episode, progress_item):

        # todo: elias
        log_string = ""

        for key, value in progress_item.items():
            if isinstance(value, (float, int)):
                log_string += f",\t {key}: {value:>7.6f}"

        print(f"Episode {episode}{log_string}")

    def _compute_loss(self, state, action, reward, next_state, done):

        # convert to PyTorch and define types
        state = torch.tensor(state, dtype=torch.float).to(self._device)
        action = torch.tensor(action, dtype=torch.int64).to(self._device) # Need 64 bit to use them as index
        next_state = torch.tensor(next_state, dtype=torch.float).to(self._device)
        reward = torch.tensor(reward, dtype=torch.float).to(self._device)
        done = torch.tensor(done, dtype=torch.uint8).to(self._device) # Boolean

        # compute the q value
        q_val = self._compute_q_val(state, action)

        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = self._compute_target(reward, next_state, done)

        # loss is measured from error between current and newly expected Q values
        loss = self.loss(q_val, target)
        return loss
