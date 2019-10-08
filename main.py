import argparse
import random
import sys

import numpy as np

from train import Trainer
from utils.constants import *
from utils.model_utils import find_right_model
from utils.system_utils import ensure_current_directory


def main(arguments: argparse.Namespace):
    device = arguments.device
    if arguments.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # for reproducibility
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(arguments.seed)

    replay_memory = find_right_model(REPLAY_DIR, arguments.replay, capacity=arguments.replay_capacity, device=device,
                                     example_param="example_value")
    environment = find_right_model(ENV_DIR, arguments.environment, device=device, example_param="example_value")
    agent = find_right_model(AG_DIR, arguments.agent_model, device=device, num_hidden=arguments.hidden_dim,
                             actions=environment.action_space, state_size=environment.observation_space)

    if arguments.test_mode:

        pass  # todo

    else:

        loss = find_right_model(LOSS_DIR, arguments.loss, device=device)

        optimizer = find_right_model(OPTIMS, arguments.optimizer, params=agent.parameters(), lr=arguments.learning_rate)

        Trainer(environment, replay_memory, loss, agent, optimizer, device, arguments.patience, arguments).train()


def parse() -> argparse.Namespace:
    """ does argument parsing """

    parser = argparse.ArgumentParser()


    # int
    parser.add_argument('--episodes', default=500, type=int, help='max number of episodes')
    parser.add_argument('--eval_freq', default=10, type=int, help='evaluate every x batches')
    parser.add_argument('--saving_freq', default=1, type=int, help='save every x epochs')  # todo: implement
    parser.add_argument('--batch_size', default=64, type=int, help='size of batches')
    parser.add_argument('--hidden_dim', default=64, type=int, help='size of batches')

    parser.add_argument('--max_training_minutes', default=24 * 60, type=int,
                        help='max mins of training be4 save-and-kill')  # todo: implement

    # float
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--discount_factor', default=0.8, type=float, help='discount factor')
    parser.add_argument('--replay_capacity', default=10000, type=int, help='max capcity of replay buffer')

    # string
    # parser.add_argument('--environment', default="BipedalWalker-v2", type=str, help='classifier model name')
    parser.add_argument('--environment', default="CartPole-v0", type=str, help='classifier model name')
    # parser.add_argument('--environment', default="LunarLander-v2", type=str, help='classifier model name')
    # parser.add_argument('--environment', default="MountainCar-v0", type=str, help='classifier model name')
    # parser.add_argument('--environment', default="FrozenLake8x8-v0", type=str, help='classifier model name')

    parser.add_argument('--replay', default="PriorityReplay", type=str, help='generator model name')
    parser.add_argument('--loss', default="SmoothF1Loss", type=str, help='loss-function model name')
    parser.add_argument('--optimizer', default="ADAM", type=str, help='loss-function model name')
    parser.add_argument('--agent_model', default="QNetworkAgent", type=str, help='loss-function model name')

    parser.add_argument('--run_name', default="", type=str, help='extra identification for run')

    # bool
    parser.add_argument('--test-mode', action='store_true', help='start in train_mode')  # todo: implement

    parser.add_argument("--device", type=str,
                        help="Device to be used. Pick from none/cpu/cuda. "
                             "If default none is used automatic check will be done")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=30,
                        help="how long will the model wait for improvement before stopping training")  # todo: implement

    # todo: add whatever you like

    return parser.parse_args()


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__, "Python version:", sys.version)
    print("Working directory: ", os.getcwd())
    print("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    ensure_current_directory()
    arguments = parse()
    print(arguments)
    main(arguments)
