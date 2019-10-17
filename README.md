# Reinforcement-Learning Experience Replay Experiment

This repository contains a training script for training a Deep Q network on several OpenAI gym environments.
It aims at comparing different memory replay methods across environments and see where which performs better.
Structure is modular so multiple additional agents, environments and replay methods can be added later. 
Moreover, training generates unique folder for output and stats of that run. 

##### Run the main.py script to train, with the proper environment (python3).

*Additional parameters are:* 

    --episodes     default=100   type=int   max number of episodes
    --eval_freq    default=10    type=int   evaluate every x batches
    --batch_size   default=64    type=int   size of batches
    --hidden_dim   default=64    type=int   size of hidden dimension
    --run_name     default=""    type=str   extra identification for run
    --plot         default=False type=bool  plot when done
    --device       default=cuda  type=str   none/cpu/cuda
    --seed         default=42    type=int   random seed
    
    --loss         default="SmoothF1Loss"   loss-function model name
    --agent_model  default="QNetworkAgent"  agent model name
    
    --optimizer                  type=str   ADAM, RMSprop or SGD
    --environment                type=str   BipedalWalker-v2, CartPole-v0
                                            LunarLander-v2, MountainCar-v0
                                            FrozenLakeExtension or 
                                            Breakout-ram-v0                                 
    --replay                     type=str   PriorityReplay, RandomReplay,
                                            RandomTrajectoryReplay or
                                            RecentReplay

*see also:*
 
 ###### python3 main.py -h