

for replay in 1 # todo: add values
do
  for environment in CartPole-v0 MountainCar-v0 LunarLander-v2 BipedalWalker-v2 CarRacing-v0 FrozenLake8x8-v0 Blackjack-v0
  do
     python3 main.py --environment $environment --episodes 2 --run_name $environment
  done
done

# .... Breakout-v0 MsPacman-v0 Reverse-v0 Ant-v2 FetchPickAndPlace-v1