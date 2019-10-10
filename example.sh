

for replayMemory in PriorityReplay RandomReplay RandomTrajectoryReplay RecentReplay  # todo: add values
do
  for environment in CartPole-v0
  do
    for seed in 42 54 23 65 34
      do
        python3 main.py --environment $environment --episodes 200 --run_name $environment  --seed $seed --replay $replayMemory
      done
  done
done

# .... Breakout-v0 MsPacman-v0 Reverse-v0 Ant-v2 FetchPickAndPlace-v1