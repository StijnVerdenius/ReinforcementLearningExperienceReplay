

for replayMemory in PriorityReplay RandomReplay RandomTrajectoryReplay RecentReplay  # todo: add values
do
  for environment in "FrozenLakeExtension"
  do
    for seed in 42 54 23 65 34
      do
        python3 main.py --environment $environment --episodes 200 --run_name "_"$environment"_"$replayMemory"_"$seed"_"  --seed $seed --replay $replayMemory
      done
  done
done