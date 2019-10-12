from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

from utils.data_manager import DataManager
'''
File used to load data and analyse it.
To do so just follow the example below.
Replace id with the name of the corresponding
folder and your data will be in the data object
'''
loader = DataManager("")

import os
sorted_dirs = list(filter(lambda x: ".py" not in x,sorted(os.listdir())))

# Get all data in a dict "arguments.txt" -> dict of data
all_data = defaultdict(list)
all_runs = set()
for dir in sorted_dirs:
    args_file = dir + "/codebase/arguments.txt"
    data_file = dir + "/progress/progress_list"
    with open(args_file) as f:
        lines = f.readlines()
        filtered_lines = [line for line in lines if "seed" not in line]
        args = ",".join(filtered_lines).replace("\n","")
        all_runs.add(args)

    data = loader.load_python_obj(data_file)
    data_dict_of_lists = {k: [dic[k] for dic in data] for k in data[0]}
    all_data[args].append(data_dict_of_lists)

# all_data has now the form
# { args1 : {"episodes" : [[...]], "rewards":[[..]],...},

# Now we aggregate the data and compute mean and std_dev of each
aggr_data = AutoVivification()
for run_details, data in all_data.items():
    single_run_data = {k: [dic[k] for dic in data] for k in data[0]}
    for stat,list_of_lists in single_run_data.items():
        if stat == "losses":
            continue
        else:
            aggr_data[run_details][stat]["mean"] = np.mean(np.array(list_of_lists),axis=0)
            aggr_data[run_details][stat]["std_dev"] = np.std(np.array(list_of_lists),axis=0)

# aggr_data has the form:
# { args1: { "episodes : { "mean" : [...], "std_dev":[...] }, {"rewards" : ... }, } ...}
rows = 5
columns = 2
# plot all data
fig, axes = plt.subplots(rows, columns, figsize=(15, rows*4))
for run_dets,metrics in aggr_data.items():
    replay_type = run_dets.split("replay='")[1].split("'")[0]
    for i, (metric, stats) in enumerate(metrics.items()):
        mean = stats["mean"]
        std = stats["std_dev"]
        # upper = mean + std
        # lower = mean - std
        ax = axes[i // columns, i % columns]
        # ax.fill_between(range(len(mean)), lower, upper, alpha=0.5, color="gray")
        # ax.plot(upper, color="black", alpha=0.9, linestyle=":")
        # ax.plot(lower, color="black", alpha=0.9, linestyle=":")
        ax.plot(moving_average(mean,n= 10), alpha=0.9, label=replay_type)
        ax.legend()
        ax.set_title(metric)

plt.show()
