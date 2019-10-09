from utils.data_manager import DataManager
'''
File used to load data and analyse it.
To do so just follow the example below.
Replace id with the name of the corresponding
folder and your data will be in the data object
'''
# Example
loader = DataManager("gitignored/results/")
id = "2019-10-09_17.40.17CartPole-v0"
filename = id + "/progress/progress_list"
data = loader.load_python_obj(filename)
for d in data[0:10]:
    print(d)