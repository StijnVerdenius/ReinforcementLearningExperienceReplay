import numpy as np

from old_shit.models.replays.ParentReplay import ParentReplay


class RandomTrajectoryReplay(ParentReplay):

    def __init__(self, capacity: int = 10000, **kwargs):
        super(RandomTrajectoryReplay, self).__init__(**kwargs)
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(transition)

    def sample(self, batch_size):
        rand = np.random.randint(0, len(self.memory) - batch_size + 1)
        return self.memory[rand:rand + batch_size]

    def __len__(self):
        return len(self.memory)
