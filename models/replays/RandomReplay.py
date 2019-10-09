import numpy as np

from models.replays.ParentReplay import ParentReplay


class RandomReplay(ParentReplay):

    def __init__(self, capacity: int = 10000, **kwargs):
        super(RandomReplay, self).__init__(**kwargs)
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(transition)

    def sample(self, batch_size):
        rand_idx = np.random.randint(0, len(self.memory), size = batch_size)
        return np.array(self.memory)[rand_idx].tolist()

    def __len__(self):
        return len(self.memory)
