import numpy as np

from models.replays.ParentReplay import ParentReplay


class RecentReplay(ParentReplay):

    def __init__(self, capacity: int = 10000, **kwargs):
        super(RecentReplay, self).__init__(**kwargs)
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(transition)

    def _get_cprob(self):
        idx = np.arange(len(self.memory))
        prob = idx/idx.sum()
        return prob.cumsum()

    def sample(self, batch_size):
        batch = []
        rand_idx = np.random.uniform(size = batch_size)
        cprob = self._get_cprob()
        for idx in rand_idx:
            batch.append(self.memory[(cprob < idx).sum()])
        return batch

    def __len__(self):
        return len(self.memory)