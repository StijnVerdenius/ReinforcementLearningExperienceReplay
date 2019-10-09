import random
import numpy as np
from models.replays.ParentReplay import ParentReplay
from utils.SumTree import SumTree

'''
Priority replay class based on the implementation of PRIORITIZED EXPERIENCE REPLAY by deepmind (https://arxiv.org/pdf/1511.05952.pdf)
Motivation: Leaf nodes store the transition priorities and the internal nodes are intermediate sums, with the parent node containing 
the sum over all priorities, p total. This provides a efficient way of calculating the cumulative sum of priorities,
allowing O(log N ) updates and sampling.

'''
class PriorityReplay(ParentReplay):  # stored as ( s, a, r, s_ ) in SumTree
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, device, **kwargs):
        super().__init__(device, **kwargs)
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.tree)

    def _get_priority(self, error):
        return (np.abs(error) + abs(np.random.normal(0.01, 0.001)))

    def push(self, sample, error):
        p = self._get_priority(error)
        self.tree.add(p, sample)


    def sample(self, k):
        # To sample a minibatch of size k
        batch = []
        idxs = []
        segment = self.tree.total() / k
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # the range [0, ptotal] is divided equally into k ranges.
        for i in range(k):
            # Next, a value is uniformly sampled from each range
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Finally the transitions that correspond to each of these sampled values are retrieved from the tree
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights wi were scaled so that maxi wi = 1
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)