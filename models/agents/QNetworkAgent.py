from models.agents.ParentAgent import ParentAgent
import torch.nn as nn


class QNetworkAgent(ParentAgent):

    def __init__(self, device="cuda", num_hidden=128, **kwargs):
        super(QNetworkAgent, self).__init__(device, **kwargs)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        return self.l2(nn.functional.relu(self.l1(x)))