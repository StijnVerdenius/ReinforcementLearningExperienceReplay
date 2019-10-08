from models.agents.ParentAgent import ParentAgent
import torch.nn as nn


class QNetworkAgent(ParentAgent):

    def __init__(self, device, num_hidden, actions, state_size, **kwargs):
        super(QNetworkAgent, self).__init__(device, num_hidden, actions, state_size, **kwargs)
        self.l1 = nn.Linear(state_size[0], num_hidden).to(device)
        self.l2 = nn.Linear(num_hidden, actions).to(device)

    def forward(self, x):
        x = x.to(self.device)
        return self.l2(nn.functional.relu(self.l1(x)))