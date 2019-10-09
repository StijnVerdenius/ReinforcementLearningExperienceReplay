import torch.nn as nn

from old_shit.models.agents.ParentAgent import ParentAgent


class QNetworkAgent(ParentAgent):

    def __init__(self, device, num_hidden, actions, state_size, **kwargs):
        super(QNetworkAgent, self).__init__(device, num_hidden, actions, state_size, **kwargs)
        self.l1 = nn.Linear(self.state_size[0], self.num_hidden).to(device)
        self.l2 = nn.Linear(self.num_hidden, self.actions[0]).to(device)

    def forward(self, x):
        x = x.to(self.device)
        return self.l2(nn.functional.relu(self.l1(x)))
