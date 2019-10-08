from typing import Any

from torch import Tensor

from models.GeneralModel import GeneralModel


class ParentAgent(GeneralModel):

    def __init__(self, device, num_hidden, actions, state_size, **kwargs):
        super(ParentAgent, self).__init__(device, **kwargs)
        self.state_size = state_size
        self.actions = actions
        self.num_hidden = num_hidden

    # todo: add function template for all agents and then make classes that implement these methods, should be neural networks

    def forward(self, *input: Any, **kwargs: Any) -> Tensor:
        pass  # todo: implement in childclasses (or here if shared functionality)
