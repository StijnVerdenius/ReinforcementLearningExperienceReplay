from typing import Any

from torch import Tensor

from old_shit.models import GeneralModel


class ParentAgent(GeneralModel):

    def __init__(self, device, num_hidden, actions, state_size, **kwargs):
        super(ParentAgent, self).__init__(device, **kwargs)
        self.state_size = self._unpack_state(state_size)
        self.actions = self._unpack_actions(actions)
        self.num_hidden = num_hidden

    def _unpack_actions(self, element):
        try:
            return [element.n]
        except:
            return element.shape

    def _unpack_state(self, element):
        try:
            return [len([element.n])]
        except:
            return element.shape

    # todo: add function template for all agents and then make classes that implement these methods, should be neural networks

    def forward(self, *input: Any, **kwargs: Any) -> Tensor:
        pass  # todo: implement in childclasses (or here if shared functionality)
