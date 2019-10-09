from typing import Any

from torch import Tensor

from old_shit.models import GeneralModel


class ParentLoss(GeneralModel):

    def __init__(self, device, **kwargs):
        super(ParentLoss, self).__init__(device, **kwargs)

    # todo: add function template for all losses and then make classes that implement these methods, should be neural networks

    def forward(self, *input: Any, **kwargs: Any) -> Tensor:
        pass  # todo: implement in childclasses (or here if shared functionality)
