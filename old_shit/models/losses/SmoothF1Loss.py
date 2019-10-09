import torch.nn.functional as F
from torch import Tensor

from old_shit.models import ParentLoss


class SmoothF1Loss(ParentLoss):

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)

    def forward(self, q_val, target) -> Tensor:
        return F.smooth_l1_loss(q_val, target)
