from typing import Literal

import torch
from torch import Tensor, einsum, nn

from MeanIoU import EPSILON, compute_dice_per_channel

NormalizationOptionT = Literal["sigmoid", "softmax", "none"]

class DiceLoss(nn.Module):
    def __init__(
        self,
        normalization: NormalizationOptionT = "sigmoid",
        weights: Tensor | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        if normalization == "sigmoid":
            self.normalization: nn.Module = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        elif normalization == "none":
            self.normalization = nn.Identity()
        else:
            raise ValueError

        self.register_buffer("weights", weights)

        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        probs = self.normalization(inputs)
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )
        return 1.0 - torch.mean(per_channel_dice)
