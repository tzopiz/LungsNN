from typing import Literal

import torch
from torch import Tensor, einsum, nn

from metric import EPSILON, compute_dice_per_channel

NormalizationOptionT = Literal["sigmoid", "softmax", "none"]


class BCEDiceBoundaryLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        is_3d: bool = True,
        normalization: NormalizationOptionT = "sigmoid",
        weights: Tensor | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(
            normalization=normalization, epsilon=epsilon, weights=weights
        )
        self.boundary = BoundaryLoss(is_3d=is_3d)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return (
            self.alpha * self.bce(inputs, targets)
            + self.beta * self.dice(inputs, targets)
            + self.gamma * self.boundary(torch.sigmoid(inputs), targets)
        )


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
        print(f'inputs: {inputs}, targets: {targets.unique()}')
        probs = self.normalization(inputs)
        print(f'probs: {probs}')
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )
        return 1.0 - torch.mean(per_channel_dice)


class BoundaryLoss(nn.Module):
    def __init__(self, is_3d: bool = True) -> None:
        super().__init__()

        self.pattern = "bcdwh,bcdwh->bcdwh" if is_3d else "bcwh,bcwh->bcwh"

    def forward(self, probs: Tensor, targets: torch.Tensor) -> torch.Tensor:
        return einsum(self.pattern, probs, targets).mean()
