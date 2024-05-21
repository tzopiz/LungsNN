import torch
from torch import Tensor, nn
from MeanIoU import EPSILON, compute_dice_per_channel

class DiceLoss(nn.Module):
    def __init__(
        self,
        weights: Tensor | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        probs = nn.Sigmoid()(inputs)
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )
        return 1.0 - torch.mean(per_channel_dice)
