import torch
from torch import Tensor, nn

EPSILON = 1e-6


class Dice(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,
        epsilon: float = EPSILON,
        onehot_conversion: bool = False,
        binarize: bool = True,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.epsilon = epsilon
        self.onehot_conversion = onehot_conversion
        self.binarize = binarize

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        inputs is `N x C x Spatial`,
        targets is `N x C x Spatial` or `N x Spatial`.
        """
        classes_num = inputs.shape[1]
        if self.onehot_conversion:
            targets = convert_to_one_hot(targets, classes_num=classes_num)

        assert inputs.dim() == targets.dim()

        if self.binarize:
            preds = binarize_probs(
                inputs=torch.sigmoid(inputs),
                classes_num=classes_num,
                threshold=self.threshold,
            )
        else:
            preds = torch.sigmoid(inputs)

        return torch.mean(
            compute_dice_per_channel(probs=preds, targets=targets, epsilon=self.epsilon)
        )


class MeanIoU(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,
        epsilon: float = EPSILON,
        onehot_conversion: bool = False,
        binarize: bool = True,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.epsilon = epsilon
        self.onehot_conversion = onehot_conversion
        self.binarize = binarize

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        inputs is `N x C x Spatial`,
        targets is `N x C x Spatial` or `N x Spatial`.
        """
        classes_num = inputs.shape[1]
        if self.onehot_conversion:
            targets = convert_to_one_hot(targets, classes_num=classes_num)

        assert inputs.dim() == targets.dim()

        if self.binarize:
            preds = binarize_probs(
                inputs=torch.sigmoid(inputs),
                classes_num=classes_num,
                threshold=self.threshold,
            )
        else:
            preds = torch.sigmoid(inputs)

        return torch.mean(
            compute_miou_per_channel(probs=preds, targets=targets, epsilon=self.epsilon)
        )


def compute_dice_per_channel(
    probs: Tensor,
    targets: Tensor,
    epsilon: float = EPSILON,
    weights: Tensor | None = None,
) -> Tensor:
    """input and target are `N x C x Spatial`, weights are `C x 1`."""

    assert probs.size() == targets.size()

    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    numerator = (probs * targets).sum(-1)
    if weights is not None:
        numerator = weights * numerator

    denominator = (probs + targets).sum(-1)

    return torch.mean(2 * (numerator / denominator.clamp(min=epsilon)), dim=1)


def compute_miou_per_channel(
    probs: Tensor,
    targets: Tensor,
    epsilon: float = EPSILON,
    weights: Tensor | None = None,
) -> Tensor:
    """input and target are `N x C x Spatial`, weights is `C x 1`."""

    assert probs.size() == targets.size()

    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).byte()

    numerator = (probs & targets).sum(-1).float()
    if weights is not None:
        numerator = weights * numerator

    denominator = (probs | targets).sum(-1).float()

    return torch.mean(numerator / denominator.clamp(min=epsilon), dim=1)


def convert_to_one_hot(targets: Tensor, classes_num: int) -> Tensor:
    # from `N x Spatial` into `N x C x Spatial`

    assert targets.dim() == 4
    targets = targets.unsqueeze(1)
    shape = list(targets.shape)
    shape[1] = classes_num
    return torch.zeros(shape).to(targets.device).scatter_(1, targets, 1)


def binarize_probs(inputs: Tensor, classes_num: int, threshold: float = 0.5) -> Tensor:
    # input is `N x C x Spatial`

    if classes_num == 1:
        return (inputs > threshold).byte()

    return torch.zeros_like(inputs, dtype=torch.uint8).scatter_(
        1, torch.argmax(inputs, dim=0), 1
    )
