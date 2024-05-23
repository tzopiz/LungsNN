import torch

from torch import Tensor, nn
from Model.MeanIoU import EPSILON

class DiceLoss(nn.Module):
    """
    Класс DiceLoss реализует функцию потерь на основе Dice коэффициента.

    Attributes:
        weights (Tensor | None): Веса для каждого канала (по умолчанию None).
        epsilon (float): Малое значение для предотвращения деления на ноль.
    """
    def __init__(self, weights: Tensor | None = None, epsilon: float = EPSILON) -> None:
        """
        Инициализирует DiceLoss с заданными весами и epsilon.

        Args:
            weights (Tensor | None): Веса для каждого канала (по умолчанию None).
            epsilon (float): Малое значение для предотвращения деления на ноль.
        """
        super().__init__()
        self.register_buffer("weights", weights)
        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Вычисляет Dice Loss между входными предсказаниями и целевыми значениями.

        Args:
            inputs (Tensor): Входные предсказания.
            targets (Tensor): Целевые значения.

        Returns:
            Tensor: Значение Dice Loss.
        """
        # Применяем сигмоиду к входным данным, чтобы получить вероятности
        probs = nn.Sigmoid()(inputs)

        # Вычисляем Dice коэффициент для каждого канала
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )

        # Возвращаем среднее значение Dice Loss по всем каналам
        return 1.0 - torch.mean(per_channel_dice)


def compute_dice_per_channel(
        probs: Tensor,
        targets: Tensor,
        epsilon: float = EPSILON,
        weights: Tensor | None = None
) -> Tensor:
    """
    Вычисляет Dice коэффициент для каждого канала.

    Parameters:
    probs (Tensor): Вероятности предсказаний, размерностью `N x C x Spatial`.
    targets (Tensor): Целевые значения, размерностью `N x C x Spatial`.
    epsilon (float): Малое значение для предотвращения деления на ноль.
    weights (Tensor | None): Веса для каждого канала.

    Returns:
    Tensor: Значения Dice коэффициентов для каждого канала.
    """
    assert probs.size() == targets.size()

    # Транспонируем и выравниваем тензоры
    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    # Вычисляем числитель формулы Dice
    numerator = (probs * targets).sum(-1)
    if weights is not None:
        numerator = weights * numerator

    # Вычисляем знаменатель формулы Dice
    denominator = (probs + targets).sum(-1)

    # Вычисляем и возвращаем среднее значение Dice по всем каналам
    return torch.mean(2 * (numerator / denominator.clamp(min=epsilon)), dim=1)
