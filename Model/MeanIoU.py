import torch

from torch import Tensor, nn

EPSILON = 1e-6


class MeanIoU(nn.Module):
    """
    Класс MeanIoU вычисляет средний показатель пересечения над объединением (IoU)
    для каждого канала.
    """

    def __init__(
            self,
            threshold: float = 0.5,
            epsilon: float = EPSILON,
            onehot_conversion: bool = False,
            binarize: bool = True
    ) -> None:
        """
        Инициализирует MeanIoU с заданным порогом, epsilon, флагами для one-hot конвертации
        и бинаризации.

        Parameters:
        threshold (float): Порог для бинаризации предсказаний.
        epsilon (float): Малое значение для предотвращения деления на ноль.
        onehot_conversion (bool): Флаг для конвертации целевых значений в one-hot формат.
        binarize (bool): Флаг для бинаризации предсказаний.
        """
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon
        self.onehot_conversion = onehot_conversion
        self.binarize = binarize

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Вычисляет Mean IoU между входными предсказаниями и целевыми значениями.

        Args:
            inputs (Tensor): Входные предсказания, размерностью `N x C x Spatial`.
            targets (Tensor): Целевые значения, размерностью `N x C x Spatial` или `N x Spatial`.

        Returns:
            Tensor: Значение Mean IoU.
        """
        classes_num = inputs.shape[1]  # кол-во классов

        # Конвертируем целевые значения в one-hot формат, если это указано
        if self.onehot_conversion:
            targets = convert_to_one_hot(targets, classes_num=classes_num)

        # Убеждаемся, что размеры входных данных и целевых значений совпадают
        assert inputs.dim() == targets.dim()

        # Бинаризуем предсказания, если это указано
        if self.binarize:
            preds = binarize_probs(
                inputs=torch.sigmoid(inputs),  # Применяем сигмоиду к входным данным
                classes_num=classes_num,
                threshold=self.threshold
            )
        else:
            preds = torch.sigmoid(inputs)

        # Вычисляем и возвращаем среднее значение IoU по всем каналам
        return torch.mean(
            compute_miou_per_channel(probs=preds, targets=targets, epsilon=self.epsilon)
        )


def compute_miou_per_channel(
        probs: Tensor,
        targets: Tensor,
        epsilon: float = EPSILON,
        weights: Tensor | None = None
) -> Tensor:
    """
    Вычисляет Mean IoU для каждого канала.

    Args:
        probs (Tensor): Вероятности предсказаний, размерностью `N x C x Spatial`.
        targets (Tensor): Целевые значения, размерностью `N x C x Spatial`.
        epsilon (float): Малое значение для предотвращения деления на ноль.
        weights (Tensor | None): Веса для каждого канала.

    Returns:
        Tensor: Значения Mean IoU для каждого канала.
    """
    assert probs.size() == targets.size()

    # Транспонируем и выравниваем тензоры
    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).byte()

    # Вычисляем числитель формулы IoU
    numerator = (probs & targets).sum(-1).float()
    if weights is not None:
        numerator = weights * numerator

    # Вычисляем знаменатель формулы IoU
    denominator = (probs | targets).sum(-1).float()

    # Вычисляем и возвращаем среднее значение IoU по всем каналам
    return torch.mean(numerator / denominator.clamp(min=epsilon), dim=1)


def convert_to_one_hot(targets: Tensor, classes_num: int) -> Tensor:
    """
    Конвертирует целевые значения в one-hot формат.

    Args:
        targets (Tensor): Целевые значения, размерностью `N x Spatial`.
        classes_num (int): Количество классов.

    Returns:
        Tensor: Целевые значения в формате one-hot, размерностью `N x C x Spatial`.
    """
    assert targets.dim() == 4
    targets = targets.unsqueeze(1)  # Добавляем новый размер
    shape = list(targets.shape)
    shape[1] = classes_num  # Устанавливаем размер для классов
    # Создаем тензор нулей и расставляем единицы в соответствующих позициях
    return torch.zeros(shape).to(targets.device).scatter_(1, targets, 1)


def binarize_probs(inputs: Tensor, classes_num: int, threshold: float = 0.5) -> Tensor:
    """
    Бинаризует вероятности предсказаний по заданному порогу.

    Args:
        inputs (Tensor): Вероятности предсказаний, размерностью `N x C x Spatial`.
        classes_num (int): Количество классов.
        threshold (float): Порог для бинаризации.

    Returns:
        Tensor: Бинаризованные предсказания, размерностью `N x C x Spatial`.
    """
    if classes_num == 1:
        # Для одного класса применяем пороговое значение
        return (inputs > threshold).byte()

    # Для многоклассовой классификации выбираем максимальные значения
    return torch.zeros_like(inputs, dtype=torch.uint8).scatter_(
        1, torch.argmax(inputs, dim=1, keepdim=True), 1
    )
