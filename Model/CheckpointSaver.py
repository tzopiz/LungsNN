import os
import torch

from torch import nn
from pathlib import Path
from dataclasses import dataclass
from os.path import join as pjoin
from shutil import copyfile, rmtree
from accelerate import Accelerator


@dataclass
class Checkpoint:
    metric_val: float  # Значение метрики
    epoch: int  # Номер эпохи
    save_path: Path  # Путь для сохранения


class CheckpointSaver:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        metric_name: str,
        save_dir: str,
        rm_save_dir: bool = False,
        max_history: int = 1,
        should_minimize: bool = True
    ) -> None:
        """
        Args:
            accelerator: Ускоритель (например, GPU)
            model: Модель PyTorch
            metric_name: Имя метрики для логирования
            save_dir: Директория для сохранения чекпоинтов
            max_history: Количество хранимых чекпоинтов
            should_minimize: Если True, метрика должна минимизироваться, иначе - максимизироваться
        """
        # Инициализация CheckpointSaver
        self._accelerator = accelerator
        self._model = model
        self.metric_name = metric_name
        self.save_dir = Path(save_dir)
        self.max_history = max_history
        self.should_minimize = should_minimize

        self._storage: list[Checkpoint] = []

        # Если нужно, удаляем существующую директорию с чекпоинтами
        if os.path.exists(save_dir) and rm_save_dir:
            rmtree(save_dir)

        # Создаем директорию для сохранения чекпоинтов
        os.makedirs(save_dir, exist_ok=True)

    def save(self, metric_val: float, epoch: int) -> None:
        """
        Сохраняет чекпоинт.

        Args:
            metric_val: Значение метрики
            epoch: Номер эпохи
        """
        # Имя файла для сохранения чекпоинта
        save_name_prefix = f"model_e{epoch:03d}_checkpoint"
        # Сохранение чекпоинта
        save_path = self._save_checkpoint(
            model=self._model, epoch=epoch, save_name_prefix=save_name_prefix
        )
        # Добавление чекпоинта в хранилище
        self._storage.append(
            Checkpoint(metric_val=metric_val, epoch=epoch, save_path=save_path)
        )
        # Сортировка хранилища по значению метрики
        self._storage = sorted(
            self._storage, key=lambda x: x.metric_val, reverse=not self.should_minimize
        )
        # Если количество чекпоинтов превышает максимальное значение, удаляем худший чекпоинт
        if len(self._storage) > self.max_history:
            worst_item = self._storage.pop()
            os.remove(worst_item.save_path)

        # Копирование лучшего чекпоинта в отдельный файл
        copyfile(
            src=self._storage[0].save_path,
            dst=self.save_dir / "model_checkpoint_best.pt",
        )
        # Вывод информации о лучшем чекпоинте
        print(
            f"Best epoch {self.metric_name} value is {self._storage[0].metric_val:.4f} "
            f"on {self._storage[0].epoch} epoch"
        )

    def _save_checkpoint(
        self, model: nn.Module, epoch: int, save_name_prefix: str
    ) -> Path:
        save_path = pjoin(self.save_dir, f"{save_name_prefix}.pt")
        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)
        self._accelerator.save(
            obj={"epoch": epoch, "model_state_dict": unwrapped_model.state_dict()},
            f=save_path,
        )
        return Path(save_path)


def load_checkpoint(model: nn.Module, load_path: str) -> nn.Module:
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
