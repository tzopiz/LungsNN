from __future__ import annotations
from typing import Any, Callable

import os
import torch

from torch import optim, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from pathlib import Path
from dataclasses import dataclass
from os.path import join as pjoin
from shutil import copyfile, rmtree
from accelerate import Accelerator




def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,
    accelerator: Accelerator,
    epoch_num: int,
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,
    save_on_val: bool = True,
    show_every_x_batch: int = 30
) -> None:
    """
    Эта функция управляет процессом обучения модели.
    Она запускает цикл обучения на протяжении заданного количества эпох,
    обновляя веса модели на каждой эпохе.

    Args:
        model (nn.Module): Модель для обучения.
        optimizer (optim.Optimizer): Оптимизатор для обучения.
        train_dataloader (DataLoader): Даталоадер для обучающих данных.
        val_dataloader (DataLoader | None): Даталоадер для валидационных данных (если есть).
        loss_function (Callable[[Any, Any], torch.Tensor]): Функция потерь.
        metric_function (Callable[[Any, Any], torch.Tensor]): Функция для вычисления метрики.
        lr_scheduler (LRScheduler): Планировщик изменения скорости обучения.
        accelerator (Accelerator): Инструмент для распределенного обучения.
        epoch_num (int): Количество эпох обучения.
        checkpointer (CheckpointSaver): Объект для сохранения контрольных точек.
        tb_logger (SummaryWriter | None): Логгер для TensorBoard.
        save_on_val (bool): Флаг для сохранения контрольной точки на этапе валидации.
        show_every_x_batch (int): Интервал отображения информации о текущем состоянии обучения.
    """
    global_train_step, global_val_step = 0, 0
    for epoch in tqdm(range(epoch_num)):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{epoch_num}")

        # Шаг тренировки
        global_train_step = train_step(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            loss_function=loss_function,
            metric_function=metric_function,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_train_step=global_train_step,
            save_on_val=save_on_val,
            show_every_x_batch=show_every_x_batch,
        )

        # Шаг валидации
        if val_dataloader is not None:
            global_val_step = validation_step(
                epoch=epoch,
                model=model,
                val_dataloader=val_dataloader,
                loss_function=loss_function,
                metric_function=metric_function,
                checkpointer=checkpointer,
                tb_logger=tb_logger,
                global_val_step=global_val_step,
                save_on_val=save_on_val,
            )


def train_step(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,
    accelerator: Accelerator,
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,
    global_train_step: int,
    save_on_val: bool = True,
    show_every_x_batch: int = 30
) -> int:
    """
    Эта функция представляет собой один шаг обучения модели.
    На каждом шаге она берет один батч данных из обучающего DataLoader,
    передает его через модель, вычисляет потери и метрики,
    а затем обновляет веса модели с помощью оптимизатора.
    Она также отвечает за логирование метрик и потерь,
    а также за сохранение чекпоинтов модели.

    Args:
        epoch (int): Текущая эпоха.
        model (nn.Module): Модель для обучения.
        optimizer (optim.Optimizer): Оптимизатор для обучения.
        train_dataloader (DataLoader): Даталоадер для обучающих данных.
        loss_function (Callable[[Any, Any], torch.Tensor]): Функция потерь.
        metric_function (Callable[[Any, Any], torch.Tensor]): Функция для вычисления метрики.
        lr_scheduler (LRScheduler): Планировщик изменения скорости обучения.
        accelerator (Accelerator): Инструмент для распределенного обучения.
        checkpointer (CheckpointSaver): Объект для сохранения контрольных точек.
        tb_logger (SummaryWriter | None): Логгер для TensorBoard.
        global_train_step (int): Глобальный шаг тренировки.
        save_on_val (bool): Флаг для сохранения контрольной точки на этапе валидации.
        show_every_x_batch (int): Интервал отображения информации о текущем состоянии тренировки.

    Returns:
        int: Обновленное значение глобального шага тренировки.
    """
    model.train()

    batch_idx = 0
    total_train_loss, total_train_metric = 0.0, 0.0
    for inputs, targets in tqdm(train_dataloader, desc="Training"):
        batch_idx += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        metric = metric_function(outputs, targets)
        total_train_loss += loss.item()
        total_train_metric += metric.item()
        accelerator.backward(loss)
        optimizer.step()

        if not batch_idx % show_every_x_batch:
            print(f"Batch train loss: {loss.item():.5f}")
            print(f"Batch train metric: {metric.item():.5f}")

        if tb_logger is not None:
            tb_logger.add_scalar("loss_train_batch", loss.item(), global_train_step)
            tb_logger.add_scalar("metric_train_batch", metric.item(), global_train_step)
            global_train_step += 1

    lr_scheduler.step()
    total_train_loss /= len(train_dataloader)
    total_train_metric /= len(train_dataloader)
    print(f"Epoch train loss: {total_train_loss:.5f}")
    print(f"Epoch train metric: {total_train_metric:.5f}")
    if tb_logger is not None:
        tb_logger.add_scalar("loss_train_epoch", total_train_loss, epoch)
        tb_logger.add_scalar("metric_train_epoch", total_train_metric, epoch)

    if not save_on_val:
        checkpointer.save(metric_val=total_train_metric, epoch=epoch)

    return global_train_step


def validation_step(
    epoch: int,
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,
    global_val_step: int,
    save_on_val: bool = True
) -> int:
    """
    Эта функция представляет собой один шаг валидации модели.
    На каждом шаге валидации она берет один батч данных из валидационного DataLoader,
    передает его через модель, вычисляет потери и метрики без обновления весов модели.
    Она также отвечает за логирование метрик и потерь во время валидации.

    Args:
        epoch (int): Текущая эпоха.
        model (nn.Module): Модель для валидации.
        val_dataloader (DataLoader): Даталоадер для валидационных данных.
        loss_function (Callable[[Any, Any], torch.Tensor]): Функция потерь.
        metric_function (Callable[[Any, Any], torch.Tensor]): Функция для вычисления метрики.
        checkpointer (CheckpointSaver): Объект для сохранения контрольных точек.
        tb_logger (SummaryWriter | None): Логгер для TensorBoard.
        global_val_step (int): Глобальный шаг валидации.
        save_on_val (bool): Флаг для сохранения контрольной точки на этапе валидации.

    Returns:
        int: Обновленное значение глобального шага валидации.
    """
    model.eval()

    total_val_loss, total_val_metric = 0.0, 0.0
    for inputs, targets in tqdm(val_dataloader, desc="Validation"):
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            metric = metric_function(outputs, targets)
            total_val_loss += loss.item()
            total_val_metric += metric.item()

        if tb_logger is not None:
            tb_logger.add_scalar("loss_val_batch", loss.item(), global_val_step)
            tb_logger.add_scalar("metric_val_batch", metric.item(), global_val_step)
            global_val_step += 1

    total_val_loss /= len(val_dataloader)
    total_val_metric /= len(val_dataloader)
    print(f"Epoch validation loss: {total_val_loss:.5f}")
    print(f"Epoch validation metric: {total_val_metric:.5f}")
    if tb_logger is not None:
        tb_logger.add_scalar("loss_val_epoch", total_val_loss, epoch)
        tb_logger.add_scalar("metric_val_epoch", total_val_metric, epoch)

    if save_on_val:
        checkpointer.save(metric_val=total_val_metric, epoch=epoch)

    return global_val_step



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
