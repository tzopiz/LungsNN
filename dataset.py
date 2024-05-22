import os
import numpy as np

from torch.utils.data import Dataset
from torch import Tensor
from PIL import Image

class LungsDataset(Dataset):
    """
    Класс LungsDataset для работы с набором данных легких и их масок.

    Attributes:
    root_dir (str): Корневая папка с данными.
    transforms (callable, optional): Трансформации, применяемые к изображениям и маскам.
    image_paths (list): Список путей к изображениям.
    mask_paths (list): Список путей к маскам.
    """
    def __init__(self, root_dir, transforms=None):
        """
        Инициализирует LungsDataset с указанной корневой папкой и трансформациями.

        Parameters:
        root_dir (str): Корневая папка с данными.
        transforms (callable, optional): Трансформации, применяемые к изображениям и маскам.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = []
        self.mask_paths = []

        # Получаем список файлов в папке Lungs и Masks
        link_dir = os.path.join(root_dir, 'Lungs')
        mask_dir = os.path.join(root_dir, 'Masks')

        # Перебираем файлы и добавляем изображения и соответствующие маски в списки
        for file in os.listdir(link_dir):
            if file.endswith(".png"):  # Предполагаем, что изображения имеют расширения .png
                image_path = os.path.join(link_dir, file)
                mask_file = "mask_" + file
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        """
        Возвращает количество изображений в наборе данных.

        Returns:
        int: Количество изображений.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Возвращает изображение и его маску по индексу.

        Parameters:
        index (int): Индекс изображения и маски.

        Returns:
        tuple[Tensor, Tensor]: Кортеж, содержащий изображение и его маску.
        """
        # Загружаем изображение и маску
        img = Image.open(self.image_paths[index]).convert("RGB")
        target = Image.open(self.mask_paths[index]).convert("L")

        # Преобразуем изображения в numpy массивы
        img = np.array(img, dtype=np.uint8)
        target = np.array(target, dtype=np.uint8)

        # Преобразуем маску в бинарный формат
        mask = np.where(target == 255, 1, target)

        # Убеждаемся, что трансформации не None
        assert self.transforms is not None

        # Применяем трансформации к изображению и маске
        augmented = self.transforms(image=img, mask=mask)

        # Возвращаем трансформированные изображение и маску
        return augmented["image"].float(), augmented["mask"].unsqueeze(0).float()


# Функция для создания маски из аннотации
def _create_mask(coco, image_id):
    """
    Создает маску для изображения по его идентификатору из аннотаций COCO.

    Parameters:
    coco (pycocotools.coco.COCO): Объект COCO с аннотациями.
    image_id (int): Идентификатор изображения.

    Returns:
    numpy.ndarray: Суммарная маска для всех аннотаций, связанных с изображением.
    """
    # Получаем идентификаторы аннотаций для данного изображения
    ann_ids = coco.getAnnIds(imgIds=image_id)
    # Загружаем аннотации по их идентификаторам
    anns = coco.loadAnns(ann_ids)
    masks = []
    # Проходим по каждой аннотации
    for ann in anns:
        # Получаем маску из аннотации
        mask = coco.annToMask(ann)
        masks.append(mask)
    # Возвращаем суммарную маску
    return sum(masks)


# Функция для создания масок из аннотаций для всех изображений
def create_masks_for_all_images(coco, output_folder):
    """
    Создает маски для всех изображений и сохраняет их в указанную папку.

    Parameters:
    coco (pycocotools.coco.COCO): Объект COCO с аннотациями.
    output_folder (str): Путь к папке для сохранения масок.

    Returns:
    None
    """
    # Создаем папку, если ее нет
    os.makedirs(output_folder, exist_ok=True)
    # Получаем идентификаторы всех изображений
    image_ids = coco.getImgIds()
    # Проходим по каждому идентификатору изображения
    for image_id in image_ids:
        # Создаем маску для текущего изображения
        mask = _create_mask(coco, image_id)
        # Преобразуем маску в формат PIL Image и сохраняем
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil.save(os.path.join(output_folder, f"mask_image_{str(image_id).zfill(4)}.png"))
