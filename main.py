from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image


# Функция для создания маски из аннотации
def create_masks(coco, image_id):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    masks = []
    for ann in anns:
        mask = coco.annToMask(ann)  # Получаем маску из аннотации
        masks.append(mask)
    return sum(masks)


# Функция для создания масок из аннотаций для всех изображений
def create_masks_for_all_images(coco, output_folder):
    # Создаем папку, если ее нет
    os.makedirs(output_folder, exist_ok=True)
    image_ids = coco.getImgIds()
    for image_id in image_ids:
        mask = create_masks(coco, image_id)
        # Преобразуем каждую маску в формат PIL Image и сохраняем
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil.save(os.path.join(output_folder, f"mask_image_{image_id}.png"))
