import os
import random
from datasets import DatasetDict, Dataset
from shutil import copy2


def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=2025):
    random.seed(seed)
    classes = os.listdir(dataset_dir)

    train_data = []
    val_data = []
    test_data = []

    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        # データをリストに追加
        train_data.extend(
            [{"image": os.path.join(cls_path, img), "label": cls} for img in train_images])
        val_data.extend(
            [{"image": os.path.join(cls_path, img), "label": cls} for img in val_images])
        test_data.extend(
            [{"image": os.path.join(cls_path, img), "label": cls} for img in test_images])

    # DatasetDict形式で保存
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "val": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    split_dataset("data/raw/train", "data/processed")
