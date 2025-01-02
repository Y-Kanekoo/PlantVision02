import os
import random
from shutil import copy2


def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=2025):
    random.seed(seed)
    classes = os.listdir(dataset_dir)

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

        for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                copy2(os.path.join(cls_path, img), split_dir)


if __name__ == "__main__":
    split_dataset("data/raw/train", "data/processed")
