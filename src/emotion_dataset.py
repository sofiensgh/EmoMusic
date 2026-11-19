"""Load FER-2013 and map its 7 emotions to our 4 moods.

Supports both Kaggle versions:
  * CSV  : fer2013.csv with columns  emotion, pixels, Usage
  * Images: fer2013/train/<emotion>/*.jpg and fer2013/test/<emotion>/*.jpg
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .config import FER_NAMES, FER_TO_MOOD, IMG_SIZE


def _map(images, labels):
    keep = [i for i, y in enumerate(labels) if FER_TO_MOOD[int(y)] is not None]
    x = np.stack([images[i] for i in keep]).astype(np.uint8)
    y = np.array([FER_TO_MOOD[int(labels[i])] for i in keep], dtype=np.int64)
    return x, y


def load_fer_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    imgs = [np.array(p.split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE) for p in df["pixels"]]
    train_idx = df["Usage"] == "Training"
    imgs = np.stack(imgs)
    tr = _map(imgs[train_idx.values], df["emotion"].values[train_idx.values])
    va = _map(imgs[~train_idx.values], df["emotion"].values[~train_idx.values])
    return tr, va


def _load_folder(folder):
    from PIL import Image
    imgs, labels = [], []
    for label, name in enumerate(FER_NAMES):
        d = Path(folder) / name
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                img = Image.open(f).convert("L").resize((IMG_SIZE, IMG_SIZE))
                imgs.append(np.array(img))
                labels.append(label)
    return _map(imgs, labels)


def load_fer(path):
    """Returns ((x_train, y_train), (x_val, y_val)) with uint8 images N x 48 x 48."""
    path = Path(path)
    if path.is_file() and path.suffix == ".csv":
        return load_fer_csv(path)
    if (path / "train").exists():
        return _load_folder(path / "train"), _load_folder(path / "test")
    raise FileNotFoundError(f"Could not find FER-2013 at {path} (expected a .csv or a folder with train/ and test/)")


def train_transform():
    # Same ideas as 04a_asl_augmentation: small rotations, zoom, flips, brightness
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
    ])


def preprocess(gray_uint8):
    """uint8 HxW (48x48) -> normalised float tensor 1x48x48 in [-1, 1]."""
    t = torch.as_tensor(gray_uint8, dtype=torch.float32).unsqueeze(0) / 255.0
    return (t - 0.5) / 0.5


class FaceDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x, self.y = x, y
        self.tf = train_transform() if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.as_tensor(self.x[i]).unsqueeze(0)  # uint8 1x48x48
        if self.tf is not None:
            img = self.tf(img)
        img = img.float() / 255.0
        return (img - 0.5) / 0.5, int(self.y[i])
