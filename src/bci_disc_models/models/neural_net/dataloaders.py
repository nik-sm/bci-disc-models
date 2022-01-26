import random

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bci_disc_models.utils import seed_everything


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Datamodule:
    def __init__(self, x: np.ndarray, y: np.ndarray, n_classes: int, val_frac: float = 0.1):
        seed_everything(0)
        x, y = x.astype(np.float32), y.astype(np.int64)
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=val_frac)
        self.train_set = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        self.val_set = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

        # Set class weights as inverse of class frequency in training set, for weighted cross-entropy
        class_counts = torch.from_numpy(train_y.flatten()).bincount()
        self.class_weights = class_counts.sum() / class_counts
        self.class_weights /= self.class_weights.sum()
        logger.info(f"Distinct classes: {np.unique(train_y)}")
        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Class weights: {self.class_weights}")
        self.n_classes = n_classes

    @staticmethod
    def get_loader(dataset: Dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=4096,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=8,
            worker_init_fn=seed_worker,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None
        return self.get_loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None
        return self.get_loader(self.val_set, shuffle=False)
