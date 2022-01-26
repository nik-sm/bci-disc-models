from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import TensorDataset

from bci_disc_models.models.neural_net.dataloaders import Datamodule
from bci_disc_models.models.neural_net.trainer import Trainer
from bci_disc_models.utils import PROJECT_ROOT

from .base import BaseDiscriminativeModel
from .neural_net import get_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NullClassIndex(Enum):
    """When giving a prediction for "item not seen", models must either place
    the value at the first position (0) or last position (-1)."""

    BEGIN = 0
    END = -1


class BaseNeuralNet(BaseDiscriminativeModel):
    def __init__(
        self,
        n_classes: int,
        input_shape: Tuple[int],
        epochs: int,
        lr: float,
        arch: str,
        prior_p_target_in_query: float,
        null_class_index: NullClassIndex,
        results_dir: Path = None,
        device=DEVICE,
    ):
        """
        Args:
            n_classes (int):
            input_shape (Tuple[int]): Shape of one data item
            epochs (int):
            arch (str, optional): model architecture
            results_dir (Path, optional): path to store model logs and checkpoints
            null_class_index (NullClassIndex): Index of model's output for "null" class.
            prior_p_target_in_query (float, optional): Prior probability of target appearing in a query sequence.
            device (torch.device):
        """
        self.input_shape = input_shape
        self.arch = arch
        self.model = get_model(arch=self.arch, n_classes=n_classes, input_shape=self.input_shape)
        self.n_classes = n_classes
        self.device = device
        self.results_dir = results_dir or PROJECT_ROOT / "results" / (
            self.arch + "_" + datetime.now().isoformat("_", "seconds")
        )
        self.epochs = epochs
        self.lr = lr
        self.null_class_index = null_class_index.value
        self.prior_p_target_in_query = prior_p_target_in_query
        logger.debug(f"N trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def fit(self, x, y):
        logger.debug(f"{x.shape=}, {y.shape=}")
        self.datamodule = Datamodule(x=x, y=y, n_classes=self.n_classes, val_frac=0.1)
        self.trainer = Trainer(
            self.model, self.datamodule, lr=self.lr, results_dir=self.results_dir, device=self.device
        )
        trainer_metrics = self.trainer(epochs=self.epochs)
        logger.debug(f"Trainer metrics: {trainer_metrics}")
        return self

    def save(self, folder: Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{type(self).__name__}.{self.arch}.{datetime.now().isoformat('_','seconds')}.pt"
        path = folder / name
        logger.info(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)
        return path

    def load(self, folder: Path):
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory")
        matches = list(folder.glob(f"{type(self).__name__}.{self.arch}.*.pt"))
        if not matches:
            raise FileNotFoundError(f"No model found in {folder}")
        if len(matches) > 1:
            raise ValueError(f"Multiple models found in {folder}")
        path = matches[0]
        logger.info(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def predict_log_proba(self, x: np.ndarray):
        self.model.eval()
        x = x.astype(np.float32)
        loader = Datamodule.get_loader(TensorDataset(torch.from_numpy(x)), shuffle=False)
        all_log_probs = []
        for (x,) in loader:
            x = x.to(self.device)
            log_probs = self.model(x).cpu().numpy()
            all_log_probs.append(log_probs)
        return np.concatenate(all_log_probs)

    @torch.no_grad()
    def predict_proba(self, data: np.ndarray):
        return np.exp(self.predict_log_proba(data))

    @torch.no_grad()
    def predict_log_likelihoods(
        self, data: np.ndarray, queried_letter_indices: np.ndarray, alphabet_len: int
    ) -> np.ndarray:
        self.model.eval()
        return super().predict_log_likelihoods(data, queried_letter_indices, alphabet_len)


class SequenceNeuralNet(BaseNeuralNet):
    def __init__(self, null_class_index=NullClassIndex.END, **kwargs):
        super().__init__(null_class_index=null_class_index, **kwargs)


class TrialNeuralNet(BaseNeuralNet):
    def __init__(self, null_class_index=NullClassIndex.BEGIN, **kwargs):
        # For a binary classifier, "not seen" is simply the negative class
        super().__init__(null_class_index=null_class_index, **kwargs)
