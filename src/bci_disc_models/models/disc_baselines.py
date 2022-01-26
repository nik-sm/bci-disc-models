import pickle as pkl
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

from bci_disc_models.models.sklearn_models import get_sklearn_model

from .base import BaseDiscriminativeModel, flatten


class ScikitModelAdaptor(BaseDiscriminativeModel):
    def __init__(self, *, clf_name: str, prior_p_target_in_query: float):
        self.prior_p_target_in_query = prior_p_target_in_query
        self.scaler = StandardScaler()
        self.clf = get_sklearn_model(clf_name)

    def save(self, folder: Path):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        name = f"{type(self.clf).__name__}.{datetime.now().isoformat('_', 'seconds')}.pkl"
        path = folder / name
        logger.debug(f"Saving model to {path}")
        with open(path, "wb") as f:
            pkl.dump([self.scaler, self.clf], f)
        return path

    def load(self, folder: Path):
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory")
        matches = list(folder.glob(f"{type(self.clf).__name__}.*.pkl"))
        if not matches:
            raise FileNotFoundError(f"No model found in {folder}")
        if len(matches) > 1:
            raise ValueError(f"Multiple models found in {folder}")
        path = matches[0]
        logger.info(f"Loading model from {path}")
        with open(path, "rb") as f:
            self.scaler, self.clf = pkl.load(f)
        return self

    def fit(self, x, y):
        logger.debug(f"Before flatten: {x.shape=}, {y.shape=}")
        x = flatten(x)
        logger.debug(f"After flatten: {x.shape=}, {y.shape=}")
        x = self.scaler.fit_transform(x)
        self.clf.fit(x, y)
        return self

    def predict_proba(self, x):
        x = self.scaler.transform(flatten(x))
        return self.clf.predict_proba(x)

    def predict_log_proba(self, x):
        return np.log(self.predict_proba(x))
