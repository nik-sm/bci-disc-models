import pickle as pkl
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bci_disc_models.models.sklearn_models import get_sklearn_model


class LogRatioPosteriors(TransformerMixin):
    """Given a binary classifier, outputs log of ratio of posterior probs:
    log(p(y=1 | x) / p(y=0 | x))."""

    def __init__(self, clf_name):
        self.clf = get_sklearn_model(clf_name)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        log_probs = self.clf.predict_log_proba(X)
        return log_probs[:, 1] - log_probs[:, 0]


class ChannelScaler(TransformerMixin):
    def __init__(self):
        self.model = StandardScaler()

    def fit(self, X, y):
        trials, channels, time = X.shape
        X = X.swapaxes(1, 2).reshape(trials * time, channels)
        self.model.fit(X)
        return self

    def transform(self, X):
        trials, channels, time = X.shape
        X = X.swapaxes(1, 2).reshape(trials * time, channels)
        X = self.model.transform(X)
        return X.reshape(trials, time, channels).swapaxes(1, 2)


class Flattener(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        trials, channels, time = X.shape
        return X.reshape(trials, channels * time)


class KDE(TransformerMixin):
    def __init__(self):
        self.models = [KernelDensity(kernel="gaussian", bandwidth=1.0) for _ in range(2)]

    def fit(self, X, y):
        self.n_class = len(np.unique(y))
        for idx, value in enumerate(np.unique(y)):
            subset = X[y == value].squeeze()
            # Add dim so KernelDensity knows we have N samples with 1 feature
            self.models[idx].fit(subset[..., None])
        return self

    def transform(self, X):
        scores = []
        for idx in range(self.n_class):
            # Add dim so KernelDensity knows we have N samples with 1 feature
            scores.append(self.models[idx].score_samples(X.squeeze()[..., None]))
        return np.array(scores).T  # Return scores with shape (trials, 2)


class GenerativeBaseline:
    def __init__(self, pca_n_components, prior_type="uniform", clf_name: str = "logr"):
        logger.debug("Initializing GenerativeBaseline")
        self.pca_n_components = pca_n_components
        self.prior_type = prior_type
        self.clf_name = clf_name
        self.pipeline = make_pipeline(
            ChannelScaler(),
            Flattener(),
            PCA(n_components=self.pca_n_components),
            LogRatioPosteriors(self.clf_name),
            KDE(),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): shape (trials, channels, time)
            y (np.ndarray): shape (trials,)
        """
        logger.debug(f"Fitting GenerativeBaseline: {X.shape=}, {y.shape=}")

        if self.prior_type == "uniform":
            self.log_prior_class_1 = self.log_prior_class_0 = np.log(0.5)
        elif self.prior_type == "empirical":
            prior_class_1 = np.sum(y == 1) / len(y)
            self.log_prior_class_1 = np.log(prior_class_1)
            self.log_prior_class_0 = np.log(1 - prior_class_1)
        else:
            raise ValueError()

        self.pipeline.fit(X, y)
        return self

    def save(self, folder: Path):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        name = f"{type(self).__name__}.{self.clf_name}.{self.prior_type}.{datetime.now().isoformat('_','seconds')}.pkl"
        path = folder / name
        logger.debug(f"Saving GenerativeBaseline to {path}...")
        with open(path, "wb") as f:
            params = [self.log_prior_class_0, self.log_prior_class_1, self.pipeline]
            pkl.dump(params, f)
        logger.debug("done saving.")
        return path

    def load(self, folder: Path):
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory.")
        matches = list(folder.glob(f"{type(self).__name__}.{self.clf_name}.{self.prior_type}.*.pkl"))
        if not matches:
            raise FileNotFoundError(f"No models found in {folder}.")
        if len(matches) > 1:
            raise ValueError(f"Multiple models found in {folder}.")
        path = matches[0]
        logger.debug(f"Loading GenerativeBaseline from {path}...")
        with open(path, "rb") as f:
            params = pkl.load(f)
            self.log_prior_class_0, self.log_prior_class_1, self.pipeline = params
        logger.debug("done loading.")
        return self

    def predict_log_proba(self, X):
        # p(l=1 | e) = p(e | l=1) p(l=1) / p(e)
        # log(p(l=1 | e)) = log(p(e | l=1)) + log(p(l=1)) - log(p(e))
        logger.debug("self.pipeline.transform...")
        log_scores = self.pipeline.transform(X)
        logger.debug("done")
        log_scores_class_0 = log_scores[:, 0]
        log_scores_class_1 = log_scores[:, 1]
        log_post_0 = log_scores_class_0 + self.log_prior_class_0
        log_post_1 = log_scores_class_1 + self.log_prior_class_1
        denom = np.logaddexp(log_post_0, log_post_1)
        log_post_0 -= denom
        log_post_1 -= denom
        log_posterior = np.stack([log_post_0, log_post_1], axis=-1)
        return log_posterior

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=-1)

    def predict_log_likelihoods(self, data, queried_letter_indices, alphabet_len):
        # Evaluate likelihood probabilities for p(e|l=1) and p(e|l=0)
        log_score = self.pipeline.transform(data)
        # log_score includes the positive and negative class log-likelihoods for each trial seen
        # shape (trials_per_sequence, 2)

        # Evaluate likelihood ratios (positive class divided by negative class)
        subset_log_likelihoods = log_score[:, 1] - log_score[:, 0]

        # Place these log likelihood terms at correct positions to update the full alphabet
        log_likelihoods = np.zeros(alphabet_len)  # Default to adding log prob 0
        for i, value in zip(queried_letter_indices, subset_log_likelihoods):
            log_likelihoods[i] = value

        return log_likelihoods
