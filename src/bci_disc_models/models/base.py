from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def flatten(x):
    return x.reshape(x.shape[0], np.prod(x.shape[1:]))


class BaseDiscriminativeModel(ABC):
    @abstractmethod
    def save(self, folder: Path):
        ...

    @abstractmethod
    def load(self, folder: Path):
        ...

    @abstractmethod
    def fit(self, data, labels, tiny: bool = False):
        ...

    @abstractmethod
    def predict_proba(self, x):
        ...

    @abstractmethod
    def predict_log_proba(self, x):
        ...

    def predict(self, x):
        return self.predict_proba(x).argmax(-1)

    def predict_log_likelihoods(self, data: np.ndarray, queried_letter_indices: np.ndarray, alp_len: int) -> np.ndarray:
        r"""
        Let:
            D be the unknown target letter
            tau be previously typed letters
            r be the queried letter

        Suppose we query "a". The update for this shown letter is:

            p(D=a | e tau r=a) ∝ p(e | D=a r=a tau) p(D=a | tau)
                               = p(e | label=+) p(D=a | tau)
                               = p(label=+ | e) p(e) / p(label=+)   p(D=a | tau)
                               ∝ p(label=+ | e) / p(label=+) p(D=a | tau)

        The update for all other unseen letters is as follows. Consider "b" for example:

            p(D=b | e tau r=a) ∝ p(e | D=b r=a tau) p(D=b | tau)
                               = p(e | label=-) p(D=b | tau)
                               = p(label=- | e) p(e) / p(label=-)   p(D=b | tau)
                               ∝ p(label=- | e) / p(label=-) p(D=b | tau)

        Args:
            data (np.ndarray): shape (items, channels, time)
            queried_letter_indices (np.ndarray): shape (trials_per_sequence, 1)
            alp_len (int): alphabet length

        Returns:
            additive update for log prob for each letter in alphabet, shape (alp_len,)
        """
        assert len(data) == len(queried_letter_indices)
        model_log_probs = self.predict_log_proba(data)

        # Make an update for each piece of evidence, and add them up
        log_update = np.zeros(alp_len)
        for log_probs, letter_idx in zip(model_log_probs, queried_letter_indices):
            # Compute label likelihood term
            # At the position of the shown letter, use the classifier's output for positive class.
            # At all other positions, use the classifier's output for negative class.
            log_p_label_given_e = log_probs[0] * np.ones(alp_len)
            log_p_label_given_e[letter_idx] = log_probs[1]

            log_p_label = (alp_len - 1) / alp_len * np.ones(alp_len)
            log_p_label[letter_idx] = 1 / alp_len
            log_p_label = np.log(log_p_label)

            log_update = log_update + log_p_label_given_e - log_p_label
        return log_update
