import numpy as np

from bci_disc_models.models import BaseDiscriminativeModel
from bci_disc_models.models.sklearn_models import ClipExtremes


class ConstantModel(BaseDiscriminativeModel):
    def __init__(self, value: int, prior_p_target_in_query: float):
        self.value = value
        self.prior_p_target_in_query = prior_p_target_in_query

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        res = np.zeros((len(x), 2))
        res[:, self.value] = 1
        return res

    def save(self, folder):
        pass

    def load(self, path):
        pass


class AlwaysClass1(ClipExtremes, ConstantModel):
    def __init__(self, prior_p_target_in_query):
        super().__init__(1, prior_p_target_in_query)


class AlwaysClass0(ClipExtremes, ConstantModel):
    def __init__(self, prior_p_target_in_query):
        super().__init__(0, prior_p_target_in_query)
