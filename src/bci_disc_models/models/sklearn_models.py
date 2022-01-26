import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _qda
from sklearn.ensemble import RandomForestClassifier as _rf
from sklearn.linear_model import LogisticRegression as _logr
from sklearn.neural_network import MLPClassifier as _mlp
from sklearn.svm import SVC as _svc


class ClipExtremes:
    """Since we construct a log ratio of probabilities, we must be careful
    in case a model returns probability close to 0 for one class, which
    would result in log(0) = -inf.

    NOTE - in sklearn implementations, clf.predict_log_proba() is always just
    np.log(clf.predict_proba()). Thus we can focus on clipping clf.predict_proba()"""

    def predict_proba(self, x):
        # NOTE - with careful multiple inheritance, super() here
        # takes us to the actual method in sklearn base class.
        return np.clip(super().predict_proba(x), 1e-6, 1 - 1e-6)

    def predict_log_proba(self, x):
        return np.log(self.predict_proba(x))


class LogR(ClipExtremes, _logr):
    pass


class LDA(ClipExtremes, _lda):
    pass


class QDA(ClipExtremes, _qda):
    pass


class SVC(ClipExtremes, _svc):
    pass


class RF(ClipExtremes, _rf):
    pass


class MLP(ClipExtremes, _mlp):
    pass


def get_sklearn_model(clf_name):
    if clf_name == "logr":
        return LogR(class_weight="balanced", max_iter=1000)
    elif clf_name == "lda":
        return LDA(solver="lsqr", shrinkage="auto")
    elif clf_name == "qda":
        return QDA(reg_param=0.1)
    elif clf_name == "svc":
        return SVC(class_weight="balanced", kernel="linear", probability=True)
    elif clf_name == "rf":
        return RF(class_weight="balanced")
    elif clf_name == "mlp":
        return MLP(hidden_layer_sizes=(256, 100), max_iter=5000, early_stopping=True)
    else:
        raise ValueError(f"Unknown clf_name: {clf_name}")
