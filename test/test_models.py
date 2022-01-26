import numpy as np
import pytest

from bci_disc_models.models import (
    AlwaysClass0,
    AlwaysClass1,
    GenerativeBaseline,
    ScikitModelAdaptor,
    TrialNeuralNet,
)
from bci_disc_models.utils import seed_everything

seed_everything(0)
TRIALS, CHANNELS, TIME = 128, 16, 100


@pytest.fixture()
def x1():
    return np.random.randn(TRIALS, CHANNELS, TIME)


@pytest.fixture()
def x2():
    return np.random.randn(TRIALS, CHANNELS, TIME)


@pytest.fixture()
def y1():
    return np.random.randint(0, 2, TRIALS)


@pytest.fixture()
def y2():
    return np.random.randint(0, 2, TRIALS)


def _cmp_models(model1, model2, tmp_path, x):
    preds1 = model1.predict(x)
    path = model1.save(tmp_path)

    preds2 = model2.predict(x)
    model2.load(path.parent)
    preds3 = model2.predict(x)

    assert not np.all(preds1 == preds2)
    assert np.all(preds1 == preds3)


class TestScikitModelAdaptor:
    def test_save_logr(self, x1, y1, x2, y2, tmp_path):
        model1 = ScikitModelAdaptor(clf_name="logr", prior_p_target_in_query=0.5).fit(x1, y1)
        model2 = ScikitModelAdaptor(clf_name="logr", prior_p_target_in_query=0.5).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)

    def test_save_lda(self, x1, y1, x2, y2, tmp_path):
        model1 = ScikitModelAdaptor(clf_name="lda", prior_p_target_in_query=0.5).fit(x1, y1)
        model2 = ScikitModelAdaptor(clf_name="lda", prior_p_target_in_query=0.5).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)

    def test_save_svc(self, x1, y1, x2, y2, tmp_path):
        model1 = ScikitModelAdaptor(clf_name="svc", prior_p_target_in_query=0.5).fit(x1, y1)
        model2 = ScikitModelAdaptor(clf_name="svc", prior_p_target_in_query=0.5).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)

    def test_save_rf(self, x1, y1, x2, y2, tmp_path):
        model1 = ScikitModelAdaptor(clf_name="rf", prior_p_target_in_query=0.5).fit(x1, y1)
        model2 = ScikitModelAdaptor(clf_name="rf", prior_p_target_in_query=0.5).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)

    def test_save_mlp(self, x1, y1, x2, y2, tmp_path):
        model1 = ScikitModelAdaptor(clf_name="mlp", prior_p_target_in_query=0.5).fit(x1, y1)
        model2 = ScikitModelAdaptor(clf_name="mlp", prior_p_target_in_query=0.5).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)


class TestGenerativeBaseline:
    def test_save(self, x1, y1, x2, y2, tmp_path):
        model1 = GenerativeBaseline(0.8).fit(x1, y1)
        model2 = GenerativeBaseline(0.8).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)


class TestTrialNeuralNet:
    def test_save(self, x1, y1, x2, y2, tmp_path):
        kw = dict(
            n_classes=2, input_shape=(CHANNELS, TIME), epochs=5, lr=1e-3, arch="eegnet", prior_p_target_in_query=0.5
        )
        seed_everything(0)
        model1 = TrialNeuralNet(**kw).fit(x1, y1)
        seed_everything(1)
        model2 = TrialNeuralNet(**kw).fit(x2, y2)
        _cmp_models(model1, model2, tmp_path, x1)


class TestControlModels:
    def test_AlwaysClass0(self, x1, y1):
        model = AlwaysClass0(0.1).fit(x1, y1)
        preds = model.predict(x1)
        assert np.all(preds == 0)

    def test_AlwaysClass1(self, x1, y1):
        model = AlwaysClass1(0.1).fit(x1, y1)
        preds = model.predict(x1)
        assert np.all(preds == 1)
