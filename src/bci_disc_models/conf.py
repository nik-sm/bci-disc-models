from bci_disc_models.models import (
    AlwaysClass0,
    AlwaysClass1,
    GenerativeBaseline,
    ScikitModelAdaptor,
    TrialNeuralNet,
)

MODELS = [
    "disc-always0",
    "disc-always1",
    "disc-logr",
    "disc-nn-eegnet",
    "disc-nn-simple-cnn-1d",
    "disc-nn-simple-cnn-2d",
    "gen-lda-empirical-prior",
    "gen-lda-uniform-prior",
    "gen-logr-empirical-prior",
    "gen-logr-uniform-prior",
]
METRICS = [
    "n_params",
    "test_acc",
    "test_bal_acc",
    # NOTE - "itr_decision_threshold" is the notion of ITR we focus on, since
    # it corresponds most directly to actual typing performance
    "itr_decision_threshold",
    "itr_argmax",
    "itr_query",
    "itr_typed",
    "n_typed",
]

TARGET_LETTER_IDX = 0  # TODO - this is hardcoded in 2 places (see src/evaluation.py)
ALPHABET_LEN = 28
DECISION_THRESHOLD = 0.8
TRIALS_PER_SEQUENCE = 10

# TODO - ideally this should be tunable when the model is being used during
# simulated typing. Right now, it is stored on the model so it is needed when model
# is created during training as well.
# Likewise - TRIALS_PER_SEQUENCE should be tunable during typing.
PRIOR_P_TARGET_IN_QUERY = TRIALS_PER_SEQUENCE / ALPHABET_LEN

# Add discriminative models
NAME_MODEL_KWARGS = []
for arch in ["eegnet", "simple-cnn-1d", "simple-cnn-2d"]:
    NAME_MODEL_KWARGS.append(
        (
            f"disc-nn-{arch}",
            TrialNeuralNet,
            dict(
                prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY,
                arch=arch,
                input_shape=(62, 63),
                n_classes=2,
                epochs=25,
                lr=0.001,
            ),
        )
    )
for clf_name in ["logr"]:
    NAME_MODEL_KWARGS.append(
        (
            f"disc-{clf_name}",
            ScikitModelAdaptor,
            dict(prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY, clf_name=clf_name),
        )
    )
# Add control discr models
NAME_MODEL_KWARGS.append(("disc-always0", AlwaysClass0, dict(prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY)))
NAME_MODEL_KWARGS.append(("disc-always1", AlwaysClass1, dict(prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY)))
# Add generative models
for prior_type in ["uniform", "empirical"]:
    for clf_name in ["lda", "logr"]:
        name = f"gen-{clf_name}-{prior_type}-prior"
        NAME_MODEL_KWARGS.append(
            (name, GenerativeBaseline, dict(prior_type=prior_type, pca_n_components=0.8, clf_name=clf_name))
        )
