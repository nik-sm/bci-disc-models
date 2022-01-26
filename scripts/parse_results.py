"""Parse saved results from model evaluation into a convenient form."""
import pickle as pkl
from pprint import pprint

import numpy as np

from bci_disc_models.conf import (
    ALPHABET_LEN,
    DECISION_THRESHOLD,
    METRICS,
    MODELS,
    TARGET_LETTER_IDX,
    TRIALS_PER_SEQUENCE,
)
from bci_disc_models.utils import PROJECT_ROOT


def ITR(N, P):
    if P == 0:
        return 0
    if P == 1:
        return np.log2(N)
    return np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))


def R(x):
    return round(x, 4)


def compute_ITR_stats(typing_history):
    # Typing history contains lists with ragged shapes, such as: (attempted_letters, queries_per_letter, ALPHABET_LEN)

    # Extract parameters used during typing
    alphabet_len = typing_history.get("alphabet_len", ALPHABET_LEN)
    trials_per_sequence = typing_history.get("trials_per_sequence", TRIALS_PER_SEQUENCE)
    decision_threshold = typing_history.get("decision_threshold", DECISION_THRESHOLD)
    target_letter_idx = typing_history.get("target_letter_idx", TARGET_LETTER_IDX)

    # Compute ITR in terms of bits per attempted letter using decision threshold, or using argmax
    all_log_posteriors = typing_history["pred_log_posteriors"]
    n_correct_decision_threshold = 0  # Compute accuracy according to DECISION_THRESHOLD
    n_correct_argmax = 0  # Compute accuracy using argmax on the final log_posteriors
    n_total = len(all_log_posteriors)
    for log_posteriors in all_log_posteriors:
        if np.exp(log_posteriors[-1][target_letter_idx]) >= decision_threshold:
            n_correct_decision_threshold += 1

        if np.argmax(log_posteriors[-1]) == target_letter_idx:
            n_correct_argmax += 1

    # Compute ITR of each query, using argmax amongst the K+1 probability buckets of each update
    n_correct_query = 0
    n_total_query = 0
    Z = zip(typing_history["pred_log_likelihoods"], typing_history["queried_letter_indices"])
    for several_log_likelihoods, several_queried_letter_idx in Z:
        # While trying to type this letter, we showed multiple queries
        for log_likelihoods, queried_letter_idx in zip(several_log_likelihoods, several_queried_letter_idx):
            # In each query, there are 11 "buckets" - the 10 letters shown, and everything else.
            # If we take the log likelihoods, group into these buckets, normalize, and take argmax,
            # we can ask whether the target letter's bucket was chosen. If so, we count it as a correct query.
            trials_per_sequence = len(queried_letter_idx)
            n_buckets = trials_per_sequence + 1
            buckets = np.zeros(n_buckets)
            buckets[:trials_per_sequence] = np.exp(log_likelihoods[queried_letter_idx])
            unseen_letter_idx = np.setdiff1d(np.arange(alphabet_len), queried_letter_idx)
            buckets[-1] = np.sum(np.exp(log_likelihoods[unseen_letter_idx]))

            # Determine which bucket is "correct".
            # If the target letter is in queried_letter_idx, that is the correct bucket.
            # Otherwise, buckets[-1] is the correct bucket.
            n_total_query += 1
            if target_letter_idx in queried_letter_idx:
                correct_bucket_idx = np.where(queried_letter_idx == target_letter_idx)[0][0]
            else:
                correct_bucket_idx = n_buckets - 1
            if buckets.argmax() == correct_bucket_idx:
                n_correct_query += 1

    typed_letters = np.array(typing_history["pred_letter_indices"])
    n_typed = np.sum(typed_letters != None)
    n_correct_typed = np.sum(typed_letters == target_letter_idx)
    acc_typed = n_correct_typed / n_typed if n_typed > 0 else 0

    acc_decision_threshold = n_correct_decision_threshold / n_total
    acc_argmax = n_correct_argmax / n_total
    acc_query = n_correct_query / n_total_query

    return {
        "acc_typed": acc_typed,
        "acc_decision_threshold": acc_decision_threshold,
        "acc_argmax": acc_argmax,
        "acc_query": acc_query,
        "itr_typed": R(ITR(alphabet_len, acc_typed)),
        "itr_decision_threshold": R(ITR(alphabet_len, acc_decision_threshold)),
        "itr_argmax": R(ITR(alphabet_len, acc_argmax)),
        "itr_query": R(ITR(alphabet_len, acc_query)),
        "n_typed": n_typed,
    }


results_dir = PROJECT_ROOT / "results"
# For each model, we have several metrics to track.
# Each metric will have a list of 5 values (one for each seed).
all_results = {model: {metric: [] for metric in METRICS} for model in MODELS}
for seed in range(5):
    typing_history_files = list((results_dir / f"seed_{seed}").glob("typing_stats.*.pkl"))
    for file in typing_history_files:
        model = file.stem.replace("typing_stats.", "")
        with open(file, "rb") as f:
            hist = pkl.load(f)
        res = {}
        res["n_params"] = hist["n_params"]
        itr_stats = compute_ITR_stats(hist)
        res.update(**itr_stats)
        res["test_acc"] = hist["test_acc"]
        res["test_bal_acc"] = hist["test_bal_acc"]
        for metric in METRICS:
            all_results[model][metric].append(R(res[metric]))
pprint(all_results)
with open(results_dir / "parsed_results.pkl", "wb") as f:
    pkl.dump(all_results, f)
