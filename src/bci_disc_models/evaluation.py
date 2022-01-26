from typing import Tuple

import numpy as np
from scipy.special import logsumexp
from tqdm import trange

from bci_disc_models.models import BaseDiscriminativeModel


def get_query(
    query_selection_method: str,
    log_probs: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    query_size: int,
    target_letter_alphabet_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns data, labels, and queried_letter_indices.

    Args:
        query_selection_method (str): How to choose K letters to present
        log_probs (np.ndarray): current log probabilities for each letter in alphabet
        test_data (np.ndarray): Test data, from which we'll grab a matching item
        test_labels (np.ndarray):
        query_size (int): How many letters to present (K)
        target_letter_alphabet_idx (int): Index of target letter in alphabet

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        After choosing a query to present, returns matching data, label, and queried letter indices
    """
    p = np.exp(log_probs)
    zeros = np.where(p == 0)[0]
    nonzeros = np.where(p > 0)[0]
    if len(nonzeros) < query_size:
        # Edge case: most letters have 0 probability. Take all the non-zero, plus several random
        queried_letter_indices = np.concatenate(
            [nonzeros, np.random.choice(zeros, size=query_size - len(nonzeros), replace=False)]
        )
    elif query_selection_method == "sampleK":
        queried_letter_indices = np.random.choice(np.arange(len(p)), size=query_size, replace=False, p=p)
    elif query_selection_method == "topK":
        # If all uniform, just shuffle and take first K. (otherwise, with
        # fixed seed and uniform prior, we always begin with first K letters of alphabet)
        # If not uniform, sort and take first K.
        p0 = p[0]
        if np.allclose(p, p0 * np.ones_like(p)):
            perm = np.random.permutation(len(p))
            queried_letter_indices = perm[:query_size]
        else:
            queried_letter_indices = np.argpartition(p, -query_size)[-query_size:]
    else:
        raise ValueError(f"Invalid query_selection_method: {query_selection_method}")

    np.random.shuffle(queried_letter_indices)
    res_data, res_label = [], []
    target_idx = np.where(test_labels == 1)[0]
    nontarget_idx = np.where(test_labels == 0)[0]
    for idx in queried_letter_indices:
        if idx == target_letter_alphabet_idx:  # Pick a random target item from test data
            random_idx = np.random.choice(target_idx)
        else:  # Pick a random non-target item from test data
            random_idx = np.random.choice(nontarget_idx)
        res_data.append(test_data[random_idx])
        res_label.append(test_labels[random_idx])
    return np.stack(res_data), np.stack(res_label), queried_letter_indices


def simulate_typing(
    model: BaseDiscriminativeModel,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    trials_per_sequence: int,
    n_chars_to_spell: int,
    alphabet_len: int,
    query_selection_method: str = "sampleK",
    decision_threshold: float = 0.8,
):
    """
    A single unified function to evaluate sequence- and trial-based RSVP typing models.

    The overall loop for models is always the same. For each attempt at typing a letter:
    1. Start from a textual context (e.g. "PIZZ_", trying to type "A")
    2. Apply a language model to get a prior over the alphabet. (just uniform to begin with)
    3. Repeat until we reach a stopping condition:
        - Get a query sequence, based on the current alphabet distribution
          (Note that we know the position of correct letter, if it is present. This gives sequence label)
        - "Present this sequence to the user" - We mock the user response by finding a data item from
          test set with the correct sequence label
        - Feed the selected data to the model, getting a log-likelihood term
        - Update the alphabet posterior and re-evaluate the stopping conditions

        We collect statistics to answer the following questions:
    1. How many queries are needed (on average) per letter attempt? (NOTE that this is clipped at MAX_SEQUENCES)
    2. How many seconds per letter (requires assuming time layout of each query and timing between queries)
        (NOTE - this should consider the device sample rate)
    3. ITR or related measures? (might not be possible without a real language model and real text history?)
    """
    assert 0 < decision_threshold < 1, f"Invalid {decision_threshold=}"

    all_true_letter_idx = []  # e.g. true letter is "A", so idx=0
    all_actual_labels = []  # First sequence was [nontarget, nontarget, target, nontarget, nontarget], so label=2
    all_pred_log_likelihoods = []  # The log-likelihood update given by model in each iteration
    all_pred_log_posteriors = []  # After apply that log-likelihood update, the log-posterior at that iteration
    all_pred_letter_idx = []  # We try this letter for a while and eventually decide to type some letter
    all_queried_letter_indices = []  # Which letters did we present to the user in each iteration?
    for _ in trange(n_chars_to_spell, desc="Letters", leave=True, position=0):
        # Get text context (skipped for now)
        # Get prior over alphabet (uniform for now)
        log_probs = np.log(np.ones(alphabet_len) / alphabet_len)

        # Present sequences and update, until stopping condition
        finished_sequence_attempts = False
        attempts_remaining = 10
        this_letter_labels = []
        this_letter_log_likelihoods = []
        this_letter_log_posteriors = []
        this_letter_queried_letter_indices = []
        predicted_letter_idx = None
        true_letter_idx = 0  # NOTE - just pretending the target letter is always "A"
        while not finished_sequence_attempts and attempts_remaining > 0:
            attempts_remaining -= 1

            # Choose query letters according to current probs,
            # and find matching data items from test set
            data, label, queried_letter_indices = get_query(
                query_selection_method=query_selection_method,
                log_probs=log_probs,
                test_data=test_data,
                test_labels=test_labels,
                query_size=trials_per_sequence,
                target_letter_alphabet_idx=true_letter_idx,
            )
            this_letter_labels.append(label.copy())
            this_letter_queried_letter_indices.append(queried_letter_indices.copy())

            # Feed the selected data to the model, getting a log-likelihood term for each presented letter
            # Store each log-likelihood term at the same index as the queried letter
            log_likelihoods = model.predict_log_likelihoods(data, queried_letter_indices, alphabet_len)
            this_letter_log_likelihoods.append(log_likelihoods.copy())

            # Update the alphabet posterior and re-normalize
            log_probs += log_likelihoods
            log_probs -= logsumexp(log_probs)
            this_letter_log_posteriors.append(log_probs.copy())

            # Check if we have finished sequence attempt
            if np.any(np.exp(log_probs) >= decision_threshold):
                predicted_letter_idx = np.where(np.exp(log_probs) >= decision_threshold)[0][0]
                finished_sequence_attempts = True

        all_actual_labels.append(this_letter_labels)
        all_pred_log_likelihoods.append(this_letter_log_likelihoods)
        all_pred_log_posteriors.append(this_letter_log_posteriors)
        all_pred_letter_idx.append(predicted_letter_idx)
        all_true_letter_idx.append(true_letter_idx)
        all_queried_letter_indices.append(this_letter_queried_letter_indices)
    return {
        "true_labels": all_actual_labels,
        "true_letter_indices": all_true_letter_idx,
        "pred_log_likelihoods": all_pred_log_likelihoods,
        "pred_log_posteriors": all_pred_log_posteriors,
        "pred_letter_indices": all_pred_letter_idx,
        "queried_letter_indices": all_queried_letter_indices,
    }
