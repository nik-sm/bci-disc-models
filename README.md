Code for "Recursive Estimation of User Intent from Noninvasive Electroencephalography using Discriminative Models" by Niklas Smedemark-Margulies, Basak Celik, Tales Imbiriba, Aziz Kocanaogullari, and Deniz Erdogmus

In this work, we seek to infer a user's desired symbol from EEG measurements during a query-and-response typing task.
We derive a framework for recursively estimating the desired posterior probabilities of symbols using classifier models such as deep neural networks.
We construct a simulated typing task for evaluating performance, and find that this approach outperforms baseline approaches that compute these probabilities using generative models.

We use https://pypi.org/project/thu-rsvp-dataset/1.1.0/ for fetching and preprocessing benchmark dataset from https://www.frontiersin.org/articles/10.3389/fnins.2020.568000/full.

# Setup

Setup project with `make` and activate virtualenv with `source venv/bin/activate`

# Usage

To reproduce our experiments, please follow these steps:

1. Preprocess data: `python scripts/prepare_data.py`
2. Pretrain models: `python scripts/train.py`
3. Evaluate models in simulated typing task: `python scripts/evaluate.py`
4. Parse saved results from evaluation: `python scripts/parse_results.py`
5. Collect statistics from parsed results: `python scripts/analyze_results.py`
6. Make plots: `python scripts/plots.py`

To run tests: `pytest --disable-warnings -s`
# Citation

If you use this code, please cite our paper:
```bibtex
@misc{2211.02630,
    Author = {Niklas Smedemark-Margulies and Basak Celik and Tales Imbiriba and Aziz Kocanaogullari and Deniz Erdogmus},
    Title = {Recursive Estimation of User Intent from Noninvasive Electroencephalography using Discriminative Models},
    Year = {2022},
    Eprint = {arXiv:2211.02630},
}
```
