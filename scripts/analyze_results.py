"""Print results table from parsed results and metrics."""
import pickle as pkl

import numpy as np
from rich.console import Console
from rich.table import Table

from bci_disc_models.conf import METRICS, MODELS
from bci_disc_models.utils import PROJECT_ROOT

results_dir = PROJECT_ROOT / "results"

with open(results_dir / "parsed_results.pkl", "rb") as f:
    parsed_results = pkl.load(f)

columns = ["Model"] + METRICS
rows = []
for model in MODELS:
    row = []
    row.append(model)
    for metric in METRICS:
        mean = np.mean(parsed_results[model][metric])
        std = np.std(parsed_results[model][metric])
        row.append(f"{mean:.3f} ± {std:.3f}")
    rows.append(row)


table = Table(title="Model Comparison")
for col in columns:
    table.add_column(col, no_wrap=True)
for row in rows:
    table.add_row(*list(map(str, row)))
console = Console()
console.print(table)
