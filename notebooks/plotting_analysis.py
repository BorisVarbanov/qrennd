# %%
import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

from qrennd.utils.analysis import (
    logical_fidelity,
    LogicalFidelityDecay,
    lmfit_par_to_ufloat,
)

# %%
EXP_NAME = pathlib.Path("20230418-d3_simulated_google_20M")
DATASETS = [
    EXP_NAME / "20230419-224354_best-config_center_7_5" / "test.nc",
    EXP_NAME / "20230419-224354_best-config_center_7_5" / "test_exp.nc",
]
COLORS = ["blue", "red", "orange", "green"]
LABELS = [
    "sim",
    "exp",
    "center_3_5",
    "center_5_3",
]

ERRORBARS = False
MIN_QEC_FIT = 3

OUTPUT_FOLDER = EXP_NAME / "20230419-224354_best-config_center_7_5"
OUTPUT_NAME = "comparison_sim-exp"

TITLE = None

# %%
DIR = pathlib.Path.cwd() / "output"

fig, ax = plt.subplots()

for dataset, color, label in zip(DATASETS, COLORS, LABELS):
    log_fid = xr.load_dataset(DIR / dataset)
    x, y = log_fid.qec_round.values, log_fid.avg.values
    yerr = log_fid.err.values if ERRORBARS else 0
    ax.errorbar(
        x, y, yerr=yerr, fmt=".", color=color, markersize=10, capsize=2, label=label
    )

    model_decay = LogicalFidelityDecay(fixed_t0=False)
    params = model_decay.guess(y, x=x)
    out = model_decay.fit(y, params, x=x, min_qec=MIN_QEC_FIT)
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    t0 = lmfit_par_to_ufloat(out.params["t0"])

    x_fit = np.linspace(MIN_QEC_FIT, max(x), 100)
    y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)
    ax.plot(
        x_fit, y_fit, "-", color=color, label=f"$\\epsilon_L = (${error_rate*100})%"
    )

ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xlim(xmin=0)
ax.set_ylim(0.5, 1)
ax.set_yticks(
    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
)
ax.legend(loc="best")
ax.grid(which="major")
ax.set_title(TITLE)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(DIR / OUTPUT_FOLDER / f"{OUTPUT_NAME}.pdf", format="pdf")
fig.savefig(DIR / OUTPUT_FOLDER / f"{OUTPUT_NAME}.png", format="png")
plt.show()
