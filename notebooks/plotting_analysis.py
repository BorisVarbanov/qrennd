# %%
import pathlib
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import yaml

from qrennd.utils.analysis import (
    logical_fidelity,
    LogicalFidelityDecay,
    lmfit_par_to_ufloat,
)

# %%
TEMPLATE = "nn_vs_pymatching.yaml"

# %%
PATH_TEMPLATES = pathlib.Path.cwd() / "plot_templates"
with open(PATH_TEMPLATES / TEMPLATE) as file:
    setup = yaml.full_load(file)


def format(string: str, formatter: dict) -> str:
    if string:
        return string.format(**formatter)
    return


# %%
DIR = pathlib.Path.cwd() / "output"

NON_DATASETS = ["description", "vars", "figure", "output_name", "output_dir"]
datasets = [v for k, v in setup.items() if k not in NON_DATASETS]
variables = setup.get("vars")

# %%
fig, ax = plt.subplots(figsize=(7, 5))

for dataset in datasets:
    log_fid = xr.load_dataset(DIR / format(dataset["dataset"], variables))
    x, y = log_fid.qec_round.values, log_fid.avg.values
    if data := dataset.get("data"):
        yerr = log_fid.err.values if data.get("errorbar") else 0
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=data.get("fmt"),
            color=data.get("color"),
            markersize=10,
            capsize=2,
            label=format(data.get("label"), variables),
        )

    if fit := dataset.get("fit"):
        model_decay = LogicalFidelityDecay(fixed_t0=fit.get("fixed_t0"))
        params = model_decay.guess(y, x=x)
        out = model_decay.fit(y, params, x=x, min_qec=fit.get("min_qec"))
        error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
        t0 = lmfit_par_to_ufloat(out.params["t0"])

        x_fit = np.linspace(fit.get("min_qec"), max(x), 100)
        y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)
        vars_fit = {
            "error_rate": error_rate,
            "t0": t0,
            "error_rate_100": error_rate * 100,
        }
        vars_fit.update(variables)
        ax.plot(
            x_fit,
            y_fit,
            linestyle=fit.get("fmt"),
            color=fit.get("color"),
            label=format(fit.get("label"), vars_fit),
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
ax.set_title(format(setup["figure"].get("title"), variables))
fig.tight_layout()
fig.savefig(
    DIR
    / format(setup["output_dir"], variables)
    / format(setup["output_name"] + ".pdf", variables),
    format="pdf",
)
fig.savefig(
    DIR
    / format(setup["output_dir"], variables)
    / format(setup["output_name"] + ".png", variables),
    format="png",
)
plt.show()
