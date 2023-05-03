# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product

import xarray as xr
import copy
from qrennd.utils.analysis import (
    LogicalError,
    lmfit_par_to_ufloat,
)
from qrennd import Config, Layout, load_datasets
import pymatching
import stim

# %%
EXP_NAME = "20230428-d3_xzzx-google_no-assign"
MODEL_FOLDER = "pymatching"
LAYOUT_NAME = "d3_rotated_layout.yaml"
DATASET_NAME = "test"

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

DATA_DIR = NOTEBOOK_DIR / "data"
if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")

OUTPUT_DIR = NOTEBOOK_DIR / "output"
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

CONFIG_FILE = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / "config.yaml"
if not CONFIG_FILE.exists():
    raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

LAYOUT_FILE = DATA_DIR / EXP_NAME / f"config/{LAYOUT_NAME}"
if not LAYOUT_FILE.exists():
    raise ValueError(f"Layout file does not exist: {LAYOUT_FILE}")

# %% [markdown]
# # Evaluation


# %%
def evaluate_MWPM(config, layout, dataset_name="test"):
    # metadata
    experiment_name = config.dataset["folder_format_name"]
    dataset_params = config.dataset[dataset_name]
    dataset_dir = config.experiment_dir / dataset_name
    basis = "X" if config.dataset["rot_basis"] else "Z"

    # dataset
    test_data = load_datasets(
        config=config, layout=layout, dataset_name=dataset_name, concat=False
    )
    rounds = config.dataset[dataset_name]["rounds"]
    states = config.dataset[dataset_name]["states"]
    num_shots = config.dataset[dataset_name]["shots"]
    sequences = product(rounds, states)
    list_errors = []

    for data, (num_rounds, state) in zip(test_data, sequences):
        print(f"QEC = {num_rounds} | state = {state}", end="\r")
        anc_defects = data.rec_input
        final_defects = data.eval_input
        log_errors = data.log_errors

        # reshape data for MWPM
        anc_defects = anc_defects.reshape(
            anc_defects.shape[0], anc_defects.shape[1] * anc_defects.shape[2]
        )
        defects = np.concatenate([anc_defects, final_defects], axis=1)

        # load MWPM decoder
        experiment = experiment_name.format(
            basis=basis,
            state=state,
            num_rounds=num_rounds,
            **dataset_params,
        )
        detector_error_model = stim.DetectorErrorModel.from_file(
            dataset_dir / experiment / "detector_error_model.dem"
        )
        MWPM = pymatching.Matching(detector_error_model)

        # decode
        predictions = np.array([MWPM.decode(i) for i in defects])
        errors = predictions.flatten() != log_errors.flatten()
        list_errors.append(errors)
        print(
            f"QEC = {num_rounds} | state = {state} | avg_errors = {np.average(errors):.4f}",
            end="\r",
        )

    list_errors = np.array(list_errors).reshape(len(rounds), len(states), num_shots)

    log_fid = xr.Dataset(
        data_vars=dict(errors=(["qec_round", "state", "shot"], list_errors)),
        coords=dict(qec_round=rounds, state=states, shot=list(range(1, num_shots + 1))),
    )

    return log_fid


# %%
layout = Layout.from_yaml(LAYOUT_FILE)
config = Config.from_yaml(
    filepath=CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)


# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
FILE_NAME = DATASET_NAME + ".nc"
if not (DIR / FILE_NAME).exists():
    print("Evaluating MWPM...")

    log_fid = evaluate_MWPM(config, layout, DATASET_NAME)
    log_fid.to_netcdf(path=DIR / FILE_NAME)

log_fid = xr.load_dataset(DIR / FILE_NAME)

# %%
x = log_fid.qec_round.values
y = log_fid.errors.mean(dim=["shot", "state"]).values

fig, ax = plt.subplots()

for FIXED_TO, fmt in zip([True, False], ["b--", "b-"]):
    model_decay = LogicalError(fixed_t0=FIXED_TO)
    params = model_decay.guess(y, x=x)
    out = model_decay.fit(y, params, x=x, min_qec=layout.distance)
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    t0 = lmfit_par_to_ufloat(out.params["t0"])

    x_fit = np.linspace(layout.distance, max(x), 100)
    y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)
    label = f"$\\epsilon_L = (${error_rate*100})%"
    if FIXED_TO:
        label += "\nwith fixed $t_0 = 0$"

    ax.plot(x_fit, y_fit, fmt, label=label)

ax.plot(x, y, "b.")

ax.set_xlabel("QEC round")
ax.set_ylabel("logical error")
ax.set_title("MWPM")
ax.set_ylim(0, 0.5)
ax.set_xlim(0, max(x) + 1)
ax.set_yticks(np.arange(0, 0.51, 0.05), np.round(np.arange(0, 0.51, 0.05), decimals=2))
ax.legend(loc="best")
ax.grid(which="major")
fig.tight_layout()
fig.savefig(DIR / f"{DATASET_NAME}_log-err_vs_QEC-round.pdf", format="pdf")
plt.show()

# %%
