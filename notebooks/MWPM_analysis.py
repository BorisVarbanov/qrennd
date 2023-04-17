# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import xarray as xr
import copy
from qrennd.utils.analysis import (
    logical_fidelity,
    LogicalFidelityDecay,
    lmfit_par_to_ufloat,
)
from qrennd import Config, Layout, load_datasets
import pymatching
import stim

# %%
EXP_NAME = "20230412-d3_rot-surf_no-Y_no-assign"
MODEL_FOLDER = "MWPM"
LAYOUT_NAME = "d3_rotated_layout.yaml"
DATASET_NAME = "test_MWPM_assign0-010"

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
    ROUNDS = []
    LOG_FID = []
    STD = []
    for rounds in config.dataset[dataset_name]["rounds"]:
        print("QEC round = ", rounds, end="\r")
        config_ = copy.deepcopy(config)
        config_.dataset[dataset_name]["rounds"] = [rounds]
        config_.train["batch_size"] = config_.dataset[dataset_name]["shots"]
        test_data = load_datasets(
            config=config_, layout=layout, dataset_name=dataset_name
        )

        # metadata
        experiment_name = config_.dataset["folder_format_name"]
        dataset_params = config_.dataset[dataset_name]
        dataset_dir = config_.experiment_dir / dataset_name
        rot_basis = config_.dataset["rot_basis"]
        basis = "X" if rot_basis else "Z"

        log_fid_list = []
        std_list = []
        for batch, state in zip(test_data, dataset_params["states"]):
            # assuming there is no reshufling in the test_data

            # reshape data for pymatching
            inputs, log_errors = batch
            anc_defects = np.array(inputs["rec_input"])
            anc_defects = anc_defects.reshape(
                anc_defects.shape[0], anc_defects.shape[1] * anc_defects.shape[2]
            )
            final_defects = np.array(inputs["eval_input"])
            defects = np.concatenate([anc_defects, final_defects], axis=1)
            log_errors = log_errors.astype(int)  # convert from float

            # load MWPM decoder
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                num_rounds=rounds,
                **dataset_params,
            )
            detector_error_model = stim.DetectorErrorModel.from_file(
                dataset_dir / experiment / "detector_error_model.dem"
            )
            MWPM = pymatching.Matching(detector_error_model)

            # decode and log fidelity
            predictions = np.array([MWPM.decode(i) for i in defects])
            correct = predictions.flatten() == log_errors.flatten()
            log_fid = np.average(correct)
            std = np.std(correct)

            log_fid_list.append(log_fid)  # all batches have same number of shots
            std_list.append(std)  # all batches have same number of shots

        ROUNDS.append(rounds)
        LOG_FID.append(np.average(log_fid_list))
        STD.append(np.average(std_list))

    # convert to xr.DataArray
    log_fid = xr.Dataset(
        data_vars=dict(avg=(["qec_round"], LOG_FID), err=(["qec_round"], STD)),
        coords=dict(qec_round=ROUNDS),
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
FILE_NAME = DATASET_NAME + "_results.nc"
if not (DIR / FILE_NAME).exists():
    print("Evaluating MWPM...")

    log_fid = evaluate_MWPM(config, layout, DATASET_NAME)
    log_fid.to_netcdf(path=DIR / FILE_NAME)

log_fid = xr.load_dataset(DIR / FILE_NAME)

# %%
x = log_fid.qec_round.values
y = log_fid.avg.values
yerr = log_fid.err.values

fig, ax = plt.subplots()

for FIXED_TO, fmt in zip([True, False], ["b--", "b-"]):
    model_decay = LogicalFidelityDecay(fixed_t0=FIXED_TO)
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

ax.errorbar(x, y, yerr=yerr, fmt="b.", capsize=2, label="MWPM")

ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_title("MWPM")
ax.set_ylim(0.5, 1)
ax.set_xlim(0, max(x) + 1)
ax.set_yticks(
    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
)
ax.legend(loc="best")
ax.grid(which="major")
fig.tight_layout()
fig.savefig(DIR / f"{DATASET_NAME}_log-fid_vs_QEC-round.pdf", format="pdf")
plt.show()

# %%
