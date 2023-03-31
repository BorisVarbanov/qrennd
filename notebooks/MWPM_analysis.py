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
EXP_NAME = "20230310-d3_rot-sruf_circ-level_meas-reset"
MODEL_FOLDER = "MWPM"
LAYOUT_NAME = "d3_rotated_layout.yaml"
DATASET_NAME = "test_MWPM_assign0-005"

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
        for batch, state in zip(test_data, dataset_params["states"]):
            # assuming there is no reshufling in the test_data

            # reshape data for pymatching
            inputs, log_errors = batch
            anc_defects = np.array(inputs["lstm_input"])
            anc_defects = anc_defects.reshape(
                anc_defects.shape[0], anc_defects.shape[1] * anc_defects.shape[2]
            )
            final_defects = np.array(inputs["eval_input"])
            defects = np.concatenate([anc_defects, final_defects], axis=1)

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
            log_fid = 1 - np.average(predictions.flatten() ^ log_errors.flatten())
            log_fid_list.append(log_fid)  # all batches have same number of shots

        ROUNDS.append(rounds)
        LOG_FID.append(np.average(log_fid_list))

    # convert to xr.DataArray
    log_fid = xr.DataArray(data=LOG_FID, coords=dict(qec_round=ROUNDS), name="log_fid")

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
model_decay = LogicalFidelityDecay()
params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
out = model_decay.fit(
    log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
)
ax = out.plot_fit()
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_title("MWPM")
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(DIR / f"{DATASET_NAME}_log-fid_vs_QEC-round.pdf", format="pdf")
plt.show()

# %%
error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
print("error_rate=", error_rate)
print(log_fid.log_fid)

# %%
