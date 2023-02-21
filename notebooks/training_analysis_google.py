# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "training_logs_google_simulated"
MODEL_FOLDER: str = "20230221-133842_first_try_google_simulated"

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

# experiment folder
EXP_DIR = NOTEBOOK_DIR / "data" / EXP_NAME
if not EXP_DIR.exists():
    raise ValueError("Experiment directory does not exist.")

LOG_FILE = EXP_DIR / f"{MODEL_FOLDER}/logs/training.log"
if not LOG_FILE.exists():
    raise ValueError(f"Log file does not exist: {MODEL_FOLDER}/logs/training.log")

# %%
dataframe = pd.read_csv(LOG_FILE)

# %%
dataframe

# %%
METRICS = ("loss", "main_output_accuracy")
EPOCH_CUT = 50
acc_MWPM = 0.7696 # Google's data simulated

for metric in METRICS:
    fig, axs = plt.subplots(figsize=(10, 4), ncols=2)

    axs[0].plot(
        dataframe.epoch, dataframe[metric], ".-", color="blue", label="Training"
    )
    axs[0].plot(
        dataframe.epoch,
        dataframe["val_" + metric],
        ".-",
        color="orange",
        label="Validation",
    )

    axs[1].plot(
        dataframe.epoch[EPOCH_CUT:],
        dataframe[metric][EPOCH_CUT:],
        ".-",
        color="blue",
        label="Training",
    )
    axs[1].plot(
        dataframe.epoch[EPOCH_CUT:],
        dataframe["val_" + metric][EPOCH_CUT:],
        ".-",
        color="orange",
        label="Validation",
    )

    if metric == "main_output_accuracy":
        axs[0].axhline(y=acc_MWPM, linestyle="--", color="gray", label="MWPM (test)")

    axs[0].legend(frameon=False)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel(metric.replace("_", " ").capitalize())
    axs[1].legend(frameon=False)
    axs[1].set_xlabel("Epochs")
    axs[1].set_xlim(EPOCH_CUT, max(dataframe.epoch) + 1)

plt.show()

# %% [markdown]
# # Evaluation
#
# ## 1) Test simulated data

# %%
import xarray as xr
from qrennd import get_model, Config
from qrennd.utils.syndrome import get_syndromes, get_defects
from qrennd.utils.analysis import (
    logical_fidelity,
    LogicalFidelityDecay,
    lmfit_par_to_ufloat,
)
from qrennd.datasets import DataGeneratorGoogle
from qrennd.layouts.layout import Layout


# %%
def evaluate_model(
    model,
    EXP_DIR,
    proj_matrix,
    DATA_INPUT="defects",
    FINAL_DATA_INPUT="defects",
    QEC_CYCLES=list(range(1, 40)),
    N_SHOTS=20_000,
    test_folder="test"
):
    LOG_FID = np.zeros(len(QEC_CYCLES))
    for r_idx, r in enumerate(QEC_CYCLES):
        print(r)
        test_generator = DataGeneratorGoogle(
                dirpath=EXP_DIR / test_folder,
                shots=N_SHOTS,
                rounds=[r],
                batch_size=N_SHOTS,
                lstm_input=DATA_INPUT,
                eval_input=FINAL_DATA_INPUT,
                proj_matrix=proj_matrix,
                folder_format_name="surface_code_bZ_d3_r{num_rounds:02d}_center_3_5",
                rot_basis=False,
        )
        assert len(test_generator) == 1
        test_input, test_output = test_generator[0]
        eval_output = model.predict(x=test_input)
        p1, p2 = eval_output
        out = p1 > 0.5
        out = out.flatten()
        error_prob = np.average(out ^ test_output)
        LOG_FID[r_idx] = 1 - error_prob

    fid_arr = xr.DataArray(
        LOG_FID,
        dims=["qec_round"],
        coords=dict(qec_round=list(QEC_CYCLES)),
    )
    log_fid = fid_arr.to_dataset(name="log_fid")

    return log_fid


# %%
SETUP = EXP_DIR / f"{MODEL_FOLDER}/config.yaml"
if not SETUP.exists():
    raise ValueError(f"Setup file does not exist: {SETUP}")
LAYOUT = EXP_DIR / "config" / "d3_rotated_layout.yaml"
if not LAYOUT.exists():
    raise ValueError(f"Layout does not exist: {LAYOUT}")

# %%
config = Config.from_yaml(SETUP, "/home", "/home")
layout = Layout.from_yaml(LAYOUT)
proj_matrix = layout.projection_matrix(stab_type="z_type")

# %%
# if results have not been stored, evaluate model
DIR = EXP_DIR / f"{MODEL_FOLDER}"
if not (DIR / "test_results_simulated.nc").exists():
    print("Evaluating model...")

    num_rounds = None
    num_anc = 8
    num_data = 4 if config.dataset["lstm_input"] == "defects" else 9
    model = get_model(
        config=config,
        metrics={},
        seq_size = num_anc,
        vec_size = num_data
    )
    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, EXP_DIR, proj_matrix=proj_matrix,
                            QEC_CYCLES=list(range(1,25+1,2)),
                            test_folder="test")
    log_fid.to_netcdf(path=DIR / "test_results_simulated.nc")

log_fid = xr.load_dataset(DIR / "test_results_simulated.nc")

# %%
# google's data (simulated)
MWPM_log_fid = np.array([0.982 , 0.9215, 0.865 , 0.792 , 0.78  , 0.729 , 0.6835, 0.664 ,
       0.648 , 0.631 , 0.602 , 0.5985, 0.5995])
MWPM_qec_round = np.arange(1, 25+1, 2)

# %%
model_decay = LogicalFidelityDecay()
params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
out = model_decay.fit(
    log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
)
error_rate = lmfit_par_to_ufloat(out.params["error_rate"])

MAX_QEC = min(len(log_fid.log_fid), len(MWPM_log_fid))

ax = out.plot_fit()
ax.plot(
    log_fid.qec_round.values[:MAX_QEC],
    log_fid.log_fid.values[:MAX_QEC],
    "b.",
    markersize=10,
)
ax.plot(
    MWPM_qec_round[:MAX_QEC], MWPM_log_fid[:MAX_QEC], "r.", markersize=10, label="MWPM"
)
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xticks(log_fid.qec_round.values[::2], log_fid.qec_round.values[::2])
ax.set_yticks(np.arange(0.5, 1, 0.05), np.round(np.arange(0.5, 1, 0.05), decimals=2))
ax.set_xlim(0, MWPM_qec_round[MAX_QEC-1] + 0.5)
ax.plot([], [], " ", label=f"$\\epsilon_L = {error_rate.nominal_value:.4f}$")
ax.legend()
ax.grid(which="major")
ax.set_title("Simulated data")

# %% [markdown]
# ## 2) Test experimental data

# %%
# if results have not been stored, evaluate model
DIR = EXP_DIR / f"{MODEL_FOLDER}"
if not (DIR / "test_results_experimental.nc").exists():
    print("Evaluating model...")

    num_rounds = None
    num_anc = 8
    num_data = 4 if config.dataset["lstm_input"] == "defects" else 9
    model = get_model(
        config=config,
        metrics={},
        seq_size = num_anc,
        vec_size = num_data
    )
    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, EXP_DIR, proj_matrix=proj_matrix,
                            QEC_CYCLES=list(range(1,25+1,2)),
                            test_folder="test_experimental")
    log_fid.to_netcdf(path=DIR / "test_results_experimental.nc")

log_fid = xr.load_dataset(DIR / "test_results_experimental.nc")

# %%
# google's data
MWPM_log_fid = np.array([0.98362, 0.90834, 0.84856, 0.78104, 0.7425 , 0.70236, 0.67078,
       0.64652, 0.61526, 0.60846, 0.58354, 0.57838, 0.56722])
MWPM_qec_round = np.arange(1, 25+1, 2)

# %%
model_decay = LogicalFidelityDecay()
params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
out = model_decay.fit(
    log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
)
error_rate = lmfit_par_to_ufloat(out.params["error_rate"])

MAX_QEC = min(len(log_fid.log_fid), len(MWPM_log_fid))

ax = out.plot_fit()
ax.plot(
    log_fid.qec_round.values[:MAX_QEC],
    log_fid.log_fid.values[:MAX_QEC],
    "b.",
    markersize=10,
)
ax.plot(
    MWPM_qec_round[:MAX_QEC], MWPM_log_fid[:MAX_QEC], "r.", markersize=10, label="MWPM"
)
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xticks(log_fid.qec_round.values[::2], log_fid.qec_round.values[::2])
ax.set_yticks(np.arange(0.5, 1, 0.05), np.round(np.arange(0.5, 1, 0.05), decimals=2))
ax.set_xlim(0, MWPM_qec_round[MAX_QEC-1] + 0.5)
ax.plot([], [], " ", label=f"$\\epsilon_L = {error_rate.nominal_value:.4f}$")
ax.legend()
ax.grid(which="major")
ax.set_title("Experimental data")

# %%
