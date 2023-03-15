# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230302-d5_rot-surf_simulated_google_20M"
MODEL_FOLDER = "20230314-113238_google_simulated_dr0-05_batch64"
LAYOUT_NAME = "d5_rotated_layout.yaml"

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

DATA_DIR = NOTEBOOK_DIR / "data"
if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")

OUTPUT_DIR = NOTEBOOK_DIR / "output"
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

LOG_FILE = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / "logs/training.log"
if not LOG_FILE.exists():
    raise ValueError(f"Log file does not exist: {LOG_FILE}")

CONFIG_FILE = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / "config.yaml"
if not CONFIG_FILE.exists():
    raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

LAYOUT_FILE = DATA_DIR / EXP_NAME / f"config/{LAYOUT_NAME}"
if not LAYOUT_FILE.exists():
    raise ValueError(f"Layout file does not exist: {LAYOUT_FILE}")

# %%
dataframe = pd.read_csv(LOG_FILE)

# %%
dataframe

# %%
METRICS = ("loss", "main_output_accuracy")
EPOCH_CUT = 50
acc_MWPM = 0.72482
goal = 0.764  # same increase in performance as O'Brien paper

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
        axs[0].axhline(y=goal, linestyle="--", color="black", label="goal")

    axs[0].legend(frameon=False)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel(metric.replace("_", " ").capitalize())
    axs[1].legend(frameon=False)
    axs[1].set_xlabel("Epochs")
    axs[1].set_xlim(EPOCH_CUT, max(dataframe.epoch) + 1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / f"{metric}.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / f"{metric}.png", format="png")

plt.show()

# %% [markdown]
# # Evaluation

# %%
import xarray as xr
import copy
from qrennd import get_model, Config
from qrennd.utils.analysis import (
    logical_fidelity,
    LogicalFidelityDecay,
    lmfit_par_to_ufloat,
)
from qrennd import Config, Layout, get_callbacks, get_model, load_datasets


# %%
def evaluate_model(model, config, layout, dataset_name="test"):
    callbacks = get_callbacks(config)
    outputs = {}
    for rounds in config.dataset[dataset_name]["rounds"]:
        print("QEC round = ", rounds, end="\r")
        config_ = copy.deepcopy(config)
        config_.dataset[dataset_name]["rounds"] = [rounds]
        config_.train["batch_size"] = config_.dataset[dataset_name]["shots"]
        test_data = load_datasets(
            config=config_, layout=layout, dataset_name=dataset_name
        )

        output = model.evaluate(
            test_data,
            callbacks=callbacks,
            verbose=0,
            return_dict=True,
        )
        outputs[rounds] = output

    # convert to xr.DataArray
    rounds, log_fid = np.array(
        [
            [rounds, metrics["main_output_accuracy"]]
            for rounds, metrics in outputs.items()
        ]
    ).T

    log_fid = xr.DataArray(data=log_fid, coords=dict(qec_round=rounds), name="log_fid")

    return log_fid


# %%
layout = Layout.from_yaml(LAYOUT_FILE)
config = Config.from_yaml(
    filepath=CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)

# %%
if config.model["use_conv"]:
    seq_size = (1, layout.distance + 1, layout.distance + 1)
else:
    seq_size = (len(layout.get_qubits(role="anc")),)

if config.dataset["input"] == "measurements":
    vec_size = len(layout.get_qubits(role="data"))
else:
    vec_size = len(layout.get_qubits(role="anc")) // 2

model = get_model(
    seq_size=seq_size,
    vec_size=vec_size,
    config=config,
)

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
if not (DIR / "test_results_simulated.nc").exists():
    print("Evaluating model...")

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, "test")
    log_fid.to_netcdf(path=DIR / "test_results_simulated.nc")

log_fid = xr.load_dataset(DIR / "test_results_simulated.nc")

# %%
# google's data (simulated)
if "d3" in LAYOUT_NAME:
    MWPM_log_fid = np.array(
        [
            0.982,
            0.9215,
            0.865,
            0.792,
            0.78,
            0.729,
            0.6835,
            0.664,
            0.648,
            0.631,
            0.602,
            0.5985,
            0.5995,
        ]
    )
elif "d5" in LAYOUT_NAME:
    MWPM_log_fid = np.array(
        [
            0.9921,
            0.942,
            0.8902,
            0.84156,
            0.80066,
            0.76196,
            0.73308,
            0.70262,
            0.67826,
            0.65834,
            0.63502,
            0.62378,
            0.60516,
        ]
    )
MWPM_qec_round = np.arange(1, 25 + 1, 2)

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
ax.set_xlim(0, MWPM_qec_round[MAX_QEC - 1] + 0.5)
ax.plot([], [], " ", label=f"$\\epsilon_L = {error_rate.nominal_value:.4f}$")
ax.legend()
ax.grid(which="major")
ax.set_title("Simulated data")
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(DIR / "log-fid_vs_qec-round_simulated.pdf", format="pdf")
fig.savefig(DIR / "log-fid_vs_qec-round_simulated.png", format="png")
plt.show()

# %% [markdown]
# ## 2) Test experimental data

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
if not (DIR / "test_results_experimental.nc").exists():
    print("Evaluating model...")

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, "test_experimental")
    log_fid.to_netcdf(path=DIR / "test_results_experimental.nc")

log_fid = xr.load_dataset(DIR / "test_results_experimental.nc")

# %%
if "d3" in LAYOUT_NAME:
    # google's data
    MWPM_log_fid = np.array(
        [
            0.98362,
            0.90834,
            0.84856,
            0.78104,
            0.7425,
            0.70236,
            0.67078,
            0.64652,
            0.61526,
            0.60846,
            0.58354,
            0.57838,
            0.56722,
        ]
    )
    CORR_log_fid = np.array(
        [
            0.98362,
            0.91344,
            0.85956,
            0.80068,
            0.76252,
            0.7271,
            0.69716,
            0.6677,
            0.6393,
            0.62876,
            0.6065,
            0.59728,
            0.57862,
        ]
    )
    BELIEF_log_fid = np.array(
        [
            0.98358,
            0.92146,
            0.87334,
            0.81486,
            0.78084,
            0.74368,
            0.71576,
            0.68738,
            0.65658,
            0.6465,
            0.6217,
            0.61562,
            0.59626,
        ]
    )
    TENSOR_log_fid = np.array(
        [
            0.98362,
            0.92314,
            0.87512,
            0.82052,
            0.78392,
            0.74664,
            0.71704,
            0.68892,
            0.66008,
            0.64562,
            0.62196,
            0.61106,
            0.59156,
        ]
    )
elif "d5" in LAYOUT_NAME:
    MWPM_log_fid = np.array(
        [
            0.99184,
            0.92712,
            0.85624,
            0.80086,
            0.75568,
            0.7117,
            0.67334,
            0.64378,
            0.621,
            0.6028,
            0.58466,
            0.57016,
            0.56142,
        ]
    )
    CORR_log_fid = np.array(
        [
            0.99184,
            0.93482,
            0.87484,
            0.82576,
            0.78732,
            0.7474,
            0.71138,
            0.68194,
            0.65242,
            0.63664,
            0.61714,
            0.60186,
            0.58976,
        ]
    )
    BELIEF_log_fid = np.array(
        [
            0.99202,
            0.9444,
            0.89276,
            0.84616,
            0.81528,
            0.77564,
            0.74194,
            0.71532,
            0.68796,
            0.66528,
            0.64676,
            0.63186,
            0.6155,
        ]
    )
    TENSOR_log_fid = np.array(
        [
            0.9923,
            0.94722,
            0.89784,
            0.8562,
            0.82162,
            0.78272,
            0.75284,
            0.72576,
            0.69972,
            0.67428,
            0.65354,
            0.63902,
            0.62306,
        ]
    )
dec_qec_round = np.arange(1, 25 + 1, 2)

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
    dec_qec_round[:MAX_QEC], MWPM_log_fid[:MAX_QEC], "r.", markersize=10, label="MWPM"
)
ax.plot(
    dec_qec_round[:MAX_QEC], CORR_log_fid[:MAX_QEC], "c.", markersize=10, label="corr"
)
ax.plot(
    dec_qec_round[:MAX_QEC],
    BELIEF_log_fid[:MAX_QEC],
    "g.",
    markersize=10,
    label="belief",
)
ax.plot(
    dec_qec_round[:MAX_QEC],
    TENSOR_log_fid[:MAX_QEC],
    "m.",
    markersize=10,
    label="tensor",
)
ax.plot(
    log_fid.qec_round.values[:MAX_QEC],
    log_fid.log_fid.values[:MAX_QEC],
    "b.",
    markersize=10,
)
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xticks(log_fid.qec_round.values[::2], log_fid.qec_round.values[::2])
ax.set_yticks(np.arange(0.5, 1, 0.05), np.round(np.arange(0.5, 1, 0.05), decimals=2))
ax.set_xlim(0, MWPM_qec_round[MAX_QEC - 1] + 0.5)
ax.plot([], [], " ", label=f"$\\epsilon_L = {error_rate.nominal_value:.4f}$")
ax.legend()
ax.grid(which="major")
ax.set_title("Experimental data")
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(DIR / "log-fid_vs_qec-round_experimental.pdf", format="pdf")
fig.savefig(DIR / "log-fid_vs_qec-round_experimental.png", format="png")
plt.show()

# %%
