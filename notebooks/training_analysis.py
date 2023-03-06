# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230302-d3_rot-surf_simulated_google_20M"
MODEL_FOLDER = "20230305-112822_google_simulated_d3_20M_dr0-05"
LAYOUT_NAME = "d3_rotated_layout.yaml"

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
seq_size = len(layout.get_qubits(role="anc"))

if config.dataset["input"] == "measurements":
    vec_size = len(layout.get_qubits(role="data"))
else:
    vec_size = int(0.5 * seq_size)

model = get_model(
    seq_size=seq_size,
    vec_size=vec_size,
    config=config,
)

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
if not (DIR / "test_results.nc").exists():
    print("Evaluating model...")

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, "test")
    log_fid.to_netcdf(path=DIR / "test_results.nc")

log_fid = xr.load_dataset(DIR / "test_results.nc")

# %%
MPWM_log_fid = np.array(
    [
        0.96275,
        0.9199,
        0.88245,
        0.84325,
        0.81565,
        0.78715,
        0.7602,
        0.7285,
        0.70985,
        0.6915,
        0.6706,
        0.65755,
        0.64445,
        0.62755,
        0.62095,
        0.6042,
        0.5971,
        0.5835,
        0.5842,
        0.56485,
        0.5673,
        0.56225,
        0.5579,
        0.54265,
        0.53465,
        0.5373,
        0.542,
        0.5274,
        0.52815,
        0.533,
        0.52485,
        0.52125,
        0.5295,
        0.5174,
        0.5197,
        0.51355,
        0.51115,
        0.51,
        0.5112,
        0.5111,
        0.5084,
        0.51125,
        0.50955,
        0.50385,
        0.505,
        0.5107,
        0.50995,
        0.5012,
        0.5022,
        0.50445,
        0.50205,
        0.50035,
        0.50045,
        0.49915,
        0.5033,
        0.5029,
        0.50055,
        0.4986,
        0.50015,
    ]
)
MWPM_qec_round = np.arange(1, 60)

# %%
model_decay = LogicalFidelityDecay()
params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
out = model_decay.fit(
    log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
)
error_rate = lmfit_par_to_ufloat(out.params["error_rate"])

MAX_QEC = min(len(log_fid.log_fid), 60)

ax = out.plot_fit()
ax.plot(
    log_fid.qec_round.values[:MAX_QEC],
    log_fid.log_fid.values[:MAX_QEC],
    "b.",
    markersize=10,
)
ax.plot(
    MWPM_qec_round[:MAX_QEC], MPWM_log_fid[:MAX_QEC], "r.", markersize=10, label="MWPM"
)
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xticks(log_fid.qec_round.values[::2], log_fid.qec_round.values[::2])
ax.set_yticks(np.arange(0.5, 1, 0.05), np.round(np.arange(0.5, 1, 0.05), decimals=2))
ax.set_xlim(0, MAX_QEC + 0.5)
ax.plot([], [], " ", label=f"$\\epsilon_L = {error_rate.nominal_value:.4f}$")
ax.legend()
ax.grid(which="major")

# %%
