# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230306-d3_rot-surf_biased-noise"
MODEL_FOLDER = "20230211-094701-base_training_4M_dr-eval-lstm2-01"
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
MWPM_data = OUTPUT_DIR / EXP_NAME / "MWPM" / "test_results.nc"
if not MWPM_data.exists():
    raise TypeError(
        (
            f"File not found: {MWPM_data}\n"
            "Run 'MWPM_analysis.py' to generate MWPM data"
        )
    )

MWPM_data = xr.load_dataset(MWPM_data)
MPWM_log_fid = MWPM_data.log_fid.values
MWPM_qec_round = MWPM_data.qec_round.values

# %%
model_decay = LogicalFidelityDecay()
params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
out = model_decay.fit(
    log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
)
error_rate = lmfit_par_to_ufloat(out.params["error_rate"])

MAX_QEC = int(min(np.max(log_fid.qec_round.values), np.max(MWPM_qec_round)))

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
plt.show()

# %%
