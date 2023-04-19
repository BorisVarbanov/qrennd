# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230418-d3_simulated_google_20M"
MODEL_FOLDER = "20230305-112822_google_simulated_d3_20M_dr0-05_center_3_5"
LAYOUT_NAME = "d3_rotated_layout.yaml"
ERRORBARS = False

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
acc_MWPM = None
goal = None

for metric in METRICS:
    fig, ax = plt.subplots()

    ax.plot(dataframe.epoch, dataframe[metric], ".-", color="blue", label="Training")
    ax.plot(
        dataframe.epoch,
        dataframe["val_" + metric],
        ".-",
        color="orange",
        label="Validation",
    )

    if metric == "main_output_accuracy":
        if acc_MWPM is not None:
            ax.axhline(y=acc_MWPM, linestyle="--", color="gray", label="MWPM (test)")
        if goal is not None:
            ax.axhline(y=goal, linestyle="--", color="black", label="goal")

    ax.legend(frameon=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric.replace("_", " ").capitalize())

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
            config=config_, layout=layout, dataset_name=dataset_name, concat=False
        )

        correct = []
        for data in test_data:
            inputs, log_errors = data[0]
            output = model.predict(
                data,
                verbose=0,
            )
            output = output[0] > 0.5
            correct.append(output.flatten() == log_errors)

        correct = np.array(correct).flatten()
        accuracy = np.average(correct)
        std = np.std(correct)
        outputs[rounds] = {"acc": accuracy, "std": std}

    accuracy = np.array([outputs[rounds]["acc"] for rounds in outputs])
    std = np.array([outputs[rounds]["std"] for rounds in outputs])
    qec_rounds = list(outputs.keys())

    log_fid = xr.Dataset(
        data_vars=dict(avg=(["qec_round"], accuracy), err=(["qec_round"], std)),
        coords=dict(qec_round=qec_rounds),
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
# get metadata
DISTANCE = layout.distance
CENTER = config.dataset["test"]["center"]
BASIS = "X" if config.dataset["rot_basis"] else "Z"

# %%
anc_qubits = layout.get_qubits(role="anc")
num_anc = len(anc_qubits)

if config.model["type"] in ("ConvLSTM", "Conv_LSTM"):
    rec_features = (layout.distance + 1, layout.distance + 1, 1)
else:
    rec_features = num_anc

if config.dataset["input"] == "measurements":
    data_qubits = layout.get_qubits(role="data")
    eval_features = len(data_qubits)
else:
    eval_features = int(num_anc / 2)


model = get_model(
    rec_features=rec_features,
    eval_features=eval_features,
    config=config,
)

# %% [markdown]
# ## 1) Test simulated data

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
filename = "test.nc"
if not (DIR / filename).exists():
    print("Evaluating model...")

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, "test")
    log_fid.to_netcdf(path=DIR / filename)

log_fid = xr.load_dataset(DIR / filename)

DIR = OUTPUT_DIR / EXP_NAME / "pymatching"
filename = f"test_b{BASIS}_d{DISTANCE}_center_{CENTER}.nc"
if not (DIR / filename).exists():
    print("Warning: Run MWPM_analysis.py to generate data")

pymatching_log_fid = xr.load_dataset(DIR / filename)


# %%
datasets = [log_fid, pymatching_log_fid]
colors = ["blue", "red"]
labels = ["NN", "MWPM"]

fig, ax = plt.subplots()

for dataset, color, label in zip(datasets, colors, labels):
    x, y = dataset.qec_round.values, dataset.avg.values
    yerr = dataset.err.values if ERRORBARS else 0
    ax.errorbar(
        x, y, yerr=yerr, fmt=".", color=color, markersize=10, capsize=2, label=label
    )

    model_decay = LogicalFidelityDecay(fixed_t0=False)
    params = model_decay.guess(y, x=x)
    out = model_decay.fit(y, params, x=x, min_qec=layout.distance)
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    t0 = lmfit_par_to_ufloat(out.params["t0"])

    x_fit = np.linspace(layout.distance, max(x), 100)
    y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)
    ax.plot(
        x_fit, y_fit, "-", color=color, label=f"$\\epsilon_L = (${error_rate*100})%"
    )

ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xlim(0, 25 + 1)
ax.set_ylim(0.5, 1)
ax.set_yticks(
    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
)
ax.legend(loc="best")
ax.grid(which="major")
fig = ax.get_figure()
fig.tight_layout()
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
fig.savefig(DIR / "log-fid_vs_qec-round_simulated.pdf", format="pdf")
fig.savefig(DIR / "log-fid_vs_qec-round_simulated.png", format="png")
plt.show()

# %% [markdown]
# ## 2) Test experimental data

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
filename = "test_exp.nc"
if not (DIR / filename).exists():
    print("Evaluating model...")

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, "test_experimental")
    log_fid.to_netcdf(path=DIR / filename)

log_fid = xr.load_dataset(DIR / filename)

DIR = OUTPUT_DIR / EXP_NAME / "pymatching"
filename = f"test_b{BASIS}_d{DISTANCE}_center_{CENTER}_exp.nc"
if not (DIR / filename).exists():
    print("Warning: Run MWPM_analysis.py to generate data")

pymatching_log_fid = xr.load_dataset(DIR / filename)

DIR = OUTPUT_DIR / EXP_NAME / "belief_matching"
filename = f"test_b{BASIS}_d{DISTANCE}_center_{CENTER}_exp.nc"
if not (DIR / filename).exists():
    print("Warning: Run MWPM_analysis.py to generate data")

bel_matching_log_fid = xr.load_dataset(DIR / filename)

DIR = OUTPUT_DIR / EXP_NAME / "correlated_matching"
filename = f"test_b{BASIS}_d{DISTANCE}_center_{CENTER}_exp.nc"
if not (DIR / filename).exists():
    print("Warning: Run MWPM_analysis.py to generate data")

corr_matching_log_fid = xr.load_dataset(DIR / filename)

DIR = OUTPUT_DIR / EXP_NAME / "tensor_network_contraction"
filename = f"test_b{BASIS}_d{DISTANCE}_center_{CENTER}_exp.nc"
if not (DIR / filename).exists():
    print("Warning: Run MWPM_analysis.py to generate data")

tensor_log_fid = xr.load_dataset(DIR / filename)

# %%
datasets = [
    log_fid,
    pymatching_log_fid,
    bel_matching_log_fid,
    corr_matching_log_fid,
    tensor_log_fid,
]
colors = ["blue", "red", "green", "orange", "purple"]
labels = ["NN", "MWPM", "BeliefM", "corrM", "tensor"]

fig, ax = plt.subplots()

for dataset, color, label in zip(datasets, colors, labels):
    x, y = dataset.qec_round.values, dataset.avg.values
    yerr = dataset.err.values if ERRORBARS else 0
    ax.errorbar(
        x, y, yerr=yerr, fmt=".", color=color, markersize=10, capsize=2, label=label
    )

    model_decay = LogicalFidelityDecay(fixed_t0=False)
    params = model_decay.guess(y, x=x)
    out = model_decay.fit(y, params, x=x, min_qec=layout.distance)
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    t0 = lmfit_par_to_ufloat(out.params["t0"])

    x_fit = np.linspace(layout.distance, max(x), 100)
    y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)
    ax.plot(
        x_fit, y_fit, "-", color=color, label=f"$\\epsilon_L = (${error_rate*100})%"
    )

ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xlim(0, 25 + 1)
ax.set_ylim(0.5, 1)
ax.set_yticks(
    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
)
ax.legend(loc="best")
ax.grid(which="major")
fig = ax.get_figure()
fig.tight_layout()
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
fig.savefig(DIR / "log-fid_vs_qec-round_experimental.pdf", format="pdf")
fig.savefig(DIR / "log-fid_vs_qec-round_experimental.png", format="png")
plt.show()
