# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230310-d3_rot-surf_circ-level_meas-reset"
MODEL_FOLDER = "20230415-175326_lstm_similar-config-lstm32"
LAYOUT_NAME = "d3_rotated_layout.yaml"
FIXED_TO = False

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
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
if not (DIR / "test_results.nc").exists():
    print("Evaluating model...")

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

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, "test")
    log_fid.to_netcdf(path=DIR / "test_results.nc")

log_fid = xr.load_dataset(DIR / "test_results.nc")
NN_log_fid = log_fid.log_fid.values
NN_qec_round = log_fid.qec_round.values

# %%
MWPM_data = OUTPUT_DIR / EXP_NAME / "MWPM" / "test_results.nc"
if not MWPM_data.exists():
    print(
        (
            f"File not found: {MWPM_data}\n"
            "Run 'MWPM_analysis.py' to generate MWPM data"
        )
    )
    MWPM_data = False
    MAX_QEC = np.max(log_fid.qec_round.values)
else:
    MWPM_data = xr.load_dataset(MWPM_data)
    MPWM_log_fid = MWPM_data.log_fid.values
    MWPM_qec_round = MWPM_data.qec_round.values
    MAX_QEC = int(min(np.max(log_fid.qec_round.values), np.max(MWPM_qec_round)))

# %%
fig, ax = plt.subplots()

if MWPM_data:
    x = MWPM_qec_round
    y = MPWM_log_fid
    ax.plot(x, y, "b.", markersize=10, label="MWPM")

    for FIXED_TO, fmt in zip([True, False], ["b--", "b-"]):
        model_decay = LogicalFidelityDecay(fixed_t0=FIXED_TO)
        params = model_decay.guess(y, x=x)
        out = model_decay.fit(y, params, x=x, min_qec=layout.distance)
        error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
        t0 = lmfit_par_to_ufloat(out.params["t0"])
        x_fit = np.linspace(layout.distance, max(x), 100)
        y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)

        label = f"$\\epsilon_L = {error_rate.nominal_value:.5f}$\n$t_0 = {t0.nominal_value:.4f}$"
        ax.plot(x_fit, y_fit, fmt, label=label)

x = NN_qec_round
y = NN_log_fid
ax.plot(x, y, "r.", markersize=10, label="NN")

for FIXED_TO, fmt in zip([True, False], ["r--", "r-"]):
    model_decay = LogicalFidelityDecay(fixed_t0=FIXED_TO)
    params = model_decay.guess(y, x=x)
    out = model_decay.fit(y, params, x=x, min_qec=layout.distance)
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    t0 = lmfit_par_to_ufloat(out.params["t0"])
    x_fit = np.linspace(layout.distance, max(x), 100)
    y_fit = model_decay.func(x_fit, error_rate.nominal_value, t0.nominal_value)

    label = f"$\\epsilon_L = {error_rate.nominal_value:.5f}$\n$t_0 = {t0.nominal_value:.4f}$"
    ax.plot(x_fit, y_fit, fmt, label=label)

ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xlim(0, MAX_QEC + 1)
ax.set_ylim(0.5, 1)
ax.set_yticks(
    np.arange(0.5, 1.01, 0.05), np.round(np.arange(0.5, 1.01, 0.05), decimals=2)
)
ax.legend()
ax.grid(which="major")
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(DIR / "log-fid_vs_qec-round.pdf", format="pdf")
fig.savefig(DIR / "log-fid_vs_qec-round.png", format="png")
plt.show()

# %%
