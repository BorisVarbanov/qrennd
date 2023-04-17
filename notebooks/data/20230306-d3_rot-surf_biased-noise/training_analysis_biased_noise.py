# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230306-d3_rot-surf_biased-noise"
MODEL_FOLDER = "20230211-094701-base_training_4M_dr-eval-lstm2-01"
LAYOUT_NAME = "d3_rotated_layout.yaml"
BIAS_FACTORS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

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
ERROR_RATES_NN = {}
for bias_factor in BIAS_FACTORS:
    bias_factor_str = f"{int(bias_factor*1000)}".zfill(4)

    # if results have not been stored, evaluate model
    DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
    if not (DIR / f"test_results_{bias_factor_str}.nc").exists():
        print("Evaluating model...")

        model.load_weights(DIR / "checkpoint/weights.hdf5")
        log_fid = evaluate_model(model, config, layout, f"test_{bias_factor_str}")
        log_fid.to_netcdf(path=DIR / f"test_results_{bias_factor_str}.nc")

    log_fid = xr.load_dataset(DIR / f"test_results_{bias_factor_str}.nc")

    model_decay = LogicalFidelityDecay()
    params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
    out = model_decay.fit(
        log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
    )
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    ERROR_RATES_NN[bias_factor_str] = error_rate

# %%
# LOAD MWPM ERROR RATE
ERROR_RATES_MWPM = {}
for bias_factor in BIAS_FACTORS:
    bias_factor_str = f"{int(bias_factor*1000)}".zfill(4)

    # if results have not been stored, evaluate model
    DIR = OUTPUT_DIR / EXP_NAME / "MWPM"
    if not (DIR / f"test_results_{bias_factor_str}.nc").exists():
        raise ValueError("Missing the MWPM error rates!")

    log_fid = xr.load_dataset(DIR / f"test_results_{bias_factor_str}.nc")

    model_decay = LogicalFidelityDecay()
    params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
    out = model_decay.fit(
        log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3
    )
    error_rate = lmfit_par_to_ufloat(out.params["error_rate"])
    ERROR_RATES_MWPM[bias_factor_str] = error_rate


# %%
BIAS_FACTORS_KEYS = sorted(ERROR_RATES_MWPM.keys())
ERROR_RATES_MWPM = np.array(
    [ERROR_RATES_MWPM[k].nominal_value for k in BIAS_FACTORS_KEYS]
)
ERROR_RATES_NN = np.array([ERROR_RATES_NN[k].nominal_value for k in BIAS_FACTORS_KEYS])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(BIAS_FACTORS, ERROR_RATES_MWPM, "b.", label="MWPM")
ax.plot(BIAS_FACTORS, ERROR_RATES_NN, "r.", label="neural network")
ax.set_xlabel("Bias factor")
ax.set_ylabel("Logical error rate per cycle")
ax.legend()
ax.grid(which="major")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / "eps-L_vs_bias-factor.pdf", format="pdf"
)
fig.savefig(
    OUTPUT_DIR / EXP_NAME / MODEL_FOLDER / "eps-L_vs_bias-factor.png", format="png"
)

plt.show()

# %%
