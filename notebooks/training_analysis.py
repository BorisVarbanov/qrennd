# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230428-d3_xzzx-google_no-assign"
MODEL_FOLDER = "20230502-091707_Boris_config_assign-0-010_digitized"
LAYOUT_NAME = "d3_rotated_layout.yaml"
TEST_DATASET = ["test"]

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
from itertools import product

import xarray as xr

from qrennd import Config, Layout, get_model, load_datasets


# %%
def evaluate_model(model, config, layout, dataset_name="test"):
    test_data = load_datasets(
        config=config, layout=layout, dataset_name=dataset_name, concat=False
    )
    rounds = config.dataset[dataset_name]["rounds"]
    states = config.dataset[dataset_name]["states"]
    num_shots = config.dataset[dataset_name]["shots"]
    sequences = product(rounds, states)
    list_errors = []

    for data, k in zip(test_data, sequences):
        print(f"QEC = {k[0]} | state = {k[1]}", end="\r")
        prediction = model.predict(data, verbose=0)
        prediction = (prediction[0] > 0.5).flatten()
        errors = prediction != data.log_errors
        list_errors.append(errors)
        print(
            f"QEC = {k[0]} | state = {k[1]} | avg_errors = {np.average(errors):.4f}",
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
if not isinstance(TEST_DATASET, list):
    TEST_DATASET = [TEST_DATASET]

# %%
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
for test_dataset in TEST_DATASET:
    NAME = f"{test_dataset}.nc"
    if not (DIR / NAME).exists():
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
        log_fid = evaluate_model(model, config, layout, test_dataset)
        log_fid.to_netcdf(path=DIR / NAME)

        print("Done!")

    else:
        print("Model already evaluated!")

    print("\nRESULTS IN:")
    print("output_dir=", NOTEBOOK_DIR)
    print("exp_name=", EXP_NAME)
    print("run_name=", MODEL_FOLDER)
    print("test_data=", test_dataset)
