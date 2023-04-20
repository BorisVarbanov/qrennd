# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "20230403-d3_rot-css-surface_circ-level_p0-001"
MODEL_FOLDER = "20230419-224354_conv_lstm_first-try_k16-16"
LAYOUT_NAME = "d3_rotated_layout.yaml"
TEST_DATASET = "test"

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
import xarray as xr
import copy

from qrennd import Config, Layout, get_callbacks, get_model, load_datasets


# %%
def evaluate_model(model, config, layout, dataset_name="test"):
    callbacks = get_callbacks(config)
    outputs = {}
    for rounds in config.dataset[dataset_name]["rounds"]:
        print("QEC round = ", rounds, end="\r")
        config_ = copy.deepcopy(config)
        config_.dataset[dataset_name]["rounds"] = [rounds]
        config_.train["batch_size"] = 1_000
        test_data = load_datasets(
            config=config_, layout=layout, dataset_name=dataset_name, concat=False
        )

        correct = []
        for data in test_data:
            output = model.predict(
                data,
                verbose=0,
            )
            output = output[0] > 0.5
            log_errors = np.array(data.log_errors)
            correct.append(output.flatten() == log_errors.flatten())

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
# if results have not been stored, evaluate model
DIR = OUTPUT_DIR / EXP_NAME / MODEL_FOLDER
NAME = f"{TEST_DATASET}.nc"
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
    log_fid = evaluate_model(model, config, layout, TEST_DATASET)
    log_fid.to_netcdf(path=DIR / NAME)

    print("Done!")

else:
    print("Model already evaluated!")

print("\nRESULTS IN:")
print("output_dir=", NOTEBOOK_DIR)
print("exp_name=", EXP_NAME)
print("run_name=", MODEL_FOLDER)
