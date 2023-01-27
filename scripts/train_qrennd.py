# %%
# Module import
import pathlib
from datetime import datetime

import tensorflow as tf
import xarray as xr

from qrennd import Config, Layout, get_model
from qrennd.utils.data_processing import get_defects, get_syndromes


# Define data preprocessing
def preprocess_data(dataset, data_input="defects"):
    anc_meas = dataset.anc_meas.transpose("run", "qec_round", "anc_qubit")
    data_meas = dataset.data_meas.transpose("run", "data_qubit")

    log_meas = dataset.data_meas.sum(dim="data_qubit") % 2
    log_errors = log_meas ^ dataset.log_state

    inputs = dict(final_defects=data_meas.values)

    if data_input == "measurements":
        inputs["defects"] = anc_meas.data

    elif data_input == "syndromes":
        syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)
        inputs["defects"] = syndromes.values

    elif data_input == "defects":
        syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)
        defects = get_defects(syndromes)
        inputs["defects"] = defects.values
    else:
        raise ValueError(
            "'data_input' type must be: 'defects', 'syndromes' or 'measurements'"
        )

    outputs = log_errors.values
    return inputs, outputs


# %%
# Parameters
EXP_NAME = "20230117-d3_rot-surf_circ-level_test-train"

LAYOUT_FILE = "d3_rotated_layout.yaml"
CONFIG_FILE = "base_config.yaml"

DATA_INPUT = "defects"

NUM_TRAIN_SHOTS = 10000
NUM_TRAIN_ROUNDS = 20

NUM_DEV_SHOTS = 1000
NUM_DEV_ROUNDS = 20

LOG_STATES = range(2)

BATCH_SIZE = 64
NUM_EPOCHS = 500
PATIENCE = 20
MIN_DELTA = 0

# %%
# Define used directories
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

USERNAME = "bmvarbanov"
SCRATH_DIR = pathlib.Path(f"/scratch/{USERNAME}")
SCRATH_DIR = NOTEBOOK_DIR

LAYOUT_DIR = NOTEBOOK_DIR / "layouts"
if not LAYOUT_DIR.exists():
    raise ValueError("Layout directory does not exist.")

CONFIG_DIR = NOTEBOOK_DIR / "configs"
if not CONFIG_DIR.exists():
    raise ValueError("Config directory does not exist.")

# The train/dev/test data directories are located in the local data directory
# experiment folder
EXP_DIR = SCRATH_DIR / "data" / EXP_NAME
if not EXP_DIR.exists():
    raise ValueError("Experiment directory does not exist.")

cur_datetime = datetime.now()
datetime_str = cur_datetime.strftime("%Y%m%d-%H%M%S")

OUTPUT_DIR = SCRATH_DIR / "output" / EXP_NAME

LOG_DIR = OUTPUT_DIR / f"logs/{datetime_str}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = OUTPUT_DIR / "tmp/checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Load setup objects
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)
config = Config.from_yaml(CONFIG_DIR / CONFIG_FILE)

# %%
datasets = []
for log_state in LOG_STATES:
    exp_label = f"surf-code_d3_bZ_s{log_state}_n{NUM_TRAIN_SHOTS}_r{NUM_TRAIN_ROUNDS}"
    dataset = xr.load_dataset(EXP_DIR / "train" / exp_label / "measurements.nc")
    datasets.append(dataset)

dataset = xr.concat(datasets, dim="log_state")
dataset = dataset.stack(run=["log_state", "shot"])

train_input, train_output = preprocess_data(dataset, data_input=DATA_INPUT)

datasets = []
for log_state in LOG_STATES:
    exp_label = f"surf-code_d3_bZ_s{log_state}_n{NUM_DEV_SHOTS}_r{NUM_DEV_ROUNDS}"
    dataset = xr.load_dataset(EXP_DIR / "dev" / exp_label / "measurements.nc")
    datasets.append(dataset)

dataset = xr.concat(datasets, dim="log_state")
dataset = dataset.stack(run=["log_state", "shot"])

dev_input, dev_output = preprocess_data(dataset, data_input=DATA_INPUT)


# %%
defects_shape = train_input["defects"].shape
final_defects_shape = train_input["final_defects"].shape

model = get_model(
    defects_shape=defects_shape[1:],
    final_defects_shape=final_defects_shape[1:],
    config=config,
)


# %%
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR / "weights-{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor="val_loss",
        mode="min",
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=MIN_DELTA,
        patience=PATIENCE,
    ),
    tf.keras.callbacks.CSVLogger(
        filename=LOG_DIR / "training.log",
        append=False,
    ),
]


# %%
history = model.fit(
    x=train_input,
    y=train_output,
    validation_data=(dev_input, dev_output),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
)

# %%
model.save(CHECKPOINT_DIR / "weights.hdf5")