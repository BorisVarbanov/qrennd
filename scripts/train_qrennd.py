# %%
# Module import
import pathlib
from datetime import datetime

import tensorflow as tf
import xarray as xr

from qrennd import Config, Layout, get_model
from qrennd.callbacks.profilers import EpochRuntime
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


def generate_dataset(folder, num_shots, nums_rounds, data_input):
    log_states = [0, 1]

    exp_datasets = []
    for num_rounds in nums_rounds:
        datasets = []
        for log_state in log_states:
            exp_label = f"surf-code_d3_bZ_s{log_state}_n{num_shots}_r{num_rounds}"
            dataset = xr.load_dataset(folder / exp_label / "measurements.nc")
            datasets.append(dataset)

        exp_dataset = xr.concat(datasets, dim="log_state")
        exp_datasets.append(exp_dataset)

    dataset = xr.concat(exp_datasets, dim="num_rounds", fill_value=False)
    dataset = dataset.stack(run=["log_state", "shot", "num_rounds"])
    input, output = preprocess_data(dataset, data_input=data_input)
    return input, output


# %%
# Parameters
EXP_NAME = "20230131-d3_rot-surf_circ-level_large-dataset"

LAYOUT_FILE = "d3_rotated_layout.yaml"
CONFIG_FILE = "base_config.yaml"

DATA_INPUT = "defects"

NUM_TRAIN_SHOTS = 100000
NUM_TRAIN_ROUNDS = 20
TRAIN_ROUNDS = list(range(1, NUM_TRAIN_ROUNDS + 1, 2))

NUM_VAL_SHOTS = 10000
NUM_VAL_ROUNDS = 20
VAL_ROUNDS = list(range(1, NUM_VAL_ROUNDS + 1, 2))

BATCH_SIZE = 64
NUM_EPOCHS = 500
PATIENCE = 50
MIN_DELTA = 0
SAVE_BEST_ONLY = True

# %%
# Define used directories
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

USERNAME = "mserraperalta"
SCRATH_DIR = pathlib.Path(f"/scratch/{USERNAME}")
#SCRATH_DIR = NOTEBOOK_DIR

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
train_input, train_output = generate_dataset(
    folder=EXP_DIR / "train",
    num_shots=NUM_TRAIN_SHOTS,
    nums_rounds=TRAIN_ROUNDS,
    data_input=DATA_INPUT,
)

dev_input, dev_output = generate_dataset(
    folder=EXP_DIR / "dev",
    num_shots=NUM_VAL_SHOTS,
    nums_rounds=VAL_ROUNDS,
    data_input=DATA_INPUT,
)

# %%
defects_shape = train_input["defects"].shape
final_defects_shape = train_input["final_defects"].shape

model = get_model(
    defects_shape=defects_shape[1:],
    final_defects_shape=final_defects_shape[1:],
    config=config,
)


# %%
checkpoint_str = "weights.hdf5" if SAVE_BEST_ONLY else "weights-{epoch:02d}-{val_loss:.5f}.hdf5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_DIR / checkpoint_str,
    monitor="val_loss",
    mode="min",
    save_best_only=SAVE_BEST_ONLY
)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    min_delta=MIN_DELTA,
    patience=PATIENCE,
)
csv_logs = tf.keras.callbacks.CSVLogger(
    filename=LOG_DIR / "training.log",
    append=False
)
epoch_runtime = EpochRuntime()

callbacks = [
    model_checkpoint,
    # tensorboard,
    # early_stop,
    csv_logs,
    epoch_runtime
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
model.save(CHECKPOINT_DIR / "final_weights.hdf5")
