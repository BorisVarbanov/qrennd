# %%
# Module import
import pathlib
from datetime import datetime

import tensorflow as tf

from qrennd import Config, Layout, get_model
from qrennd.utils import DataGenerator

# %%
# Parameters
EXP_NAME = "20230131-d3_rot-surf_circ-level_large-dataset"

LAYOUT_FILE = "d3_rotated_layout.yaml"
CONFIG_FILE = "base_config.yaml"

DATA_INPUT = "defects"
LOG_STATES = [0, 1]

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
# SCRATH_DIR = NOTEBOOK_DIR

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
train_generator = DataGenerator(
    folder=EXP_DIR / "train",
    num_shots=NUM_TRAIN_SHOTS,
    states=LOG_STATES,
    qec_rounds=TRAIN_ROUNDS,
    batch_size=BATCH_SIZE,
)

val_generator = DataGenerator(
    folder=EXP_DIR / "dev",
    num_shots=NUM_VAL_SHOTS,
    states=LOG_STATES,
    qec_rounds=VAL_ROUNDS,
    batch_size=BATCH_SIZE,
)

# %%
num_anc = len(layout.get_qubits(role="anc"))
num_data = len(layout.get_qubits(role="data"))

model = get_model(
    defects_shape=(None, num_anc),
    final_defects_shape=(num_data,),
    config=config,
)


# %%
checkpoint_str = (
    "weights.hdf5" if SAVE_BEST_ONLY else "weights-{epoch:02d}-{val_loss:.5f}.hdf5"
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_DIR / checkpoint_str,
    monitor="val_loss",
    mode="min",
    save_best_only=SAVE_BEST_ONLY,
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    min_delta=MIN_DELTA,
    patience=PATIENCE,
)
csv_logs = tf.keras.callbacks.CSVLogger(filename=LOG_DIR / "training.log", append=False)

callbacks = [
    model_checkpoint,
    early_stop,
    csv_logs,
]


# %%
history = model.fit(
    train_generator,
    validation_data=val_generator,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
)

# %%
model.save(CHECKPOINT_DIR / "final_weights.hdf5")
