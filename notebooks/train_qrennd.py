# %%
# Module import
import pathlib
from datetime import datetime

import xarray as xr
from qec_util import Layout
from qec_util.util.syndrome import (get_defects, get_final_defects,
                                    get_syndromes)
from tensorflow import keras

from qrennd import Config, get_model


# Define data preprocessing
def preprocess_data(dataset, proj_mat):
    anc_meas = dataset.anc_meas.stack(state_shot=("log_state", "shot"))
    anc_meas = anc_meas.transpose("state_shot", "qec_round", "anc_qubit")

    syndromes = get_syndromes(anc_meas)
    defects = get_defects(syndromes)

    data_meas = dataset.data_meas.stack(state_shot=("log_state", "shot"))
    data_meas = data_meas.transpose("state_shot", "data_qubit")

    proj_syndrome = (data_meas @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    log_meas = dataset.data_meas.sum(dim="data_qubit") % 2
    log_inits = dataset.init_state.sum(dim="data_qubit") % 2
    log_errors = log_meas ^ log_inits
    log_errors = log_errors.stack(state_shot=("log_state", "shot"))

    # inputs = dict(defects=defects.data, final_defects=final_defects.data)
    # inputs = dict(defects=syndromes.data, final_defects=data_meas.data)
    inputs = dict(defects=dataset.anc_meas.data, final_defects=dataset.data_meas.data)
    outputs = log_errors.data

    return inputs, outputs


# %%
# Define used directories
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

LAYOUT_DIR = NOTEBOOK_DIR / "layouts"
if not LAYOUT_DIR.exists():
    raise ValueError("Layout directory does not exist.")

CONFIG_DIR = NOTEBOOK_DIR / "configs"
if not CONFIG_DIR.exists():
    raise ValueError("Config directory does not exist.")

# The train/dev/test data directories are located in the local data directory
DATA_DIR = NOTEBOOK_DIR / "data"
if not DATA_DIR.exists():
    raise ValueError("Train data directory does not exist.")

cur_datetime = datetime.now()
datetime_str = cur_datetime.strftime("%Y%m%d-%H%M%S")

LOG_DIR = NOTEBOOK_DIR / f"logs/{datetime_str}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = NOTEBOOK_DIR / "tmp/checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Parameters
LAYOUT_FILE = "d3_rotated_layout.yaml"
CONFIG_FILE = "base_config.yaml"

TRAIN_DATASET_FILE = "d3_surf_circ_noise_shots_10000_rounds_60.nc"
DEV_DATASET_FILE = "d3_surf_circ_noise_shots_10000_rounds_60.nc"
BATCH_SIZE = 64
NUM_EPOCHS = 500
PATIENCE = 20
MIN_DELTA = 0

# %%
# Load setup objects
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)
config = Config.from_yaml(CONFIG_DIR / CONFIG_FILE)

# %%

train_dataset = xr.load_dataset(DATA_DIR / "train" / TRAIN_DATASET_FILE)
proj_mat = layout.projection_matrix(stab_type="z_type")
train_input, train_output = preprocess_data(train_dataset, proj_mat)

dev_dataset = xr.load_dataset(DATA_DIR / "dev" / DEV_DATASET_FILE)
proj_mat = layout.projection_matrix(stab_type="z_type")
dev_input, dev_output = preprocess_data(dev_dataset, proj_mat)


# %%
num_rounds = train_dataset.qec_round.size
num_anc = train_dataset.anc_qubit.size
num_data = train_dataset.data_qubit.size

model = get_model(
    defects_shape=(num_rounds, num_anc),
    final_defects_shape=(num_anc,),
    config=config,
)


# %%
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR / "weights.hdf5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
    keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=MIN_DELTA,
        patience=PATIENCE,
    ),
]


# %%
history = model.fit(
    x=train_input,
    y=train_output,
    validation_data=[dev_input, dev_output],
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
)
