# %%
import pathlib
from datetime import datetime
from typing import Optional

import numpy as np
import xarray as xr
import keras

from qrennd import get_model, Config, Layout

# %%
PAR_MAT = xr.DataArray(
    data=[
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
    ],
    dims=["anc_qubit", "data_qubit"],
    coords=dict(
        anc_qubit=["Z1", "Z2", "Z3", "Z4"],
        data_qubit=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
    ),
)


# %%
%load_ext tensorboard

# %%
def get_syndromes(anc_meas: xr.DataArray) -> xr.DataArray:
    syndromes = anc_meas ^ anc_meas.shift(qec_round=1, fill_value=0)
    syndromes.name = "syndromes"
    return syndromes


def get_defects(
    syndromes: xr.DataArray, frame: Optional[xr.DataArray] = None
) -> xr.DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    defects.name = "defects"
    return defects


def get_final_defects(
    syndromes: xr.DataArray,
    proj_syndrome: xr.DataArray,
) -> xr.DataArray:
    last_syndrome = syndromes.isel(qec_round=-1)
    proj_anc = proj_syndrome.anc_qubit

    final_defects = last_syndrome.sel(anc_qubit=proj_anc) ^ proj_syndrome
    final_defects.name = "final_defects"
    return final_defects


def preprocess_data(dataset):
    syndromes = get_syndromes(dataset.anc_meas)
    defects = get_defects(syndromes)

    proj_syndrome = (dataset.data_meas @ PAR_MAT) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    init_states = dataset.init_state.sum(dim="data_qubit") % 2
    log_states = dataset.data_meas.sum(dim="data_qubit") % 2

    labels = log_states.astype(int) ^ init_states

    inputs = dict(defects=defects.data, final_defects=final_defects.data)
    outputs = labels.data

    return inputs, outputs

# %% [markdown]
# # Load the datasets

# %%
NOTEBOOK_DIR = pathlib.Path.cwd() # define the path where the notebook is placed.

LAYOUT_DIR = NOTEBOOK_DIR / "layouts"
if not LAYOUT_DIR.exists():
    raise ValueError("Layout directory does not exist.")

CONFIG_DIR = NOTEBOOK_DIR / "configs"
if not CONFIG_DIR.exists():
    raise ValueError("Config directory does not exist.")

# The train/dev/test data directories are located in the local data directory
TRAIN_DATA_DIR = NOTEBOOK_DIR / "data/train"
if not TRAIN_DATA_DIR.exists():
    raise ValueError("Train data directory does not exist.")

DEV_DATA_DIR = NOTEBOOK_DIR / "data/dev"
if not DEV_DATA_DIR.exists():
    raise ValueError("Dev data directory does not exist.")

TEST_DATA_DIR = NOTEBOOK_DIR / "data/test"
if not TEST_DATA_DIR.exists():
    raise ValueError("Test data directory does not exist.")

cur_datetime = datetime.now()
datetime_str = cur_datetime.strftime("%Y%m%d-%H%M%S")
LOG_DIR = NOTEBOOK_DIR / f".logs/{datetime_str}"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# %%
LAYOUT_FILE = "d3_code_layout.yaml"
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)

# %%
CONFIG_FILE = "base_config.yaml"
config = Config.from_yaml(CONFIG_DIR / CONFIG_FILE)

# %%
train_dataset = xr.load_dataset(
    TRAIN_DATA_DIR / "d3_surf_code_seq_round_state_0_shots_1000000_rounds_40.nc"
)
train_input, train_output = preprocess_data(train_dataset)

dev_dataset = xr.load_dataset(
    DEV_DATA_DIR / "d3_surf_code_seq_round_state_0_shots_20000_rounds_40.nc"
)
dev_input, dev_output = preprocess_data(dev_dataset)

# %%
num_shots, num_rounds, num_anc = train_input["defects"].shape
num_final_anc = int(0.5*num_anc)

# %%
model = get_model(
    defects_shape=(num_rounds, num_anc),
    final_defects_shape=(num_final_anc, ),
    config=config,
)


# %%
model.summary()


# %% [markdown]
# # Training

# %%
%tensorboard --logdir={LOG_DIR}

# %%
tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

model.fit(
    x=train_input,
    y=train_output,
    validation_data=[dev_input, dev_output],
    batch_size=64,
    epochs=5,
    callbacks=[
        tensorboard,
    ],
)


# %%
test_dataset = xr.load_dataset(
    TEST_DATA_DIR / "d3_surf_code_seq_round_state_0_shots_20000_rounds_20.nc"
)
test_input, test_output = preprocess_data(test_dataset)


# %%
eval_output = model.evaluate(x=test_input, y=test_output, batch_size=64)


# %%
test_dataset = xr.load_dataset(
    TEST_DATA_DIR / "d3_surf_code_seq_round_state_0_shots_20000_rounds_20_v2.nc"
)
test_input, test_output = preprocess_data(test_dataset)


# %%
eval_output = model.evaluate(x=test_input, y=test_output, batch_size=64)
