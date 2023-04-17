# %%
import copy
import pathlib

# %%
from itertools import product
from typing import Any, Dict, Iterable, Iterator

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from qrennd import Config, Layout, get_callbacks, get_model, load_datasets


def parameter_product(**params: Any) -> Iterator[Dict[str, Any]]:
    keys = params.keys()
    vals = params.values()

    val_iters = [val if isinstance(val, Iterable) else (val,) for val in vals]
    val_prods = product(*val_iters)

    param_prods = (dict(zip(keys, prod)) for prod in val_prods)
    yield from param_prods


# %%
EXP_NAME: str = "20230403-d3_rot-css-surface_circ-level_p0.001"
RUN_NAME: str = "20230403-213931_LSTMs64x2_EVALs64x1drop0.2_LR0.001_bs256"

NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

# experiment folder
RUN_DIR = NOTEBOOK_DIR.parent / "data" / EXP_NAME / RUN_NAME

HOME_DIR = pathlib.Path.home()
DATA_DIR = HOME_DIR / "data"
EXP_DIR = DATA_DIR / EXP_NAME

RESULT_DIR = NOTEBOOK_DIR / "data"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# %%
LAYOUT_FILE = "d3_rotated_layout.yaml"
CONFIG_FILE = "config_updated.yaml"

WEIGHTS_FILE = "weights.hdf5"
LOG_FILE = "training.log"

# %%
DATASET_TYPE = "test"  # Possible types are "train", "dev" and "test"

# Fixed parameters
ROOT_SEED = np.random.randint(999999)  # Initial seed for the RNG

# NUM_SHOTS: int = 500000  # Train dataset number of shots
# NUM_SHOTS: int = 50000  # Validation dataset number of shots
NUM_SHOTS: int = 20000  # Test dataset number of shots

# ROUNDS = list(range(1, 41, 4))  # Train and dev dataset
ROUNDS = list(range(10, 301, 10))  # Test dataset

MEAS_RESET = False  # No resets following measurements
BASES = "Z"
STAB_TYPE = "z_type"
STATES = [0, 1]
PROBS = 0.001

experiment_name = "surf-code_d{distance}_b{basis}_s{state}_n{shots}_r{rounds}_p0.001"

# %%
config = Config.from_yaml(
    filepath=RUN_DIR / CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=None,
)

layout = Layout.from_yaml(EXP_DIR / "config" / LAYOUT_FILE)

# %%
# %%
anc_qubits = layout.get_qubits(role="anc")

rec_features = len(anc_qubits)
eval_features = int(0.5 * rec_features)

model = get_model(
    rec_features=rec_features,
    eval_features=eval_features,
    config=config,
)

model.load_weights(RUN_DIR / "checkpoint" / WEIGHTS_FILE)

# %%
result_shape = tuple(len(params) for params in (ROUNDS, STATES))
results = np.zeros(shape=(*result_shape, NUM_SHOTS), dtype=bool)

dim_inds = (range(dim) for dim in result_shape)
ind_products = product(*dim_inds)

datasets = load_datasets(config, layout, dataset_name="test", concat=False)

# %%
threshold = 0.5

for inds, dataset in zip(ind_products, datasets):
    inputs, outputs = dataset.values()
    main_probs, aux_probs = model.predict(inputs)
    predictions = np.squeeze(main_probs >= threshold)

    errors = predictions != outputs
    results[inds] = errors

# %%
shots = list(range(1, NUM_SHOTS + 1))
logical_errors = xr.DataArray(
    results,
    dims=["qec_round", "state", "shot"],
    coords=dict(
        state=STATES,
        qec_round=ROUNDS,
        shot=shots,
    ),
)

# %%
logical_errors.to_netcdf(RESULT_DIR / "qrennd_errors.nc")

# %%
mwpm_logical_errors = xr.load_dataarray(RESULT_DIR / "mwpm_errors.nc")

# %%
fig, ax = plt.subplots()

avg_errors = logical_errors.mean(dim=["state", "shot"])
ax.scatter(avg_errors.qec_round, avg_errors, color="red", label="Neural Network")

avg_errors = mwpm_logical_errors.mean(dim=["state", "shot"])
ax.scatter(avg_errors.qec_round, avg_errors, color="blue", label="MWPM")

fig.legend(frameon=False)

plt.show

# %%
