# %% [markdown]
# # MWPM decoder (with `pymatching`)

# %%
import pathlib
from datetime import datetime
from typing import Optional
import re

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

import pymatching
import stim
import sinter

from qrennd import get_model, Config
from qec_util.util.syndrome import get_syndromes, get_defects, get_final_defects
from qec_util.util.analysis import (
    logical_fidelity,
    LogicalFidelityDecay,
    lmfit_par_to_ufloat,
)
from qec_util import Layout
from qec_util.layouts import plot as plot_layout

# %%
def preprocess_data(dataset, proj_mat):
    anc_meas = dataset.anc_meas.transpose("shot", "qec_round", "anc_qubit")
    data_meas = dataset.data_meas.transpose("shot", "data_qubit")

    syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)
    defects = get_defects(syndromes)
    defects = defects.stack(n=["qec_round", "anc_qubit"]).values

    proj_syndrome = (data_meas @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    defects = np.append(defects, final_defects, axis=1)

    log_meas = dataset.data_meas.sum(dim="data_qubit") % 2
    log_errors = log_meas ^ dataset.log_state

    return defects, log_errors.data


# %% [markdown]
# # Setting up

# %%
EXP_NAME: str = "20230117-d3_rot-surf_circ-level_logical-error-rate-first-try"

NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

# experiment folder
EXP_DIR = NOTEBOOK_DIR / "data" / EXP_NAME
if not EXP_DIR.exists():
    raise ValueError("Experimental directory does not exist.")

# folder with the layout of the surface code
LAYOUT_DIR = EXP_DIR / "config"
if not LAYOUT_DIR.exists():
    raise ValueError("Layout directory does not exist.")

cur_datetime = datetime.now()
datetime_str = cur_datetime.strftime("%Y%m%d-%H%M%S")

LOG_DIR = NOTEBOOK_DIR / f"logs/{datetime_str}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = NOTEBOOK_DIR / "tmp/checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# %%
LAYOUT_FILE = "d3_rotated_layout.yaml"
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)

fig, ax = plt.subplots(figsize=(4, 4))
plot_layout(layout, label_qubits=True, draw_patches=True, axis=ax)
plt.tight_layout()
plt.show()

# %%
proj_mat = layout.projection_matrix(stab_type="z_type")

# %%
DIR = EXP_DIR / "test"
directories = DIR.glob("*")
for circuit_dir in directories:
    print(circuit_dir.name)

# %% [markdown]
# # Load dataset and decode

# %%
ROUNDS_RANGE = range(3, 60)
LOG_STATES = [0, 1]


num_rounds = len(ROUNDS_RANGE)
num_states = len(LOG_STATES)
data = np.zeros((num_states, num_rounds))

for round_idx, num_rounds in enumerate(ROUNDS_RANGE):
    for state_idx, state in enumerate(LOG_STATES):
        circ_dir = f"surf-code_d3_bZ_s{state}_n10000_r{num_rounds}"
        print(circ_dir, end="\r")

        # load dataset
        test_dataset = xr.load_dataset(EXP_DIR / "test" / circ_dir / "measurements.nc")
        defects, log_errors = preprocess_data(test_dataset, proj_mat)

        # generate decoder
        DETECTOR_ERROR_MODEL = EXP_DIR / "test" / circ_dir / "detector_error_model"
        if not DETECTOR_ERROR_MODEL.exists():
            raise ValueError(
                f"stim model does not exist in file: {DETECTOR_ERROR_MODEL}"
            )

        detector_error_model = stim.DetectorErrorModel.from_file(DETECTOR_ERROR_MODEL)
        MWPM = pymatching.Matching.from_detector_error_model(detector_error_model)

        # decode
        prediction = np.array([MWPM.decode(i) for i in defects]).flatten()

        # analyse performance
        log_fid = logical_fidelity(prediction, log_errors)

        data[state_idx, round_idx] = log_fid

fid_arr = xr.DataArray(
    data,
    dims=["log_state", "qec_round"],
    coords=dict(log_state=LOG_STATES, qec_round=list(ROUNDS_RANGE)),
)

# %% [markdown]
# # Analysis metrics of decoding

# %%
log_fid = fid_arr.mean(dim="log_state")

# %%
model = LogicalFidelityDecay()
params = model.guess(log_fid.values, x=log_fid.qec_round.values)
out = model.fit(log_fid.values, params, x=log_fid.qec_round.values, min_qec=3)

# %%
ax = out.plot_fit()
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")

# %%
print("logical error rate per cycle =", lmfit_par_to_ufloat(out.params["error_rate"]))

# %%
