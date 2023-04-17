# %%
import copy
import pathlib

import lmfit
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from qrennd import Config, Layout, get_callbacks, get_model, load_datasets
from qrennd.utils.analysis import error_prob_decay


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
EXPERIMENTS_DIR = "20230302-d5_rot-surf_simulated_google_20M"
RUN_NAME = "20230326-161328_google_simulated_dr0-05_dim128_continue2"

NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR.parent / "data" / EXPERIMENTS_DIR

# %%
RUN_DIR = DATA_DIR / RUN_NAME

nn_vals = xr.load_dataarray(RUN_DIR / "test_results_experimental_.nc")

# %%
MWPM_log_fid = np.array(
    [
        0.99184,
        0.92712,
        0.85624,
        0.80086,
        0.75568,
        0.7117,
        0.67334,
        0.64378,
        0.621,
        0.6028,
        0.58466,
        0.57016,
        0.56142,
    ]
)
CORR_log_fid = np.array(
    [
        0.99184,
        0.93482,
        0.87484,
        0.82576,
        0.78732,
        0.7474,
        0.71138,
        0.68194,
        0.65242,
        0.63664,
        0.61714,
        0.60186,
        0.58976,
    ]
)
BELIEF_log_fid = np.array(
    [
        0.99202,
        0.9444,
        0.89276,
        0.84616,
        0.81528,
        0.77564,
        0.74194,
        0.71532,
        0.68796,
        0.66528,
        0.64676,
        0.63186,
        0.6155,
    ]
)
TENSOR_log_fid = np.array(
    [
        0.9923,
        0.94722,
        0.89784,
        0.8562,
        0.82162,
        0.78272,
        0.75284,
        0.72576,
        0.69972,
        0.67428,
        0.65354,
        0.63902,
        0.62306,
    ]
)

qec_rounds = np.arange(1, 25 + 1, 2)

# %%
decoders = ("mwpm", "corr_mwpm", "belief_match", "tensor_network", "neural_network")

dataset = xr.Dataset(
    data_vars=dict(
        mwpm=(["qec_round"], 1 - MWPM_log_fid),
        corr_mwpm=(["qec_round"], 1 - CORR_log_fid),
        belief_match=(["qec_round"], 1 - BELIEF_log_fid),
        tensor_network=(["qec_round"], 1 - TENSOR_log_fid),
        neural_network=(["qec_round"], 1 - nn_vals.values),
    ),
    coords=dict(qec_round=qec_rounds),
)

# %%
dataset.to_netcdf("data/log_error_probs.nc")

# %%
error_prob_dataset = xr.load_dataset("log_error_probs.nc")
