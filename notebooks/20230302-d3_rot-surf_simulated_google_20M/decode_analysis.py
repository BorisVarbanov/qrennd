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
EXPERIMENTS_DIR = "20230302-d3_rot-surf_simulated_google_20M"
RUN_NAME = "20230305-112822_google_simulated_d3_20M_dr0-05"

NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR.parent / "data" / EXPERIMENTS_DIR

# %%
error_prob_dataset = xr.load_dataset("log_error_probs.nc")

# %%


# %%


# %%
