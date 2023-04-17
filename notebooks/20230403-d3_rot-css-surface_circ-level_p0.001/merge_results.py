# %%
import pathlib

import xarray as xr

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR / "data"

# %%
decoder_dataarrays = dict(neural_network="qrennd_errors.nc", mwpm="mwpm_errors.nc")

# %%
data_vars = {}

for decoder, dataarray in decoder_dataarrays.items():
    log_errors = xr.load_dataarray(DATA_DIR / dataarray)

    error_probs = log_errors.mean(dim=["state", "shot"])
    data_vars[decoder] = error_probs

error_prob_dataset = xr.Dataset(data_vars)
error_prob_dataset.to_netcdf(DATA_DIR / "log_error_probs.nc")

# %%
