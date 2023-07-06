# %%
# Module import
import os
import pathlib
import random

import numpy as np
import tensorflow as tf

from qrennd import Config, Layout, get_callbacks, get_model, load_datasets, set_coords

# %%
# Parameters
LAYOUT_FILE = "d5_rotated_layout.yaml"
CONFIG_FILE = "base_config_google_d5.yaml"

USERNAME = os.environ.get("USER")
SCRATH_DIR = pathlib.Path(f"/scratch/{USERNAME}")

DATA_DIR = SCRATH_DIR / "data"
OUTPUT_DIR = SCRATH_DIR / "output"


# %%
# Load setup objects
CONFIG_DIR = pathlib.Path.cwd() / "configs"
config = Config.from_yaml(
    filepath=CONFIG_DIR / CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)
config.log_dir.mkdir(exist_ok=True, parents=True)
config.checkpoint_dir.mkdir(exist_ok=True, parents=True)

LAYOUT_DIR = DATA_DIR / config.experiment / "config"
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)
set_coords(layout)

# %%
config.to_yaml(config.run_dir / "config.yaml")

# %%
# set random seed for tensorflow, numpy and python
random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)


# %%
train_data = load_datasets(config=config, layout=layout, dataset_name="train")
val_data = load_datasets(config=config, layout=layout, dataset_name="dev")

# %%
anc_qubits = layout.get_qubits(role="anc")
num_anc = len(anc_qubits)

if config.model["type"] in ("ConvLSTM", "Conv_LSTM"):
    rec_features = (layout.distance + 1, layout.distance + 1, 1)
else:
    rec_features = num_anc

if config.dataset["input"] == "measurements":
    data_qubits = layout.get_qubits(role="data")
    eval_features = len(data_qubits)
else:
    eval_features = int(num_anc / 2)


model = get_model(
    rec_features=rec_features,
    eval_features=eval_features,
    config=config,
)

# %%
callbacks = get_callbacks(config)

# %%
history = model.fit(
    train_data,
    validation_data=val_data,
    batch_size=config.train["batch_size"],
    epochs=config.train["epochs"],
    callbacks=callbacks,
    shuffle=True,
    verbose=0,
)

# %%
model.save(config.checkpoint_dir / "final_weights.hdf5")

# %%
