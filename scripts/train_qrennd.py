# %%
# Module import
import os
import pathlib
import random
import numpy as np
import tensorflow as tf

from qrennd import Config, Layout, get_callbacks, get_model, load_datasets

# %%
# Parameters
LAYOUT_FILE = "d3_rotated_layout.yaml"
CONFIG_FILE = "base_config_google_d5.yaml"

USERNAME = os.environ.get("USER")
SCRATH_DIR = pathlib.Path(f"/scratch/{USERNAME}")

DATA_DIR = SCRATH_DIR / "data"
OUTPUT_DIR = SCRATH_DIR / "output"

# %%
# Define used directories
SCRIPT_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.
LAYOUT_DIR = SCRIPT_DIR / "layouts"
CONFIG_DIR = SCRIPT_DIR / "configs"

# %%
# Load setup objects
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)
config = Config.from_yaml(
    filepath=CONFIG_DIR / CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)

config.log_dir.mkdir(exist_ok=True, parents=True)
config.checkpoint_dir.mkdir(exist_ok=True, parents=True)

# %%
# set random seed for tensorflow, numpy and python
random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)


# %%
train_data = load_datasets(config=config, layout=layout, dataset_name="train")
val_data = load_datasets(config=config, layout=layout, dataset_name="dev")

# %%
seq_size = len(layout.get_qubits(role="anc"))

if config.dataset["input"] == "measurements":
    vec_size = len(layout.get_qubits(role="data"))
else:
    vec_size = int(0.5 * seq_size)

model = get_model(
    seq_size=seq_size,
    vec_size=vec_size,
    config=config,
)

# %%
callbacks = get_callbacks(config)

# %%
config.to_yaml(config.run_dir / "config.yaml")

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
