# %%
import pathlib

import pandas as pd
from matplotlib import pyplot as plt

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR / "data"
if not DATA_DIR.exists():
    raise ValueError("Data directory not found")

# %%
LOG_FILE = "training.log"
dataframe = pd.read_csv(DATA_DIR / LOG_FILE)

# %%
dataframe

# %%
metrics = ("loss", "main_output_accuracy")

fig, axs = plt.subplots(figsize=(10, 4), ncols=2)
for ax, metric in zip(axs, metrics):

    ax.plot(dataframe.epoch, dataframe[metric], color="blue", label="Training")

    ax.plot(
        dataframe.epoch, dataframe["val_" + metric], color="orange", label="Validation"
    )

    ax.legend(frameon=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric.replace("_", " ").capitalize())

plt.show()

# %%
