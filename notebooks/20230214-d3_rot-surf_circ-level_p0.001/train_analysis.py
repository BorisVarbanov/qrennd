# %%
import pathlib
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from qrennd.layouts import Layout

# %%
EXPERIMENTS_DIR = "20230320-d3_rot-css-surface_circ-level_p0.001"
RUN_NAME = "20230322-212801_LSTMs16x2_EVALs64x1drop0.2_LR0.001_bs256"

NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR.parent / "data" / EXPERIMENTS_DIR

# %% [markdown]
# # Inspect the training history

# %%
RUN_NAMES = (
    "20230322-191823_LSTMs16x2_EVALs32x1drop0.2_LR0.001_bs256",
    "20230322-182230_LSTMs32x2_EVALs32x1drop0.2_LR0.001_bs256",
    "20230322-212801_LSTMs16x2_EVALs64x1drop0.2_LR0.001_bs256",
    "20230322-195244_LSTMs32x2_EVALs64x1drop0.2_LR0.001_bs256",
)

METRICS = ("loss", "main_output_accuracy")

MWPM_ACCURACIES = dict(
    validation=0.953128,
)

# %%
fig, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=2, sharey=True, sharex=True)

metric = "main_output_accuracy"
ax_iter = chain.from_iterable(axes)

for run_name, ax in zip(RUN_NAMES, ax_iter):
    train_logs = pd.read_csv(DATA_DIR / run_name / "logs/training.log")
    max_acc = np.max(train_logs["val_" + metric])
    print(f"{run_name}: {max_acc}")

    ax.plot(
        train_logs.epoch,
        train_logs[metric],
        linestyle="-",
        linewidth=2,
        color="#2196f3",
        label="Training",
    )

    ax.plot(
        train_logs.epoch,
        train_logs["val_" + metric],
        linestyle="-",
        linewidth=2,
        color="#f44336",
        label="Validation",
    )
    if "accuracy" in metric:
        mwpm_acc = MWPM_ACCURACIES["validation"]
        ax.axhline(mwpm_acc, linestyle="--", color="gray", label=f"MWPM (val)")

    _, lstm_params, eval_params, lr_params, _ = run_name.split("_")
    lstm_size = lstm_params[5:9]
    eval_size = eval_params[5:7]
    learn_rate = lr_params[2:8]

    ax.set_title(f"LSTM: {lstm_size}, EVAL: {eval_size}, LR={learn_rate}")
    ax.legend(frameon=False)
    ax.set_ylim(0.82, 0.97)

for ind in (0, 1):
    axes[1][ind].set_xlabel("Epochs")
    axes[ind][0].set_ylabel("Accuracy")

# %%
fig, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=2, sharey=True, sharex=True)

metric = "main_output_accuracy"
ax_iter = chain.from_iterable(axes)

for run_name, ax in zip(RUN_NAMES, ax_iter):
    train_logs = pd.read_csv(DATA_DIR / run_name / "logs/training.log")
    max_acc = np.max(train_logs["val_" + metric])
    print(f"{run_name}: {max_acc}")

    ax.plot(
        train_logs.epoch,
        train_logs[metric],
        linestyle="-",
        linewidth=2,
        color="#2196f3",
        label="Training",
    )

    ax.plot(
        train_logs.epoch,
        train_logs["val_" + metric],
        linestyle="-",
        linewidth=2,
        color="#f44336",
        label="Validation",
    )
    if "accuracy" in metric:
        mwpm_acc = MWPM_ACCURACIES["validation"]
        ax.axhline(mwpm_acc, linestyle="--", color="gray", label=f"MWPM (val)")

    _, lstm_params, eval_params, lr_params, _ = run_name.split("_")
    lstm_size = lstm_params[5:9]
    eval_size = eval_params[5:7]
    learn_rate = lr_params[2:8]

    ax.set_title(f"LSTM: {lstm_size}, EVAL: {eval_size}, LR={learn_rate}")
    ax.legend(frameon=False)
    ax.set_ylim(0.93, 0.97)

for ind in (0, 1):
    axes[1][ind].set_xlabel("Epochs")
    axes[ind][0].set_ylabel("Accuracy")

# %%
RUN_NAMES = (
    "20230321-203054_LSTMs16x2_EVALs32x1drop0.2_LR0.0001_bs256",
    "20230322-005827_LSTMs32x2_EVALs32x1drop0.2_LR0.0001_bs256",
    "20230321-222403_LSTMs16x2_EVALs64x1drop0.2_LR0.0001_bs256",
    "20230322-181258_LSTMs32x2_EVALs64x1drop0.2_LR0.0001_bs256",
)

METRICS = ("loss", "main_output_accuracy")

MWPM_ACCURACIES = dict(
    validation=0.953128,
)


# %%
fig, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=2, sharey=True, sharex=True)

metric = "main_output_accuracy"
ax_iter = chain.from_iterable(axes)

for run_name, ax in zip(RUN_NAMES, ax_iter):
    train_logs = pd.read_csv(DATA_DIR / run_name / "logs/training.log")
    max_acc = np.max(train_logs["val_" + metric])
    print(f"{run_name}: {max_acc}")

    ax.plot(
        train_logs.epoch,
        train_logs[metric],
        linestyle="-",
        linewidth=2,
        color="#2196f3",
        label="Training",
    )

    ax.plot(
        train_logs.epoch,
        train_logs["val_" + metric],
        linestyle="-",
        linewidth=2,
        color="#f44336",
        label="Validation",
    )
    if "accuracy" in metric:
        mwpm_acc = MWPM_ACCURACIES["validation"]
        ax.axhline(mwpm_acc, linestyle="--", color="gray", label=f"MWPM (val)")

    _, lstm_params, eval_params, lr_params, _ = run_name.split("_")
    lstm_size = lstm_params[5:9]
    eval_size = eval_params[5:7]
    learn_rate = lr_params[2:8]

    ax.set_title(f"LSTM: {lstm_size}, EVAL: {eval_size}, LR={learn_rate}")
    ax.legend(frameon=False)
    ax.set_ylim(0.82, 0.97)

for ind in (0, 1):
    axes[1][ind].set_xlabel("Epochs")
    axes[ind][0].set_ylabel("Accuracy")

# %%
fig, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=2, sharey=True, sharex=True)

metric = "main_output_accuracy"
ax_iter = chain.from_iterable(axes)

for run_name, ax in zip(RUN_NAMES, ax_iter):
    train_logs = pd.read_csv(DATA_DIR / run_name / "logs/training.log")
    max_acc = np.max(train_logs["val_" + metric])
    print(f"{run_name}: {max_acc}")

    ax.plot(
        train_logs.epoch,
        train_logs[metric],
        linestyle="-",
        linewidth=2,
        color="#2196f3",
        label="Training",
    )

    ax.plot(
        train_logs.epoch,
        train_logs["val_" + metric],
        linestyle="-",
        linewidth=2,
        color="#f44336",
        label="Validation",
    )
    if "accuracy" in metric:
        mwpm_acc = MWPM_ACCURACIES["validation"]
        ax.axhline(mwpm_acc, linestyle="--", color="gray", label=f"MWPM (val)")

    _, lstm_params, eval_params, lr_params, _ = run_name.split("_")
    lstm_size = lstm_params[5:9]
    eval_size = eval_params[5:7]
    learn_rate = lr_params[2:8]

    ax.set_title(f"LSTM: {lstm_size}, EVAL: {eval_size}, LR={learn_rate}")
    ax.legend(frameon=False)
    ax.set_ylim(0.93, 0.97)

for ind in (0, 1):
    axes[1][ind].set_xlabel("Epochs")
    axes[ind][0].set_ylabel("Accuracy")

# %%
