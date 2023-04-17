# %%
import pathlib
import warnings
from typing import Union

import lmfit
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


# %%
def get_latex_rc_params(fig_width=None, fig_height=None, columns=1):
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    MAX_HEIGHT = 8.0

    if columns not in [1, 2]:
        raise ValueError("Plotting in three-column style is not supported")
    golden_mean = (np.sqrt(5) - 1.0) / 2.0

    fig_width = fig_width or 3.39 if columns == 1 else 6.9  # width in inches

    fig_height = fig_height or fig_width * golden_mean  # height in inches

    if fig_height > MAX_HEIGHT:
        warnings.warn(
            "Figure height {} is too large, setting to {}"
            "inches instead".format(fig_height, MAX_HEIGHT)
        )
        fig_height = MAX_HEIGHT

    params = {
        "backend": "ps",
        "text.latex.preamble": [r"\usepackage{gensymb}"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    return params


# %%
def error_prob_decay(
    qec_round: Union[int, np.ndarray], error_rate: float, round_offset: float
) -> Union[int, np.ndarray]:
    return 0.5 * (1 - (1 - 2 * error_rate) ** (qec_round - round_offset))


def to_percent(value):
    return 100 * value


# %%
NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR / "data"
IMG_DIR = NOTEBOOK_DIR / "img"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# %%
error_prob_dataset = xr.load_dataset(DATA_DIR / "log_error_probs.nc")

# %%
decoders = ("mwpm", "corr_mwpm", "belief_match", "tensor_network", "neural_network")

colors = dict(
    neural_network="#f44336",
    mwpm="#03a9f4",
    corr_mwpm="#0d47a1",
    belief_match="#4caf50",
    tensor_network="#ffc107",
)

labels = dict(
    neural_network="Neural Network",
    mwpm="MWPM",
    corr_mwpm="Corr. MWPM",
    belief_match="Belief matching",
    tensor_network="Tensor Network",
)


markers = dict(
    mwpm="o",
    belief_match="v",
    neural_network="h",
    tensor_network="d",
    corr_mwpm="s",
)


# %%
decay_model = lmfit.Model(error_prob_decay)
decay_model.set_param_hint("error_rate", min=0, max=1, value=0.1, vary=True)
decay_model.set_param_hint("round_offset", value=0, vary=False)

start_round = 5

# %%
formatter = ScalarFormatter()


fig, (prob_ax, fid_ax) = plt.subplots(
    nrows=2,
    sharex=True,
    figsize=(6, 8),
    dpi=100,
    gridspec_kw=dict(height_ratios=(3, 2)),
)


for decoder in decoders:
    color = colors.get(decoder)
    error_probs = error_prob_dataset[decoder]
    decoder_label = labels.get(decoder)
    marker = markers.get(decoder)

    sel_probs = error_probs.where(error_probs.qec_round >= start_round, drop=True)
    model_fit = decay_model.fit(
        sel_probs,
        qec_round=sel_probs.qec_round,
    )

    error_rate = to_percent(model_fit.params["error_rate"].value)
    rate_stderr = to_percent(model_fit.params["error_rate"].stderr)

    label = f"{decoder_label} $\\varepsilon_L$ = ({error_rate:.2f} $\pm$ {rate_stderr:.2f})%"

    prob_ax.plot(
        error_probs.qec_round,
        error_probs,
        linestyle="None",
        marker=marker,
        color=color,
        label=label,
    )

    prob_ax.plot(
        sel_probs.qec_round,
        model_fit.best_fit,
        linestyle="-",
        marker=None,
        color=color,
    )

    logical_fids = 1 - 2 * error_probs
    fid_best_fit = 1 - 2 * model_fit.best_fit

    fid_ax.plot(
        error_probs.qec_round,
        logical_fids,
        linestyle="None",
        marker=marker,
        color=color,
        label=label,
    )

    fid_ax.plot(
        sel_probs.qec_round,
        fid_best_fit,
        linestyle="-",
        marker=None,
        color=color,
    )

prob_ax.set_ylim(0, 0.45)
prob_ax.legend(frameon=False)
prob_ax.set_ylabel(r"Logical error probability, $p_L$")
prob_ax.minorticks_on()
fid_ax.set_xticks(range(0, 26, 5), minor=False)


fid_ax.set_ylabel(r"Logical fidelity, $F = 1 - 2p_L$")
fid_ax.set_xlabel("QEC round")

fid_ax.set_xlim(0, 26)
fid_ax.set_xticks([0, 5, 10, 15, 20, 25], minor=False)

fid_ax.set_ylim(0.1, 1.1)
fid_ax.set_yscale("log")
fid_ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
fid_ax.yaxis.set_major_formatter(formatter)

image_name = "logical_performance_comparison"
for file_format in ("pdf", "png"):
    full_name = f"{image_name}.{file_format}"
    fig.savefig(
        IMG_DIR / full_name,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        format=file_format,
        pad_inches=0,
    )

plt.show()

# %%
