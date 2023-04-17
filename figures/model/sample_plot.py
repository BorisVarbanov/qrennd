# %%
import pathlib

# %%
from itertools import tee

from matplotlib import pyplot as plt
from matplotlib.patches import BoxStyle, Circle, FancyBboxPatch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# %%
NOTEBOOK_DIR = pathlib.Path.cwd()

IMG_DIR = NOTEBOOK_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %%
ZORDERS = dict(patch=1, line=2, text=3)

LABEL_PARAMS = dict(
    horizontalalignment="center",
    verticalalignment="center",
    weight="bold",
    fontsize=10,
)

LAYER_COLORS = dict(
    LSTM="#ffd54f",
    ConvLSTM="#f44336",
    ReLU="#ff9800",
    sigmoid="#ff9800",
    input="#90a4ae",
    Linear="#2196f3",
    Drop="#0d47a1",
    output="#90a4ae",
)

LABEL_COLORS = dict(
    LSTM="#000000",
    ConvLSTM="#ffffff",
    ReLU="#ffffff",
    sigmoid="#ffffff",
    Linear="#ffffff",
    Drop="#ffffff",
    input=None,
    output=None,
)

COL_WIDTH = 2
ROW_HEIGHT = 1

# %%
from typing import NamedTuple, Optional


class Point(NamedTuple):
    x: float
    y: float


class Layer:
    def __init__(self, corner: Point, width: float, height: float) -> None:
        self.corner = corner
        self.width = width
        self.height = height

    @property
    def center(self) -> Point:
        center_x = self.corner.x + 0.5 * self.width
        center_y = self.corner.y + 0.5 * self.height
        return Point(center_x, center_y)

    @property
    def center_left(self) -> Point:
        return Point(self.corner.x, self.corner.y + 0.5 * self.height)

    @property
    def center_right(self) -> Point:
        return Point(self.corner.x + self.width, self.corner.y + 0.5 * self.height)

    @property
    def center_bottom(self) -> Point:
        return Point(self.corner.x + 0.5 * self.width, self.corner.y)

    @property
    def center_top(self) -> Point:
        return Point(self.corner.x + 0.5 * self.width, self.corner.y + self.height)


class LayerRect(Layer):
    """Class for keeping track of an item in inventory."""

    def __init__(self, name, corner: Point, width: float, height: float) -> None:
        self.name = name
        super().__init__(corner, width, height)

    def get_patch(
        self,
        stylename="round",
        pad=0.05,
        facecolor="white",
        edgecolor="black",
        **kwargs,
    ):
        boxstyle = BoxStyle(stylename=stylename, pad=pad)
        patch = FancyBboxPatch(
            tuple(self.corner),
            self.width,
            self.height,
            boxstyle=boxstyle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        return patch


class InputCirc(Layer):
    """Class for keeping track of an item in inventory."""

    def __init__(
        self, corner: Point, radius: float, width: float, height: float
    ) -> None:
        self.radius = radius
        super().__init__(corner, width, height)

    def get_patch(self, edgecolor="black", facecolor="white", **kwargs):
        patch = Circle(
            tuple(self.center),
            self.radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        return patch


class ContLayer(Layer):
    def __init__(
        self,
        corner: Point,
        width: float,
        height: float,
        padding: float,
    ) -> None:
        self.padding = padding
        super().__init__(corner, width, height)

    def get_patches(self, edgecolor="black", facecolor="white", **kwargs):
        prefactors = (0, 0.5, 1)
        padded_width = self.width - 2 * self.padding
        offsets = (prefactor * padded_width for prefactor in prefactors)

        x = self.corner.x + self.padding
        y = self.corner.y + 0.5 * self.height
        for offset in offsets:
            center = (x + offset, y)
            patch = Circle(
                center,
                radius=0.1,
                facecolor=facecolor,
                edgecolor=edgecolor,
                **kwargs,
            )
            yield patch


# %%
fig, ax = plt.subplots(figsize=(14, 14), dpi=100)

rec_cols = (1, 4, 7, 10, 13)
TIME_STEPS = ("layer", "layer", "dots", "layer", "layer")

rec_rows = (1, 3, 5)
LAYER_NAMES = ("input", "LSTM", "LSTM")

layers = []

for row, name in zip(rec_rows, LAYER_NAMES):
    row_layers = []
    for col, step in zip(rec_cols, TIME_STEPS):
        corner = Point(col, row)

        if step == "layer":
            if name == "input":
                layer = InputCirc(corner, radius=0.3, width=2, height=1)
            else:
                layer = LayerRect(name, corner, width=2, height=1)

            patch = layer.get_patch(facecolor=LAYER_COLORS[name])
            ax.add_artist(patch)

            if name != "input":
                center = layer.center
                ax.text(
                    center.x,
                    center.y,
                    layer.name,
                    color=LABEL_COLORS[name],
                    zorder=ZORDERS["text"],
                    **LABEL_PARAMS,
                )
        elif step == "dots":
            layer = ContLayer(corner, width=2, height=1, padding=0.5)
            for patch in layer.get_patches(facecolor="black", linewidth=0):
                ax.add_artist(patch)
        else:
            raise ValueError("Unknown step")

        row_layers.append(layer)
    layers.append(row_layers)

num_rows = len(layers)

col_pad = 1
arrow_pad = 0.05

for row_ind, row_layers in enumerate(layers):
    num_cols = len(row_layers)
    for col_ind, layer in enumerate(row_layers):
        if col_ind != (num_cols - 1) and row_ind in (1, 2):
            x, y = layer.center_right
            ax.arrow(
                x + arrow_pad,
                y,
                col_pad - 2 * arrow_pad,
                0,
                head_width=0.2,
                head_length=0.2,
                color="black",
                length_includes_head=True,
            )
        if row_ind != num_rows - 1:
            if not isinstance(layer, ContLayer):
                offset = -0.25 if isinstance(layer, InputCirc) else 0
                x, y = layer.center_top
                ax.arrow(
                    x,
                    y + arrow_pad + offset,
                    0,
                    col_pad - 2 * arrow_pad - offset,
                    head_width=0.2,
                    head_length=0.2,
                    color="black",
                    length_includes_head=True,
                )
        else:
            if col_ind == (num_cols - 1):
                offset = -0.25 if isinstance(layer, InputCirc) else 0
                x, y = layer.center_top
                ax.arrow(
                    x,
                    y + arrow_pad + offset,
                    0,
                    col_pad - 2 * arrow_pad - offset,
                    head_width=0.2,
                    head_length=0.2,
                    color="black",
                    length_includes_head=True,
                )

relu_layer = LayerRect(name="ReLU", corner=Point(13, 7), width=2, height=1)
patch = relu_layer.get_patch(facecolor=LAYER_COLORS["ReLU"])
ax.add_artist(patch)

center = relu_layer.center
ax.text(
    center.x,
    center.y,
    relu_layer.name,
    color="white",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

proj_input = InputCirc(corner=Point(16, 1), radius=0.3, width=2, height=1)
patch = proj_input.get_patch(facecolor=LAYER_COLORS["input"])
ax.add_artist(patch)

x, y = proj_input.center_top
ax.arrow(
    x,
    y + arrow_pad - 0.25,
    0,
    7 - 2 * arrow_pad + 0.25,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

concat_layer = LayerRect(name="Concat", corner=Point(16, 9), width=2, height=1)
patch = concat_layer.get_patch(facecolor="#80cbc4")
ax.add_artist(patch)

center = concat_layer.center
ax.text(
    center.x,
    center.y,
    concat_layer.name,
    color="black",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

x, y = concat_layer.center_right
ax.arrow(
    x + arrow_pad,
    y,
    col_pad - 2 * arrow_pad,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

eval_names = ("Linear", "ReLU", "Drop", "Linear", "sigmoid", "output")
eval_rows = (9, 11)
eval_cols = (19, 22, 25, 28, 31, 34)

num_cols = len(eval_cols)
col_inds = range(1, num_cols + 1)

for row in eval_rows:
    for ind, col, name in zip(col_inds, eval_cols, eval_names):
        if name == "output":
            corner = Point(col - 0.75, row)
            layer = InputCirc(corner, radius=0.3, width=2, height=1)
        else:
            corner = Point(col, row)
            layer = LayerRect(name=name, corner=corner, width=2, height=1)

        patch = layer.get_patch(facecolor=LAYER_COLORS[name])
        ax.add_artist(patch)

        if name != "output":
            ax.text(
                layer.center.x,
                layer.center.y,
                name,
                color=LABEL_COLORS[name],
                zorder=ZORDERS["text"],
                **LABEL_PARAMS,
            )

        if ind != num_cols:
            x, y = layer.center_right
            ax.arrow(
                x + arrow_pad,
                y,
                col_pad - 2 * arrow_pad + offset,
                0,
                head_width=0.2,
                head_length=0.2,
                color="black",
                length_includes_head=True,
            )

ax.plot((14, 14), (8 + arrow_pad, 11.5), color="black")
ax.arrow(
    14,
    9.5,
    2 - arrow_pad,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)
ax.arrow(
    14,
    11.5,
    5 - arrow_pad,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

ax.text(
    35.25,
    9.5,
    r"$\mathrm{o_{main}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)
ax.text(
    35.25,
    11.5,
    r"$\mathrm{o_{aux}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)

ax.text(
    2,
    0.75,
    r"$\mathrm{\vec{d}_{0}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)
ax.text(
    5,
    0.75,
    r"$\mathrm{\vec{d}_{1}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)

ax.text(
    11,
    0.75,
    r"$\mathrm{\vec{d}_{N-1}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)
ax.text(
    14,
    0.75,
    r"$\mathrm{\vec{d}_{N}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)
ax.text(
    17,
    0.75,
    r"$\mathrm{\vec{d}_{N}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)

ax.set_xlim(0, 36)
ax.set_ylim(0, 13)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.axis("off")

image_name = "base_model"
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
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

layer = InputCirc(corner=Point(1, 1), radius=0.3, width=1, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["input"])
ax.add_artist(patch)

ax.arrow(
    1.9,
    1.5,
    1,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

ax.plot(
    [4.5, 4.5, 3.5, 3.5, 4.5], [2.05, 2.55, 2.55, 0.55, 0.55], color="black", zorder=0
)
ax.arrow(
    4.5,
    0.55,
    0,
    0.4,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = LayerRect(name="LSTM", corner=Point(3, 1), width=2, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["LSTM"])
ax.add_artist(patch)

center = layer.center
ax.text(
    center.x,
    center.y,
    layer.name,
    color="black",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

ax.arrow(
    5.05,
    1.5,
    0.9,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

ax.plot(
    [7.5, 7.5, 6.5, 6.5, 7.5], [2.05, 2.55, 2.55, 0.55, 0.55], color="black", zorder=0
)
ax.arrow(
    7.5,
    0.55,
    0,
    0.4,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = LayerRect(name="LSTM", corner=Point(6, 1), width=2, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["LSTM"])
ax.add_artist(patch)

center = layer.center
ax.text(
    center.x,
    center.y,
    layer.name,
    color="black",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

ax.arrow(
    8.05,
    1.5,
    0.9,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = LayerRect(name="ReLU", corner=Point(9, 1), width=2, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["ReLU"])
ax.add_artist(patch)

center = layer.center
ax.text(
    center.x,
    center.y,
    layer.name,
    color="white",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

ax.plot([11.05, 11.5], [1.5, 1.5], color="black")
ax.plot([11.55, 11.5], [0.5, 2.5], color="black")
ax.arrow(
    11.55,
    0.5,
    0.45,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)
ax.arrow(
    11.55,
    2.5,
    3.45,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = LayerRect(name="Concat", corner=Point(12, 0), width=2, height=1)
patch = layer.get_patch(facecolor="#80cbc4")
ax.add_artist(patch)

center = layer.center
ax.text(
    center.x,
    center.y,
    layer.name,
    color="black",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

ax.arrow(
    14.05,
    0.5,
    0.9,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = LayerRect(name="Eval", corner=Point(15, 0), width=2, height=1)
patch = layer.get_patch(facecolor="#2196f3")
ax.add_artist(patch)

center = layer.center
ax.text(
    center.x,
    center.y,
    layer.name,
    color="white",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

ax.arrow(
    17.05,
    0.5,
    0.9,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = LayerRect(name="Eval", corner=Point(15, 2), width=2, height=1)
patch = layer.get_patch(facecolor="#2196f3")
ax.add_artist(patch)

center = layer.center
ax.text(
    center.x,
    center.y,
    layer.name,
    color="white",
    zorder=ZORDERS["text"],
    **LABEL_PARAMS,
)

ax.arrow(
    17.05,
    2.5,
    0.9,
    0,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

layer = InputCirc(corner=Point(17.9, 0), radius=0.3, width=1, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["output"])
ax.add_artist(patch)

layer = InputCirc(corner=Point(17.9, 2), radius=0.3, width=1, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["output"])
ax.add_artist(patch)

layer = InputCirc(corner=Point(1, -1), radius=0.3, width=1, height=1)
patch = layer.get_patch(facecolor=LAYER_COLORS["input"])
ax.add_artist(patch)

ax.plot([1.9, 13], [-0.5, -0.5], color="black")
ax.arrow(
    13,
    -0.5,
    0,
    0.45,
    head_width=0.2,
    head_length=0.2,
    color="black",
    length_includes_head=True,
)

ax.text(
    19.5,
    0.5,
    r"$\mathrm{o_{main}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)
ax.text(
    19.5,
    2.5,
    r"$\mathrm{o_{aux}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)

ax.text(
    0.25,
    1.25,
    r"$\mathrm{\vec{d}_{n}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)
ax.text(
    0.25,
    -0.75,
    r"$\mathrm{\vec{d}_{N+1}}$",
    ha="center",
    va="center",
    weight="normal",
    fontsize=12,
)

ax.set_xlim(1, 19)
ax.set_ylim(-1, 4)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.axis("off")

image_name = "simple_base_model"
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
