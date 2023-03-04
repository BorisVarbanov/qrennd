"""Layout plotting module."""
import re
from itertools import combinations
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from .layout import Layout

RE_FILTER = re.compile("([a-zA-Z]+)([0-9]+)")


def plot(
    layout: Layout,
    label_qubits: bool = True,
    draw_patches: bool = True,
    *,
    axis: Optional[plt.Axes] = None,
) -> Union[Figure, None]:
    """
    plot Function to plot a layout

    Parameters
    ----------
    layout : Layout
        [description]
    label_qubits : bool, optional
        [description], by default True
    axis : Optional[plt.Axes], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    plotter = MatplotlibPlotter(layout, axis)
    return plotter.plot(label_qubits, draw_patches)


class MatplotlibPlotter:
    """A plotter based on the matplotlib library for Layout objects."""

    zorders = dict(
        line=3,
        patch=1,
        circle=5,
        text=20,
    )

    qubit_circ_params = dict(
        radius=0.35,
        lw=1,
        ec="black",
    )

    label_params = dict(
        ha="center",
        va="center",
        weight="bold",
    )

    line_params = dict(
        color="black",
        linestyle="--",
        lw=1,
    )

    patch_params = dict(lw=0, alpha=0.3)

    def __init__(
        self,
        layout: Layout,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        for qubit in layout.get_qubits():
            if not layout.param("coords", qubit):
                raise ValueError(
                    f"All qubits in 'layout' must have 'coords' parameter set, qubit {qubit} does not."
                )
        self.layout = layout

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def _label_qubit(self, qubit: str, x: float, y: float) -> None:
        match = RE_FILTER.match(str(qubit))
        if match is None:
            raise ValueError(f"Unexpected qubit label {qubit}")
        label, ind = match.groups()
        q_label = f"${label}_\\mathrm{{{ind}}}$"
        self.ax.text(
            x,
            y,
            q_label,
            zorder=self.zorders["text"],
            **self.label_params,
        )

    def _draw_qubit_circ(self, x: float, y: float, color: str) -> None:
        qubit_circ = Circle(
            (x, y),
            color=color,
            zorder=self.zorders["circle"],
            **self.qubit_circ_params,
        )
        self.ax.add_artist(qubit_circ)

    def _draw_patch(self, cords: Iterable[float], color: str) -> None:
        patch = Polygon(
            cords,
            color=color,
            zorder=self.zorders["patch"],
            **self.patch_params,
        )
        self.ax.add_artist(patch)

    def _draw_connection(self, qubits: Iterable[str]) -> None:
        q_coords = (self.layout.param("coords", qubit) for qubit in qubits)
        x_cords, y_cords = zip(*q_coords)
        self.ax.plot(x_cords, y_cords, **self.line_params)

    def _draw_qubits(self, label_qubits: Optional[bool] = True) -> None:
        qubits = self.layout.get_qubits()

        init_qubit = qubits.pop()
        drawn_qubits = set()

        def _dfs_draw(qubit: str) -> None:
            if qubit not in drawn_qubits:
                role = self.layout.param("role", qubit)
                if role == "data":
                    color = "#fafafa"
                else:
                    stab_type = self.layout.param("stab_type", qubit)
                    color = "#2196f3" if stab_type == "x_type" else "#4caf50"

                x, y = self.layout.param("coords", qubit)
                self._draw_qubit_circ(x, y, color)
                if label_qubits:
                    self._label_qubit(qubit, x, y)
                drawn_qubits.add(qubit)

                for neighbour in self.layout.get_neighbors(qubit):
                    _dfs_draw(neighbour)
                    self._draw_connection((qubit, neighbour))

        _dfs_draw(init_qubit)
        qubit_cords = (self.layout.param("coords", qubit) for qubit in qubits)
        x_cords, y_cords = zip(*qubit_cords)
        self.ax.set_xlim(min(x_cords) - 2, max(x_cords) + 2)
        self.ax.set_ylim(min(y_cords) - 2, max(y_cords) + 2)

    def _draw_patches(self):
        anc_qubits = self.layout.get_qubits(role="anc")
        for anc in anc_qubits:
            anc_cords = self.layout.param("coords", anc)
            stab_type = self.layout.param("stab_type", anc)
            color = "#2196f3" if stab_type == "x_type" else "#4caf50"
            neigbors = self.layout.get_neighbors(anc)
            for data_pair in combinations(neigbors, 2):
                pair_cords = list(
                    self.layout.param("coords", data) for data in data_pair
                )
                dist = sum([abs(i - j) for i, j in zip(*pair_cords)])  # type: ignore
                if dist <= 2:
                    cords = [anc_cords, *pair_cords]
                    self._draw_patch(cords, color)

    def plot(
        self,
        label_qubits: Optional[bool] = True,
        draw_patches: Optional[bool] = True,
    ) -> Union[Figure, None]:
        self._draw_qubits(label_qubits)
        if draw_patches:
            self._draw_patches()
        return self.fig
