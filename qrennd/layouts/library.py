from collections import defaultdict
from functools import partial
from itertools import count, cycle, product
from typing import Dict, Tuple

from .layout import Layout


def get_data_index(row: int, col: int, distance: int, start_ind: int = 1) -> int:
    row_ind = row // 2
    col_ind = col // 2
    index = start_ind + (row_ind * distance) + col_ind
    return index


def shift_direction(row_shift: int, col_shift: int) -> str:
    ver_direction = "north" if row_shift > 0 else "south"
    hor_direction = "east" if col_shift > 0 else "west"
    direction = f"{ver_direction}_{hor_direction}"
    return direction


def invert_shift(row_shift: int, col_shift: int) -> Tuple[int, int]:
    return -row_shift, -col_shift


def is_valid(row: int, col: int, max_size: int) -> bool:
    if not 0 <= row < max_size:
        return False
    if not 0 <= col < max_size:
        return False
    return True


def add_missing_neighbours(neighbor_data: Dict) -> None:
    directions = ["north_east", "north_west", "south_east", "south_west"]
    for neighbors in neighbor_data.values():
        for dir in directions:
            if dir not in neighbors:
                neighbors[dir] = None


def rot_surf_code(distance: int) -> Layout:
    _check_distance(distance)

    name = f"Rotated d-{distance} surface code layout."
    description = None

    freq_order = ["low", "mid", "high"]

    int_order = dict(
        x_type=["north_east", "north_west", "south_east", "south_west"],
        z_type=["north_east", "south_east", "north_west", "south_west"],
    )

    layout_setup = dict(
        name=name,
        description=description,
        distance=distance,
        freq_order=freq_order,
        interaction_order=int_order,
    )

    grid_size = 2 * distance + 1
    data_indexer = partial(get_data_index, distance=distance, start_ind=1)
    valid_coord = partial(is_valid, max_size=grid_size)

    pos_shifts = (1, -1)
    nbr_shifts = tuple(product(pos_shifts, repeat=2))

    layout_data = []
    neighbor_data = defaultdict(dict)

    freq_seq = cycle(("low", "high"))

    for row in range(1, grid_size, 2):
        freq_group = next(freq_seq)
        for col in range(1, grid_size, 2):
            index = data_indexer(row, col)

            qubit_info = dict(
                qubit=f"D{index}",
                role="data",
                coords=[row, col],
                freq_group=freq_group,
                stab_type=None,
            )
            layout_data.append(qubit_info)

    x_index = count(1)
    for row in range(0, grid_size, 2):
        for col in range(2 + row % 4, grid_size - 1, 4):
            anc_qubit = f"X{next(x_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[row, col],
                freq_group="mid",
                stab_type="x_type",
            )
            layout_data.append(qubit_info)

            for row_shift, col_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_row, data_col):
                    continue
                data_index = data_indexer(data_row, data_col)
                data_qubit = f"D{data_index}"

                direction = shift_direction(row_shift, col_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(row_shift, col_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    z_index = count(1)
    for row in range(2, grid_size - 1, 2):
        for col in range(row % 4, grid_size, 4):
            anc_qubit = f"Z{next(z_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[row, col],
                freq_group="mid",
                stab_type="z_type",
            )
            layout_data.append(qubit_info)

            for row_shift, col_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_row, data_col):
                    continue
                data_index = data_indexer(data_row, data_col)
                data_qubit = f"D{data_index}"

                direction = shift_direction(row_shift, col_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(row_shift, col_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    add_missing_neighbours(neighbor_data)

    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)
    return layout


def _check_distance(distance: int) -> None:
    if not isinstance(distance, int):
        raise ValueError("distance provided must be an integer")
    if distance < 0 or (distance % 2) == 0:
        raise ValueError("distance must be an odd positive integer")
