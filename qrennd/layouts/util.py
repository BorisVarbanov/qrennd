from collections import deque

from .layout import Layout


def set_coords(layout: Layout) -> None:
    """
    set_coords Automatically sets the qubit coordinates. This is used for
    plotting the layout.
    """

    def get_shift(direction: str) -> int:
        if direction in ("south", "west"):
            return -1
        return 1

    nodes = list(layout.graph.nodes)
    init_node = nodes.pop()
    init_coord = (0, 0)

    set_nodes = set()

    queue = deque()

    queue.appendleft((init_node, init_coord))
    while queue:
        node, coords = queue.pop()

        layout.graph.nodes[node]["coords"] = coords
        set_nodes.add(node)

        for _, nbr_node, ord_dir in layout.graph.edges(node, data="direction"):
            if nbr_node not in set_nodes:
                card_dirs = ord_dir.split("_")
                shifts = tuple(map(get_shift, card_dirs))
                nbr_coords = tuple(map(sum, zip(coords, shifts)))
                queue.appendleft((nbr_node, nbr_coords))
