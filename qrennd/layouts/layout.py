from collections import deque
from copy import deepcopy
from os import path
from typing import Any, Dict, List

import networkx as nx
import yaml
from xarray import DataArray


class Layout:
    """
    A general qubit layout class
    """

    def __init__(self, layout_setup: Dict[str, Dict[str, Any]]) -> None:
        """
        __init__ Initializes the layout class

        Parameters
        ----------
        layout_setup : Dict
            dictionary with the layout name, description and qubit layout.
            The name and description are expected to have string values.
            The qubit layout is expected to be a list of dictionaries.
            Each dictionary defines the name, role, frequency group,
            stabilizer type (if the ancilla performs a parity check)
            and neighbours.

        Raises
        ------
        NotImplementedError
            If the input arguement is not a dictionary.
        """
        if not isinstance(layout_setup, dict):
            raise ValueError(
                f"layout_setup expected as dict, instead got {type(layout_setup)}"
            )

        self.name = layout_setup.get("name", "")
        self.description = layout_setup.get("description", "")

        self.int_order = layout_setup.get("int_order", None)

        self.graph = nx.DiGraph()
        self._load_layout(layout_setup)
        self._set_coords()

    def get_inds(self, **conds: Dict[str, Any]) -> List[str]:
        if conds:
            node_attrs = self.graph.nodes.values()
            inds = [
                i for i, attrs in enumerate(node_attrs) if valid_attrs(attrs, **conds)
            ]
            return inds

        inds = list(range(self.graph.number_of_nodes()))
        return inds

    def get_qubits(self, **conds: Dict[str, Any]) -> List[str]:
        """
        get_qubits Returns the list of qubits in the layout

        Parameters
        ----------
        **qubit_params : dict, optional
        Extra parameter arguements that can be used to filter the qubit list.
        Refer to Layout.param for the possible values.

        Returns
        -------
        List[str]
            List of qubit names.
        """
        if conds:
            node_view = self.graph.nodes(data=True)
            nodes = [node for node, attrs in node_view if valid_attrs(attrs, **conds)]
            return nodes

        nodes = list(self.graph.nodes)
        return nodes

    def get_neighbors(self, qubit: str, **conds: Dict[str, Any]) -> List[str]:
        nbr_nodes = list(self.graph.adj[qubit])

        if conds:
            nodes = [n for n in nbr_nodes if valid_attrs(self.graph[n], **conds)]
            return nodes
        return nbr_nodes

    def adjacency_matrix(self) -> DataArray:
        qubits = self.get_qubits()
        adj_matrix = nx.adjacency_matrix(self.graph)

        data_arr = DataArray(
            data=adj_matrix.toarray(),
            dims=["from_qubit", "to_qubit"],
            coords=dict(
                from_qubit=qubits,
                to_qubit=qubits,
            ),
        )
        return data_arr

    def projection_matrix(self, stab_type: str) -> DataArray:
        adj_mat = self.adjacency_matrix()

        anc_qubits = self.get_qubits(role="anc", stab_type=stab_type)
        data_qubits = self.get_qubits(role="data")

        proj_mat = adj_mat.sel(from_qubit=data_qubits, to_qubit=anc_qubits)
        return proj_mat.rename(from_qubit="data_qubit", to_qubit="anc_qubit")

    @classmethod
    def from_yaml(cls, filename: str) -> "Layout":
        """
        from_file Loads the layout class from a .yaml file.

        Returns
        -------
        Layout
            The initialized layout object.

        Raises
        ------
        ValueError
            If the specified file does not exist.
        ValueError
            If the specified file is not a string.
        """
        if not path.exists(filename):
            raise ValueError("Given path doesn't exist")

        with open(filename, "r") as file:
            layout_setup = yaml.safe_load(file)
            return cls(layout_setup)

    def param(self, param: str, qubit: str) -> Any:
        """
        param Returns the parameter value of a qubit

        Parameters
        ----------
        param : str
            The name of the qubit parameter.
        qubit : str
            The name of the qubit that is being queried.

        Returns
        -------
        Any
            The value of the parameter
        """
        return self.graph.nodes[qubit][param]

    def set_param(self, param: str, qubit: str, value: Any) -> None:
        """
        set_param Sets the value of a given qubit parameter

        Parameters
        ----------
        param : str
            The name of the qubit parameter.
        qubit : str
            The name of the qubit that is being queried.
        value : Any
            The new value of the qubit parameter.
        """
        self.graph.nodes[qubit][param] = value

    def _load_layout(self, layout_dict: Dict[str, Any]) -> None:
        """
        _load_layout Internal function that loads the qubit_info dictionary from
        a provided layout dictionary.

        Parameters
        ----------
        layout_dict : Dict[str, Any]
            The qubit info dictionary that must be specified in the layout.

        Raises
        ------
        ValueError
            If there are unlabeled qubits in the dictionary.
        ValueError
            If any of the qubits is repeated in the layout.
        """
        chip_layout = deepcopy(layout_dict.get("layout"))

        for qubit_info in chip_layout:
            qubit = qubit_info.pop("qubit", None)
            if not qubit:
                raise ValueError("Each qubit in the layout must be labeled.")

            if qubit in self.graph:
                raise ValueError("Qubit label repeated, ensure labels are unique.")

            self.graph.add_node(qubit, **qubit_info)

        for node, attrs in self.graph.nodes(data=True):
            nbr_dict = attrs.pop("neighbors", None)
            for edge_dir, nbr_qubit in nbr_dict.items():
                if nbr_qubit is not None:
                    self.graph.add_edge(node, nbr_qubit, direction=edge_dir)

    def _set_coords(self):
        """
        set_coords Automatically sets the qubit coordinates, if they are not already set

        Parameters
        ----------
        layout : Layout
            The layout of the qubit device.
        """

        def get_shift(direction: str) -> int:
            if direction in ("south", "west"):
                return -1
            return 1

        nodes = list(self.graph.nodes)
        init_node = nodes.pop()
        init_coord = (0, 0)

        set_nodes = set()
        queue = deque()

        queue.appendleft((init_node, init_coord))
        while queue:
            node, coords = queue.pop()

            self.graph.nodes[node]["coords"] = coords
            set_nodes.add(node)

            for _, nbr_node, ord_dir in self.graph.edges(node, data="direction"):
                if nbr_node not in set_nodes:
                    card_dirs = ord_dir.split("_")
                    shifts = tuple(map(get_shift, card_dirs))
                    nbr_coords = tuple(map(sum, zip(coords, shifts)))
                    queue.appendleft((nbr_node, nbr_coords))


def valid_attrs(attrs: Dict[str, Any], **conditions: Dict[str, Any]) -> bool:
    for key, val in conditions.items():
        attr_val = attrs[key]
        if attr_val is None or attr_val != val:
            return False
    return True
