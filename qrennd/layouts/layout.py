from __future__ import annotations

from collections import deque
from copy import copy, deepcopy
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import yaml
from xarray import DataArray


class Layout:
    """
    A general qubit layout class
    """

    def __init__(self, setup: Dict[str, Any]) -> None:
        """
        __init__ Initiailizes the layout.

        Parameters
        ----------
        setup : Dict[str, Any]
            The layout setup, provided as a dict.

            The setup dictionary is expected to have a 'layout' item, containing
            a list of dictionaries. Each such dictionary (dict[str, Any]) must define the
            qubit label (str) corresponding the 'qubit' item. In addition, each dictionary
            must also have a 'neighbors" item that defines a dictonary (dict[str, str])
            of ordinal directions and neighbouring qubit labels. Apart from these two items,
            each dictionary can hold any other metadata or parameter relevant to these qubits.

            In addition to the layout list, the setup dictionary can also optioonally
            define the name of the layout (str), a description (str) of the layout as well
            as the interaction order of the different types of check, if the layout is used
            for a QEC code.

        Raises
        ------
        ValueError
            If the type of the setup provided is not a dictionary.
        """
        if not isinstance(setup, dict):
            raise ValueError(
                f"layout_setup expected as dict, instead got {type(setup)}"
            )

        self.name = setup.get("name")
        self.distance = setup.get("distance")
        self.description = setup.get("description")
        self.interaction_order = setup.get("interaction_order")

        self.graph = nx.DiGraph()
        self._load_layout(setup)
        self._set_coords()

        qubits = list(self.graph.nodes)
        num_qubits = len(qubits)
        self._qubit_inds = dict(zip(qubits, range(num_qubits)))

    def __copy__(self) -> Layout:
        """
        __copy__ copies the Layout.

        Returns
        -------
        Layout
            _description_
        """
        return Layout(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """
        to_dict return a setup dictonary for the layout.

        Returns
        -------
        Dict[str, Any]
            The setup dictionary of the setup.
        """
        setup = dict()

        setup["name"] = self.name
        setup["description"] = self.description
        setup["interaction_order"] = self.interaction_order

        layout = []
        for node, attrs in self.graph.nodes(data=True):
            node_dict = deepcopy(attrs)
            node_dict["qubit"] = node

            nbr_dict = dict()
            adj_view = self.graph.adj[node]

            for nbr_node, edge_attrs in adj_view.items():
                edge_dir = edge_attrs["direction"]
                nbr_dict[edge_dir] = nbr_node

            node_dict["neighbors"] = nbr_dict
            del node_dict["coords"]

            layout.append(node_dict)
        setup["layout"] = layout
        return setup

    def get_inds(self, qubits: List[str]) -> List[int]:
        """
        get_inds Returns the indices of the qubits.

        Returns
        -------
        List[int]
            The list of qubit indices.
        """
        inds = [self._qubit_inds[qubit] for qubit in qubits]
        return inds

    def get_qubits(self, **conds: Any) -> List[str]:
        """
        get_qubits Return the qubit labels that meet a set of conditions.

        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (Any)
        that each parameter label (str) needs to take.

        Returns
        -------
        List[str]
            The list of qubit indices that meet all conditions.
        """
        if conds:
            node_view = self.graph.nodes(data=True)
            nodes = [node for node, attrs in node_view if valid_attrs(attrs, **conds)]
            return nodes

        nodes = list(self.graph.nodes)
        return nodes

    def get_neighbors(
        self,
        qubits: Union[str, List[str]],
        direction: Optional[str] = None,
        as_pairs: bool = False,
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """
        get_neighbors Returns the list of qubit labels, neighboring specific qubits
        that meet a set of conditions.

        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (Any)
        that each parameter label (str) needs to take.

        Parameters
        ----------
        qubits : str
            The qubit labels, whose neighbors are being considered

        direction : Optional[str]
            The direction along which to consider the neigbors along.

        Returns
        -------
        List[str]
            The list of qubit label, neighboring qubit, that meet the conditions.
        """
        edge_view = self.graph.out_edges(qubits, data=True)

        start_nodes = []
        end_nodes = []
        for start_node, end_node, attrs in edge_view:
            if direction is None or attrs["direction"] == direction:
                start_nodes.append(start_node)
                end_nodes.append(end_node)

        if as_pairs:
            return list(zip(start_nodes, end_nodes))
        return end_nodes

    def index_qubits(self) -> Layout:
        indexed_layout = copy(self)
        nodes = list(self.graph.nodes)

        num_nodes = len(nodes)
        inds = list(range(num_nodes))

        mapping = dict(zip(nodes, inds))
        relabled_graph = nx.relabel_nodes(indexed_layout.graph, mapping)
        for node, ind in zip(nodes, inds):
            relabled_graph.nodes[ind]["name"] = node
        indexed_layout.graph = relabled_graph
        return indexed_layout

    def adjacency_matrix(self) -> DataArray:
        """
        adjacency_matrix Returns the adjaceny matrix corresponding to the layout.

        The layout is encoded as a directed graph, such that there are two edges
        in opposite directions between each pair of neighboring qubits.

        Returns
        -------
        DataArray
            The adjacency matrix
        """
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
        """
        projection_matrix Returns the projection matrix, mapping
        data qubits (defined by a parameter 'role' equal to 'data')
        to ancilla qubits (defined by a parameter 'role' equal to 'anc')
        measuing a given stabilizerr type (defined by a parameter
        'stab_type' equal to stab_type).

        This matrix can be used to project a final set of data-qubit
        measurements to a set of syndromes.

        Parameters
        ----------
        stab_type : str
            The type of the stabilizers that the data qubit measurement
            is being projected to.

        Returns
        -------
        DataArray
            The projection matrix.
        """
        adj_mat = self.adjacency_matrix()

        anc_qubits = self.get_qubits(role="anc", stab_type=stab_type)
        data_qubits = self.get_qubits(role="data")

        proj_mat = adj_mat.sel(from_qubit=data_qubits, to_qubit=anc_qubits)
        return proj_mat.rename(from_qubit="data_qubit", to_qubit="anc_qubit")

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]) -> "Layout":
        """
        from_yaml Loads the layout class from a YAML file.

        The file must define the setup dictionary that initializes
        the layout.

        Parameters
        ----------
        filename : Union[str, Path]
            The pathfile name of the YAML setup file.

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

    def to_yaml(self, filename: Union[str, Path]) -> None:
        """
        to_yaml Saves the layout as a YAML file.

        Parameters
        ----------
        filename : Union[str, Path]
            The pathfile name of the YAML setup file.

        """
        setup = self.to_dict()
        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)

    def param(self, param: str, qubit: str) -> Any:
        """
        param Returns the parameter value of a given qubit

        Parameters
        ----------
        param : str
            The label of the qubit parameter.
        qubit : str
            The label of the qubit that is being queried.

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
            The label of the qubit parameter.
        qubit : str
            The label of the qubit that is being queried.
        value : Any
            The new value of the qubit parameter.
        """
        self.graph.nodes[qubit][param] = value

    def _load_layout(self, setup: Dict[str, Any]) -> None:
        """
        _load_layout Internal function that loads the directed graph from the
        setup dictionary that is provided during initialization.

        Parameters
        ----------
        setup : Dict[str, Any]
            The setup dictionary that must specify the 'layout' list
            of dictionaries, containing the qubit informaiton.

        Raises
        ------
        ValueError
            If there are unlabeled qubits in the any of the layout dictionaries.
        ValueError
            If any qubit label is repeated in the layout list.
        """
        layout = deepcopy(setup.get("layout"))

        for qubit_info in layout:
            qubit = qubit_info.pop("qubit", None)
            if qubit is None:
                raise ValueError("Each qubit in the layout must be labeled.")

            if qubit in self.graph:
                raise ValueError("Qubit label repeated, ensure labels are unique.")

            self.graph.add_node(qubit, **qubit_info)

        for node, attrs in self.graph.nodes(data=True):
            nbr_dict = attrs.pop("neighbors", None)
            for edge_dir, nbr_qubit in nbr_dict.items():
                if nbr_qubit is not None:
                    self.graph.add_edge(node, nbr_qubit, direction=edge_dir)

    def _set_coords(self) -> None:
        """
        set_coords Automatically sets the qubit coordinates. This is used for
        plotting the layout.
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


def valid_attrs(attrs: Dict[str, Any], **conditions: Any) -> bool:
    """
    valid_attrs Checks if the items in attrs match each condition in conditions.
    Both attrs and conditions are dictionaries mapping parameter labels (str)
    to values (Any).

    Parameters
    ----------
    attrs : Dict[str, Any]
        The attribute dictionary.

    Returns
    -------
    bool
        Whether the attributes meet a set of conditions.
    """
    for key, val in conditions.items():
        attr_val = attrs[key]
        if attr_val is None or attr_val != val:
            return False
    return True
