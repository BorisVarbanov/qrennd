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

    def __init__(self, setup: Dict[str, Dict[str, Any]]) -> None:
        """
        __init__ Initiailizes the layout.

        Parameters
        ----------
        setup : Dict[str, Dict[str, Any]]
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

        self.name = setup.get("name", "")
        self.description = setup.get("description", "")

        self.int_order = setup.get("int_order", None)

        self.graph = nx.DiGraph()
        self._load_layout(setup)
        self._set_coords()

    def get_inds(self, **conds: Dict[str, Any]) -> List[int]:
        """
        get_inds Return the qubit indices that meet a set of conditions.

        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (Any)
        that each parameter label (str) needs to take.

        Returns
        -------
        List[int]
            The list of qubit indices that meet all conditions.
        """
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
        get_qubits Return the qubit labels that meet a set of conditions.

        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (Any)
        that each parameter label (str) needs to take.

        Returns
        -------
        List[str]
            The list of qubit labels that meet all conditions.
        """
        if conds:
            node_view = self.graph.nodes(data=True)
            nodes = [node for node, attrs in node_view if valid_attrs(attrs, **conds)]
            return nodes

        nodes = list(self.graph.nodes)
        return nodes

    def get_neighbors(self, qubit: str, **conds: Dict[str, Any]) -> List[str]:
        """
        get_neighbors Returns the list of qubit labels, neighboring a specific qubit
        that meet a set of conditions.

        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (Any)
        that each parameter label (str) needs to take.

        Parameters
        ----------
        qubit : str
            The qubit label, whose neighbors are being considered

        Returns
        -------
        List[str]
            The list of qubit label, neighboring qubit, that meet the conditions.
        """
        nbr_nodes = list(self.graph.adj[qubit])

        if conds:
            nodes = [n for n in nbr_nodes if valid_attrs(self.graph[n], **conds)]
            return nodes
        return nbr_nodes

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
        s
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
    def from_yaml(cls, filename: str) -> "Layout":
        """
        from_yaml Loads the layout class from a YAML file.

        The file must define the setup dictionary that initializes
        the layout.

        Parameters
        ----------
        filename : str
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


def valid_attrs(attrs: Dict[str, Any], **conditions: Dict[str, Any]) -> bool:
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
