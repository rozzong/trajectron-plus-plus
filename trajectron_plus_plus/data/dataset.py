from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from torch.utils.data import IterableDataset

from .preprocessing import get_node_timestep_data


class NodeTypeIterableDataset(IterableDataset):
    """An iterable implementation of the original NodeTypeDataset.

    Parameters
    ----------
    scenes : Iterable[Scene]
        Scenes to iterate over.
    node_type : trajectron.environment.node_type.NodeType
        Node type to build the dataset for.
    edge_types : List[Tuple["NodeType", "NodeType"]]
        The list of all edge types to take into account when processing the
        node's neighbours.
    standardization_params : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Standardization parameters for state and prediction state structures,
        for all nodes.

        >>> mean, std = standardization_params["state"][node_type]

    attention_radius : Dict[Tuple[NodeType, NodeType], float]
        The dictionary holding all attention radii for all edge types.
    hyperparams : Mapping[str, Any]
        Model hyperparameters.
    """

    def __init__(
            self,
            scenes: Iterable["Scene"],
            node_type: "NodeType",
            edge_types: List[Tuple["NodeType", "NodeType"]],
            standardization_params: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
            attention_radius: Dict[Tuple["NodeType", "NodeType"], float],
            hyperparams: Mapping[str, Any],
            scene_multiplier: int = 1,
            node_multiplier: int = 1,
    ):
        super().__init__()

        self.scenes = scenes
        self.node_type = node_type
        self.edge_types = edge_types
        self.standardization_params = standardization_params
        self.attention_radius = attention_radius
        self.hyperparams = hyperparams
        self.scene_multiplier = scene_multiplier
        self.node_multiplier = node_multiplier

        # Check edge types
        self.edge_types = [
            edge_type for edge_type in self.edge_types
            if edge_type[0] == self.node_type
        ]

        # Compute the multiplier, assuming it is constant
        self.multiplier = scene_multiplier * node_multiplier

    def __iter__(self):
        for scene in self.scenes:
            present_nodes_dict = scene.present_nodes(
                np.arange(scene.timesteps),
                type=self.node_type,
                min_history_timesteps=self.hyperparams["minimum_history_length"],
                min_future_timesteps=self.hyperparams["prediction_horizon"],
                return_robot=self.hyperparams["include_robot"]
            )
            for timestep, nodes in present_nodes_dict.items():
                scene_graph = scene.get_scene_graph(
                    timestep,
                    self.attention_radius,
                    self.hyperparams["edge_addition_filter"],
                    self.hyperparams["edge_removal_filter"]
                )
                for node in nodes:
                    yield from [
                        get_node_timestep_data(
                            scene,
                            timestep,
                            node,
                            self.hyperparams["state"],
                            self.hyperparams["pred_state"],
                            self.edge_types,
                            self.standardization_params,
                            self.attention_radius,
                            self.hyperparams["maximum_history_length"],
                            self.hyperparams["prediction_horizon"],
                            self.hyperparams,
                            scene_graph
                        )
                    ] * self.multiplier
