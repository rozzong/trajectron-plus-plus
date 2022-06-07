from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from torch.utils.data import IterableDataset

from .preprocessing import get_node_timestep_data
from .node_type import NodeType
from .scene import Scene


class NodeTypeIterableDataset(IterableDataset):
    """An iterable implementation of the original NodeTypeDataset.

    Parameters
    ----------
    scenes : Iterable[Scene]
        Scenes to iterate over.
    node_type : NodeType
        Node type to build the dataset for.
    edge_types : List[Tuple[NodeType, NodeType]]
        The list of all edge types to take into account when processing the
        node's neighbours.
    include_robot : bool
        True to include the ego-agent, False to not include it.
    min_len_history : int
        The minimum number of timesteps to use in the past.
    max_len_history : int
        The maximum number of timesteps to use in the past.
    min_len_future : int
        The minimum number of timesteps to use in the future.
    max_len_future : int
        The maximum number of timesteps to use in the future.
    standardization_params : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Standardization parameters for state and prediction state structures,
        for all nodes.

        >>> mean, std = standardization_params["state"][node_type]

    attention_radius : Dict[Tuple[NodeType, NodeType], float]
        The dictionary holding all attention radii for all edge types.
    state : Dict[str, Dict[str, List[str]]]
        The dictionary defining all nodes states.
    pred_state : Dict[str, Dict[str, List[str]]]
        The dictionary defining all nodes states to be predicted.
    hyperparams : Mapping[str, Any]
        Model hyperparameters.
    scene_multiplier : bool, default: False
        Whether to use scene multiplication or not.
    node_multiplier : bool, default: False
        Whether to use node multiplication or not.
    edge_addition_filter : Optional[List[float]], default: None

    edge_removal_filter : Optional[List[float]], default: None

    """

    def __init__(
            self,
            scenes: Iterable[Scene],
            node_type: NodeType,
            edge_types: List[Tuple[NodeType, NodeType]],
            include_robot: bool,
            min_len_history: int,
            max_len_history: int,
            min_len_future: int,
            max_len_future: int,
            standardization_params: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
            attention_radius: Dict[Tuple[NodeType, NodeType], float],
            state: Dict[str, Dict[str, List[str]]],
            pred_state: Dict[str, Dict[str, List[str]]],
            hyperparams: Mapping[str, Any],
            scene_multiplier: bool = False,
            node_multiplier: bool = False,
            edge_addition_filter: Optional[List[float]] = None,
            edge_removal_filter: Optional[List[float]] = None,
    ):
        super().__init__()

        # TODO: Get rid of hyperparams completely

        self.scenes = scenes
        self.node_type = node_type
        self.edge_types = edge_types
        self.include_robot = include_robot
        self.min_len_history = min_len_history
        self.max_len_history = max_len_history
        self.min_len_future = min_len_future
        self.max_len_future = max_len_future
        self.standardization_params = standardization_params
        self.attention_radius = attention_radius
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.scene_multiplier = scene_multiplier
        self.node_multiplier = node_multiplier
        self.edge_addition_filter = edge_addition_filter
        self.edge_removal_filter = edge_removal_filter

        # Check edge types
        self.edge_types = [
            edge_type for edge_type in self.edge_types
            if edge_type[0] == self.node_type
        ]

    def __iter__(self):
        for scene in self.scenes:
            present_nodes_dict = scene.present_nodes(
                np.arange(scene.timesteps),
                type=self.node_type,
                min_history_timesteps=self.min_len_history,
                min_future_timesteps=self.min_len_future,
                return_robot=self.include_robot
            )
            scene_multiplier = max(
                1,
                scene.frequency_multiplier * self.scene_multiplier
            )
            for _ in range(scene_multiplier):
                for timestep, nodes in present_nodes_dict.items():
                    scene_graph = scene.get_scene_graph(
                        timestep,
                        self.attention_radius,
                        self.edge_addition_filter,
                        self.edge_removal_filter
                    )
                    for node in nodes:
                        yield from [
                            get_node_timestep_data(
                                scene,
                                timestep,
                                node,
                                self.state,
                                self.pred_state,
                                self.edge_types,
                                self.standardization_params,
                                self.attention_radius,
                                self.max_len_history,
                                self.max_len_future,
                                self.hyperparams,
                                scene_graph
                            )
                        ] * max(
                            1,
                            node.frequency_multiplier * self.node_multiplier
                        )
