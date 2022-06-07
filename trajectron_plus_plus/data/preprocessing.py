from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import dill
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .map import GeometricMap


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, Sequence):
        zipped = zip(*batch)
        if len(elem) == 4:
            scene_maps, scene_pts, angles, patch_sizes = zipped
            scene_pts = torch.tensor(scene_pts).float()
            angles = torch.tensor(angles)
            patch_size = patch_sizes[0]
            return GeometricMap.crop_rotate_maps(
                scene_maps,
                patch_size,
                scene_pts,
                angles
            )
        else:
            return [collate(samples) for samples in zipped]
    elif isinstance(elem, Mapping):
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return neighbor_dict
        # return dill.dumps(neighbor_dict) \
        #     if torch.utils.data.get_worker_info() \
        #     else neighbor_dict
    else:
        return default_collate(batch)


def standardize(
        array: np.ndarray,
        state_node_standardization_params: Tuple[np.ndarray, np.ndarray],
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
) -> np.ndarray:
    """Standardize data of a state or prediction state.

    Parameters
    ----------
    array : np.ndarray
        The data to standardize.
    state_node_standardization_params : Tuple[np.ndarray, np.ndarray]
        The standardization parameters for either the state or prediction
        state of a node type.

        >>> mean, std = state_node_standardization_params

    mean : Optional[np.ndarray]
        If provided, replaces the default mean.
    std : Optional[np.ndarray]
        If provided, replaces the default standard deviation.

    Returns
    -------
    np.ndarray
        The standardized data.
    """
    mean = mean if mean is not None else state_node_standardization_params[0]
    std = std if std is not None else state_node_standardization_params[1]

    return np.where(
        np.isnan(array),
        np.array(np.nan),
        (array - mean) / std
    )


def unstandardize(
        array: np.ndarray,
        state_node_standardization_params: Tuple[np.ndarray, np.ndarray],
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
) -> np.ndarray:
    """Unstandardize data of a state or prediction state.

    Parameters
    ----------
    array : np.ndarray
        The data to unstandardize.
    state_node_standardization_params : Tuple[np.ndarray, np.ndarray]
        The standardization parameters for either the state or prediction
        state of a node type.

        >>> mean, std = state_node_standardization_params

    mean : Optional[np.ndarray]
        If provided, replaces the default mean.
    std : Optional[np.ndarray]
        If provided, replaces the default standard deviation.

    Returns
    -------
    np.ndarray
        The unstandardized data.
    """
    mean = mean if mean is not None else state_node_standardization_params[0]
    std = std if std is not None else state_node_standardization_params[1]

    return array * std + mean


def get_relative_robot_traj(
        state: Dict[str, Dict[str, List[str]]],
        standardization_params: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        attention_radius: Dict[Tuple["NodeType", "NodeType"], float],
        node_traj: np.ndarray,
        robot_traj: np.ndarray,
        node_type: "NodeType",
        robot_type: "NodeType"
) -> torch.Tensor:
    """Get the standardized trajectory of the robot relatively to a given node.

    Parameters
    ----------
    state : Dict[str, Dict[str, List[str]]]
        The state dictionary for different agent types.
    standardization_params : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Standardization parameters for state and prediction state structures,
        for all nodes.

        >>> mean, std = standardization_params["state"][node_type]

    attention_radius : Dict[Tuple[NodeType, NodeType], float]
        The dictionary holding all attention radii for all edge types.
    node_traj : np.ndarray
        The other agent trajectory.
    robot_traj : np.ndarray
        The robot trajectory.
    node_type : NodeType
        The node type of the other agent.
    robot_type : NodeType
        The node type of the robot.

    Returns
    -------
    torch.Tensor
        The relative robot trajectory.
    """
    mean, std = standardization_params["state"][robot_type]

    # Use only the available motion as mean
    if node_type != robot_type:
        node_state_cols = [
            (q, a) for q in state[node_type] for a in state[node_type][q]
        ]
        robot_state_cols = [
            (q, a) for q in state[robot_type] for a in state[robot_type][q]
        ]
        cols = [robot_state_cols.index(q_a) for q_a in node_state_cols]
        mean = np.tile(mean, (len(robot_traj), 1))
        mean[:, cols] = node_traj
    else:
        mean = node_traj

    # Use the attention radius as standard deviation
    std[0:2] = attention_radius[(node_type, robot_type)]

    # Standardize the motion
    robot_traj_st = standardize(
        robot_traj,
        standardization_params["state"][robot_type],
        mean=mean,
        std=std
    )

    return torch.tensor(robot_traj_st, dtype=torch.float)


def get_node_timestep_data(
        scene: "Scene",
        timestep: int,
        node: "Node",
        state: Dict[str, Dict[str, List[str]]],
        pred_state: Dict[str, Dict[str, List[str]]],
        edge_types: List[Tuple["NodeType", "NodeType"]],
        standardization_params: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        attention_radius: Dict[Tuple["NodeType", "NodeType"], float],
        max_ht: int,
        max_ft: int,
        hyperparams: Mapping[str, Any],
        scene_graph: Optional[Union["SceneGraph", "SparseSceneGraph"]] = None,
):
    """Pre-process the data for a single batch element. The function returns
    the node state over time for a specific timestep in a specific scene, as
    well as the neighbour data for it.

    Parameters
    ----------
    scene : Scene
        The scene to get the data from.
    timestep : int
        The timestep to get the data from in scene.
    node : Node
        The node to get the data from.
    state : Dict[str, Dict[str, List[str]]]
        The state dictionary for different agent types.
    pred_state : Dict[str, Dict[str, List[str]]]
        The prediction state dictionary for different agent types.
    edge_types : List[Tuple["NodeType", "NodeType"]]
        The list of all edge types to take into account when processing the
        node's neighbours.
    standardization_params : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Standardization parameters for state and prediction state structures,
        for all nodes.

        >>> mean, std = standardization_params["state"][node_type]

    attention_radius : Dict[Tuple[NodeType, NodeType], float]
        The dictionary holding all attention radii for all edge types.
    max_ht : int
        The maximum amount of timesteps to take in the history.
    max_ft : int
        The maximum amount of timesteps to take in the future.
    hyperparams : Mapping[str, Any]
        Model hyperparameters
    scene_graph : Optional[Union[SceneGraph, SparseSceneGraph]]
        If the scene graph was already computed for this scene and timestep,
        it can be passed here.

    Returns
    -------
    tuple
        A single data batch element.
    """
    # Node
    timestep_range_x = np.array([timestep - max_ht, timestep])
    timestep_range_y = np.array([timestep + 1, timestep + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(timestep)).clip(0)

    _, std = standardization_params["state"][node.type]
    std[0:2] = attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]

    x_st = standardize(
        x,
        standardization_params["state"][node.type],
        mean=rel_state,
        std=std
    )

    # If we predict position, we do it relatively to current position
    y_st = standardize(
        y,
        standardization_params["pred_state"][node.type],
        mean=rel_state[0:2]
        if list(pred_state[node.type].keys())[0] == "position"
        else None
    )

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None

    if hyperparams["parameters"]["use_edges"]:
        neighbors_data_st = dict()
        neighbors_edge_value = dict()

        # Retrieve the scene graph
        if scene_graph is None:
            scene_graph = scene.get_scene_graph(
                timestep,
                attention_radius,
                hyperparams["parameters"]["edge_addition_filter"],
                hyperparams["parameters"]["edge_removal_filter"]
            )

        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()

            # Get all nodes which are connected to the current node for the
            # current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams["architecture"]["encoder"]["use_dynamic_edges"]:
                # Get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(
                    scene_graph.get_edge_scaling(node),
                    dtype=torch.float
                )
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(
                    np.array([timestep - max_ht, timestep]),
                    state[connected_node.type],
                    padding=0.0
                )

                # Make state relative to node where neighbor and node have the
                # same state
                _, std = standardization_params["state"][connected_node.type]
                std[0:2] = attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = standardize(
                    neighbor_state_np,
                    standardization_params["state"][connected_node.type],
                    mean=rel_state,
                    std=std
                )
                neighbor_state = torch.tensor(
                    neighbor_state_np_st,
                    dtype=torch.float
                )
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    timestep_range_r = np.array([timestep, timestep + max_ft])

    if hyperparams["parameters"]["include_robot"]:
        x_node = node.get(timestep_range_r, state[node.type])
        robot = scene.get_node_by_id(scene.non_aug_scene.robot.id) \
            if scene.non_aug_scene is not None \
            else scene.robot
        robot_traj = robot.get(
            timestep_range_r,
            state[robot.type],
            padding=np.nan
        )
        robot_traj_st_t = get_relative_robot_traj(
            state,
            standardization_params,
            attention_radius,
            x_node,
            robot_traj,
            node.type,
            robot.type
        )
        robot_traj_st_t[torch.isnan(robot_traj_st_t)] = 0.0

    # Map
    map_tuple = None
    if hyperparams["parameters"]["use_maps"]:
        if node.type in hyperparams["architecture"]["encoder"]["map_encoder"]:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(
                    np.array([timestep]),
                    state[node.type]
                )
            me_hyp = hyperparams["architecture"]["encoder"]["map_encoder"][node.type]
            if "heading_state_index" in me_hyp:
                heading_state_index = me_hyp["heading_state_index"]
                # We have to rotate the map in the opposite direction of the
                # agent to match them
                if isinstance(heading_state_index, list):
                    # infer from velocity or heading vector
                    heading_angle = -np.arctan2(
                        x[-1, heading_state_index[1]],
                        x[-1, heading_state_index[0]]
                    ) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = me_hyp["patch_size"]
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    return (
        first_history_index,
        x_t,
        x_st_t,
        y_t,
        y_st_t,
        neighbors_data_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map_tuple
    )


def get_timesteps_data(
        scene: "Scene",
        timesteps: np.ndarray,
        node_type: "NodeType",
        state: Mapping[str, Mapping[str, Sequence[str]]],
        pred_state: Mapping[str, Mapping[str, Sequence[str]]],
        edge_types: Sequence[Tuple["NodeType", "NodeType"]],
        standardization_params: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]],
        attention_radius: Mapping[Tuple["NodeType", "NodeType"], float],
        min_ht: int,
        max_ht: int,
        max_ft: int,
        hyperparams: Mapping[str, Any],
) -> Union[list, List["Node"], List[int], None]:
    """Pre-process a data batch. The function returns data for all nodes in a
    given scene and the given timesteps.

    scene : Scene
        The scene to get the data from.
    timestep : int
        The timestep to get the data from in scene.
    node : Node
        The node to get the data from.
    state : Dict[str, Dict[str, List[str]]]
        The state dictionary for different agent types.
    pred_state : Dict[str, Dict[str, List[str]]]
        The prediction state dictionary for different agent types.
    edge_types : List[Tuple["NodeType", "NodeType"]]
        The list of all edge types to take into account when processing the
        node's neighbours.
    standardization_params : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Standardization parameters for state and prediction state structures,
        for all nodes.

        >>> mean, std = standardization_params["state"][node_type]

    attention_radius : Dict[Tuple[NodeType, NodeType], float]
        The dictionary holding all attention radii for all edge types.
    max_ht : int
        The maximum amount of timesteps to take in the history.
    max_ft : int
        The maximum amount of timesteps to take in the future.
    hyperparams : Mapping[str, Any]
        The model hyperparameters.

    Returns
    -------
    tuple
        A data batch.
    """
    nodes_per_ts = scene.present_nodes(
        timesteps,
        type=node_type,
        min_history_timesteps=min_ht,
        min_future_timesteps=max_ft,
        return_robot=not hyperparams["parameters"]["include_robot"]
    )

    batch = list()
    nodes = list()
    out_timesteps = list()

    for timestep, present_nodes in nodes_per_ts.items():
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            batch.append(
                get_node_timestep_data(
                    scene,
                    timestep,
                    node,
                    state,
                    pred_state,
                    edge_types,
                    standardization_params,
                    attention_radius,
                    max_ht,
                    max_ft,
                    hyperparams,
                )
            )

    return (collate(batch), nodes, out_timesteps) \
        if len(out_timesteps) \
        else None
