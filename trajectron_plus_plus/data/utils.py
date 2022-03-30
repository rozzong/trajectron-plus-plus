from itertools import product
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np

from .node_type import NodeType, NodeTypeEnum


def get_attention_radius(
        node_type_enum: NodeTypeEnum,
        attention_radius_config: Mapping[str, Mapping[str, float]]
) -> Dict[Tuple[NodeType, NodeType], float]:
    """Retrieve the attention radii for each node from a configuration.

    Parameters
    ----------
    node_type_enum : NodeTypeEnum
        The node type enumeration built from agent types.
    attention_radius_config : Mapping[str, Mapping[str, float]]
        The attention radius configuration mapping to use.

    Returns
    -------
   Dict[Tuple[NodeType, NodeType], float]
        Attention radii for each edge.
    """
    attention_radius = {}
    for source, target in product(node_type_enum, repeat=2):
        edge = (source, target)
        radius = attention_radius_config[source.name][target.name]
        attention_radius[edge] = radius

    return attention_radius


def get_standardization_params(
        node_type_enum: NodeTypeEnum,
        config: Mapping[str, Any]
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Retrieve the standardization parameters for each node from a
    configuration.

    Parameters
    ----------
    node_type_enum : NodeTypeEnum
        The node type enumeration built from agent types.
    config : Mapping[str, Mapping[str, float]]
        The configuration mapping to use, containing state and prediction state
        structures, as well as standardization parameters values.

    Returns
    -------
    Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Standardization parameters for state and prediction state structures,
        for all nodes.
    """
    structs = ("state", "pred_state")
    standardization_params = {struct: {} for struct in structs}

    for struct in structs:
        for node_type in node_type_enum:
            standardize_mean_list = []
            standardize_std_list = []
            for entity, dims in config[struct][node_type.name].items():
                for dim in dims:
                    p = config["standardization"][node_type.name][entity][dim]
                    standardize_mean_list.append(p["mean"])
                    standardize_std_list.append(p["std"])
                standardization_params[struct][node_type] = (
                    np.stack(standardize_mean_list),
                    np.stack(standardize_std_list)
                )

    return standardization_params


def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x


def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx
