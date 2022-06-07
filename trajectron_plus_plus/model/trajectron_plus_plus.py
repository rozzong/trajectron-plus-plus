from inspect import stack
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .mgcvae import MultimodalGenerativeCVAE
from .mgcvae.encoding import NodeFutureEncoder
from .mgcvae.latent import Mode
from ..data import restore
from ..misc import check_if_original, convert_state_dict


class TrajectronPlusPlus(nn.Module):
    """Trajectron++

    A PyTorch implementation of the Trajectron++ model, introduced by Salzmann
    et al. [1].

    Parameters
    ----------
    agent_types : Sequence[str]
        Node types to create MGCVAE models for.
    state : Mapping[str, Mapping[str, Sequence[str]]]
        The state structures for different agent types.
    pred_state : Mapping[str, Mapping[str, Sequence[str]]]
        The prediction state structures for different agent types.
    include_robot : bool
        If True, encode and use planned robot trajectories.
    use_edges : bool
        If True, encode and use edges between graph nodes (i.e. agents).
    use_maps : bool
        If True, encode and use maps.
    config : Mapping[str, Any]
        The configuration to use to build the model.

    References
    ----------
    [1] Tim Salzmann, Boris Ivanovic, Punarjay Chakravarty, and Marco Pavone.
        "Trajectron++: Dynamically-feasible trajectory forecasting with
        heterogeneous data." In European Conference on Computer Vision, pp.
        683-700. Springer, Cham, 2020.
    """

    def __init__(
            self,
            agent_types: Sequence[str],
            robot_agent_type: str,
            state: Mapping[str, Mapping[str, Sequence[str]]],
            pred_state: Mapping[str, Mapping[str, Sequence[str]]],
            include_robot: bool,
            use_edges: bool,
            use_maps: bool,
            config: Mapping[str, Any]
    ):
        super().__init__()

        self._include_robot = include_robot
        self._use_edges = use_edges
        self._use_maps = use_maps

        self.agent_types = agent_types
        self.robot_agent_type = robot_agent_type
        self.state = state
        self.pred_state = pred_state
        self.config = config

        self.edge_types = None
        self.len_state = None
        self.len_pred_state = None
        self.robot_future_encoder: Optional[NodeFutureEncoder] = None
        self.node_models: nn.ModuleDict[str, MultimodalGenerativeCVAE] = None

        self._build()

    def _set_attr_and_propagate(self, value: Any):
        # Get the name of the caller method, i.e. property name
        name = stack()[1][3]
        setattr(self, "_" + name, value)
        for model in self.node_models.values():
            setattr(model, name, value)

    @property
    def include_robot(self) -> bool:
        return self._include_robot

    @include_robot.setter
    def include_robot(self, value: bool):
        self._set_attr_and_propagate(value)

    @property
    def use_edges(self) -> bool:
        return self._use_edges

    @use_edges.setter
    def use_edges(self, value: bool):
        self._set_attr_and_propagate(value)

    @property
    def use_maps(self) -> bool:
        return self._use_maps

    @use_maps.setter
    def use_maps(self, value: bool):
        self._set_attr_and_propagate(value)

    def _build(self):
        # Compute states lengths
        self.len_state = {}
        self.len_pred_state = {}
        for agent_type in self.agent_types:
            self.len_state[agent_type] = sum(
                [len(axes) for axes in self.state[agent_type].values()]
            )
            self.len_pred_state[agent_type] = sum(
                [len(axes) for axes in self.pred_state[agent_type].values()]
            )

        # Create a unique robot encoder, if asked
        if self.include_robot:
            self.robot_future_encoder = NodeFutureEncoder(
                self.len_state[self.robot_agent_type],
                self.config["encoder"]["future_dim"],
                self.len_state[self.robot_agent_type],
                self.config["encoder"]["p_dropout"]
            )

        # Compute all possible edge types
        self.edge_types = list(product(self.agent_types, repeat=2))

        # Create a CVAE model for each node type
        self.node_models = nn.ModuleDict()
        for agent_type in self.agent_types:
            # Filter possible edge types for the current node
            agent_edge_types = [
                edge_type for edge_type in self.edge_types
                if edge_type[0] == agent_type
            ]
            self.node_models[agent_type] = MultimodalGenerativeCVAE(
                agent_type,
                agent_edge_types,
                self.state,
                self.len_state[agent_type],
                self.pred_state,
                self.len_pred_state[agent_type],
                self.include_robot,
                self.use_edges,
                self.use_maps,
                self.config,
                self.len_state.get(self.robot_agent_type)
            )

    def load_state_dict(
            self,
            state_dict: "OrderedDict[str, Tensor]",
            strict: bool = True
    ) -> Tuple[List[str], List[str]]:
        # Convert the state dict to this implementation's format, if needed
        if check_if_original(state_dict):
            state_dict = convert_state_dict(state_dict)

        return super().load_state_dict(state_dict, strict)

    @staticmethod
    def get_agent_type(inputs: Union[list, dict]) -> str:
        """Get the agent type related to some batched data.

        Parameters
        ----------
        inputs : Union[list, dict]
            A whole data batch, or a batch neighbors dictionary.

        Returns
        -------
        str
            The agent type for the batched data.
        """
        neighbors_dict = inputs[5] if isinstance(inputs, list) else inputs

        return list(restore(neighbors_dict))[0][0].name

    def forward(
            self,
            first_timesteps: torch.Tensor,
            inputs: torch.Tensor,
            inputs_st: torch.Tensor,
            labels: Optional[torch.Tensor],
            labels_st: Optional[torch.Tensor],
            neighbors_data_st: Dict[Tuple[str, str], List[torch.Tensor]],
            neighbors_edge_value: Dict[Tuple[str, str], List[torch.Tensor]],
            robot_future_st: Optional[torch.Tensor],
            maps: Optional[torch.Tensor],
            prediction_horizon: int,
            n_samples: int = 1,
            mode: Optional[Mode] = None,
            gmm_mode: bool = False
    ) -> Tuple[Dict[str, "GMM2D"], Dict[str, torch.Tensor]]:
        # Identify the agent type of the batch
        agent_type = self.get_agent_type(neighbors_data_st)

        # Encode the robot's future plans, if asked
        encoded_robot_future = self.robot_future_encoder(
            robot_future_st[..., 0, :],
            robot_future_st[..., 1:, :]
        ) if robot_future_st is not None and self.include_robot else None

        y_dist, preds = self.node_models[agent_type](
            first_timesteps,
            inputs,
            inputs_st,
            labels,
            labels_st,
            neighbors_data_st,
            neighbors_edge_value,
            robot_future_st,
            encoded_robot_future,
            maps,
            prediction_horizon,
            n_samples,
            mode,
            gmm_mode
        )

        return y_dist, preds
