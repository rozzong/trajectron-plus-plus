from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from .node_encoding import NodeHistoryEncoder, NodeFutureEncoder
from .edge_state_encoding import EdgeStateEncoder, ReducingEdgeStateEncoder
from .map_encoding import CNNMapEncoder
from .edge_influence_encoding import EdgeInfluenceEncoder, \
    ReducingEdgeInfluenceEncoder, BiRNNEdgeInfluenceEncoder, \
    AttentionEdgeInfluenceEncoder


class MultimodalGenerativeCVAEEncoder(nn.Module):

    def __init__(
            self,
            agent_type,
            edge_types,
            state: Mapping[str, Mapping[str, Sequence[str]]],
            len_state: int,
            len_pred_state: int,
            include_robot: bool,
            use_edges: bool,
            use_maps: bool,
            history_dim: int,
            future_dim: int,
            p_dropout: float = 0.5,
            edge_state_dim: Optional[int] = None,
            edge_state_combine_method: Optional[str] = None,
            edge_influence_dim: Optional[int] = None,
            edge_influence_combine_method: Optional[str] = None,
            use_dynamic_edges: Optional[bool] = None,
            map_encoder_config: Optional[Mapping[str, Any]] = None
    ):
        super().__init__()

        self._include_robot = include_robot
        self._use_edges = use_edges
        self._use_maps = use_maps

        self.agent_type = agent_type
        self.edge_types = edge_types
        self.state = state
        self.len_state = len_state
        self.len_pred_state = len_pred_state
        self.history_dim = history_dim
        self.future_dim = future_dim
        self.p_dropout = p_dropout
        self.edge_state_dim = edge_state_dim
        self.edge_state_combine_method = edge_state_combine_method
        self.edge_influence_dim = edge_influence_dim
        self.edge_influence_combine_method = edge_influence_combine_method
        self.use_dynamic_edges = use_dynamic_edges
        self.map_encoder_config = map_encoder_config

        self.node_history_encoder: Optional[NodeHistoryEncoder] = None
        self.node_future_encoder: Optional[NodeFutureEncoder] = None
        self.edge_influence_encoder: Optional[EdgeInfluenceEncoder] = None
        self.edge_state_encoders: Optional[
            nn.ModuleDict[tuple, EdgeStateEncoder]
        ] = None
        self.map_encoder: Optional[CNNMapEncoder] = None

        self._build()

    @property
    def include_robot(self) -> bool:
        return self._include_robot

    @include_robot.setter
    def include_robot(self, value: bool):
        self._include_robot = value

    @property
    def use_edges(self) -> bool:
        return self._use_edges

    @use_edges.setter
    def use_edges(self, value: bool):
        self._use_edges = value

    @property
    def use_maps(self) -> bool:
        return self._use_maps

    @use_maps.setter
    def use_maps(self, value: bool):
        self._use_maps = value

    @property
    def x_size(self) -> int:
        """Size of the encoded input. This encoded tensor is the concatenation
        of the following tensors, depending on the uses:
        - encoded history
        - encoded edges
        - encoded future
        - encoded map
        """
        dummy_encoded_edges = [
            torch.zeros((1, self.history_dim)) for _ in self.edge_types
        ]
        dummy_encoded_history = torch.zeros((1, self.history_dim))
        dummy_batch_size = 1

        dims = (
            self.history_dim,
            self.edge_influence_encoder(
                dummy_encoded_edges,
                dummy_encoded_history,
                dummy_batch_size
            ).shape[-1] * self.use_edges
            if self.edge_influence_encoder is not None
            else 0,
            4 * self.future_dim * self.include_robot,
            (self.map_encoder.fc.out_features * self.use_maps)
            if self.map_encoder is not None
            else 0
        )

        return sum(dims)

    @property
    def y_size(self) -> int:
        """Size of the encoded label.
        """
        return 4 * self.future_dim

    def _build(self):
        # Build the node past encoder
        self.node_history_encoder = NodeHistoryEncoder(
            self.len_state,
            self.history_dim,
            self.p_dropout
        )

        # Build the node future encoder
        self.node_future_encoder = NodeFutureEncoder(
            self.len_pred_state,
            self.future_dim,
            self.len_state,
            self.p_dropout
        )

        # Build the edge influence and edge encoders, if asked
        if self.use_edges:
            # Build the edge influence encoder
            if self.edge_influence_combine_method in ("sum", "max", "mean"):
                self.edge_influence_encoder = ReducingEdgeInfluenceEncoder(
                    self.edge_influence_combine_method
                )
            elif self.edge_influence_combine_method == "bi-rnn":
                self.edge_influence_encoder = BiRNNEdgeInfluenceEncoder(
                    self.edge_state_dim,
                    self.edge_influence_dim,
                    self.p_dropout
                )
            elif self.edge_influence_combine_method == "attention":
                self.edge_influence_encoder = AttentionEdgeInfluenceEncoder(
                    self.edge_influence_dim,
                    self.history_dim,
                    self.p_dropout
                )
            else:
                raise ValueError

            # Build the edge state encoders
            self.edge_state_encoders = nn.ModuleDict()
            for edge_type in self.edge_types:
                edge_type_key = " -> ".join(edge_type)
                len_neighbor_state = sum(
                    [len(axes) for axes in self.state[edge_type[1]].values()]
                )
                edge_encoder_input_size = self.len_state + len_neighbor_state
                if self.edge_state_combine_method in ("sum", "max", "mean"):
                    self.edge_state_encoders[edge_type_key] = \
                        ReducingEdgeStateEncoder(
                            edge_type,
                            edge_encoder_input_size,
                            self.edge_state_dim,
                            self.p_dropout,
                            self.use_dynamic_edges,
                            self.edge_state_combine_method,
                        )
                elif self.edge_state_combine_method == "pointnet":
                    edge_encoder_input_size += self.len_state
                    raise NotImplementedError
                elif self.edge_state_combine_method == "attention":
                    raise NotImplementedError
                else:
                    raise ValueError

        # Build the map encoder, if asked
        if self.map_encoder_config is not None and self.use_maps:
            self.map_encoder = CNNMapEncoder(
                self.map_encoder_config["map_channels"],
                self.map_encoder_config["hidden_channels"],
                self.map_encoder_config["output_size"],
                self.map_encoder_config["masks"],
                self.map_encoder_config["strides"],
                self.map_encoder_config["patch_size"],
                self.map_encoder_config["p_dropout"]
            )

    def forward(
            self,
            first_timesteps,
            inputs_st: torch.Tensor,
            labels_st: Optional[torch.Tensor] = None,
            neighbors_data_st: Optional[torch.Tensor] = None,
            neighbors_edge_value: Optional[torch.Tensor] = None,
            encoded_robot_future: Optional[torch.Tensor] = None,
            maps: Optional[torch.Tensor] = None,
    ):
        batch_size = len(inputs_st)
        is_predicting = labels_st is None

        x_list = []
        inputs_present_st = inputs_st[:, -1]

        # Encode the history
        encoded_history = self.node_history_encoder(
            inputs_st,
            first_timesteps
        )
        x_list.append(encoded_history)

        # Include the future robot plans
        if self.include_robot:
            x_list.append(encoded_robot_future)

        # Encode the future
        y = self.node_future_encoder(
            inputs_present_st,
            labels_st
        ) if not is_predicting else None

        # Encode edge states and influence, if asked
        if self.use_edges:
            encoded_edges = []
            for edge_type in self.edge_types:
                edge_type_key = " -> ".join(edge_type)
                encoded_edges.append(
                    self.edge_state_encoders[edge_type_key](
                        inputs_st,
                        neighbors_data_st[edge_type],
                        neighbors_edge_value[edge_type],
                        first_timesteps,
                        self.state
                    )
                )

            encoded_edge_influence = self.edge_influence_encoder(
                encoded_edges,
                encoded_history,
                batch_size
            )

            x_list.append(encoded_edge_influence)

        # Encode the map, if asked
        if self.use_maps and maps is not None:
            scaled_maps = (maps * 2 / 255) - 1
            encoded_maps = self.map_encoder(scaled_maps)
            x_list.append(encoded_maps)

        # Concatenate encoded data
        x = torch.cat(x_list, dim=1)

        return x, y
