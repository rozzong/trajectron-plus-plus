from inspect import stack
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .encoding import MultimodalGenerativeCVAEEncoder
from .latent import DiscreteLatent, Mode
from .decoding import MultimodalGenerativeCVAEDecoder


class MultimodalGenerativeCVAE(nn.Module):
    """Multimodal Generative Conditional Variational Autoencoder.

    Parameters
    ----------
    agent_type : str
        The node type for which to build the model.
    agent_edge_types : List[Tuple[str, str]]
        Edge types involving the node type.
    state : Mapping[str, Mapping[str, Sequence[str]]]
        The dictionary defining all nodes states.
    len_state : int
        The number of elements in the agent state.
    pred_state : Mapping[str, Mapping[str, Sequence[str]]]
        The dictionary defining all nodes states to be predicted.
    len_pred_state : int
        The number of elements in the agent prediction state.
    include_robot : bool
        True to include the ego-agent's future, False to not include it.
    use_edges : bool
        True to use edges in encoding, False to not use them.
    use_maps : bool
        True to use map rasters in encoding, False to not use them.
    config : Mapping[str, Any]
        A mapping describing the architecture of the multimodal generative
        CVAE. The configuration keys are the following:

            encoder:
                history_dim: int
                future_dim: int
                edge_state_dim: int
                edge_state_combine_method: str
                edge_influence_dim: int
                edge_influence_combine_method: str
                p_dropout: float
                use_dynamic_edges: bool
                map_encoder:
                    <agent_type>:
                        heading_state_index: int
                        map_channels: int
                        hidden_channels: List[int]
                        output_size: int
                        masks: List[int]
                        strides: List[int]
                        patch_size: Tuple[int, int, int, int]
                        p_dropout: float
                    ...
            latent:
                n: int
                k: int
                p_z_x_mlp_dim: Optional[int]
                q_z_xy_mlp_dim: Optional[int]
                p_dropout: float
                kl_min: float
            decoder:
                rnn_dim: int
                n_gmm_components: int
                dynamical_model:
                    <agent_type>:
                        type: str
                        use_distribution: bool
                        kwargs: Mapping[str, Any]
                    ...

    len_state_robot : Optional[int]
        The number of elements in the ego-agent state.
    """

    def __init__(
            self,
            agent_type: str,
            agent_edge_types: List[Tuple[str, str]],
            state: Mapping[str, Mapping[str, Sequence[str]]],
            len_state: int,
            pred_state: Mapping[str, Mapping[str, Sequence[str]]],
            len_pred_state: int,
            include_robot: bool,
            use_edges: bool,
            use_maps: bool,
            config: Mapping[str, Any],
            len_state_robot: Optional[int] = None
    ):
        super().__init__()

        self._include_robot = include_robot
        self._use_edges = use_edges
        self._use_maps = use_maps

        self.agent_type = agent_type
        self.agent_edge_types = agent_edge_types
        self.state = state
        self.len_state = len_state
        self.pred_state = pred_state
        self.len_pred_state = len_pred_state
        self.config = config
        self.len_state_robot = len_state_robot

        self.encoder: Optional[MultimodalGenerativeCVAEEncoder] = None
        self.latent: Optional[DiscreteLatent] = None
        self.decoder: Optional[MultimodalGenerativeCVAEDecoder] = None

        self._build()

    def _set_attr_and_propagate(self, value: Any):
        # Get the name of the caller method, i.e. property name
        name = stack()[1][3]
        setattr(self, "_" + name, value)
        for child in (self.encoder, self.decoder):
            if hasattr(child, name):
                setattr(child, name, value)

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

    def _build(self) -> None:
        """Build all submodules.
        """
        self.encoder = MultimodalGenerativeCVAEEncoder(
            self.agent_type,
            self.agent_edge_types,
            self.state,
            self.len_state,
            self.len_pred_state,
            self.include_robot,
            self.use_edges,
            self.use_maps,
            self.config["encoder"]["history_dim"],
            self.config["encoder"]["future_dim"],
            self.config["encoder"]["p_dropout"],
            #self.config["encoder"]["node_state_attention"],
            self.config["encoder"]["edge_state_dim"],
            self.config["encoder"]["edge_state_combine_method"],
            self.config["encoder"]["edge_influence_dim"],
            self.config["encoder"]["edge_influence_combine_method"],
            self.config["encoder"]["use_dynamic_edges"],
            self.config["encoder"]["map_encoder"].get(self.agent_type)
        )
        self.latent = DiscreteLatent(
            self.config["latent"]["n"],
            self.config["latent"]["k"],
            self.encoder.x_size,
            self.encoder.y_size,
            self.config["latent"]["p_z_x_mlp_dim"],
            self.config["latent"]["q_z_xy_mlp_dim"],
            self.config["latent"]["p_dropout"],
            self.config["latent"].get("kl_min")
        )
        self.decoder = MultimodalGenerativeCVAEDecoder(
            self.agent_type,
            self.len_state,
            self.len_pred_state,
            self.include_robot,
            self.encoder.x_size,
            self.latent.z_size,
            self.config["decoder"]["rnn_dim"],
            self.config["decoder"]["n_gmm_components"],
            self.config["decoder"]["dynamical_model"][self.agent_type],
            self.len_state_robot
        )

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
            encoded_robot_future: Optional[torch.Tensor],
            maps: Optional[torch.Tensor],
            prediction_horizon: int,
            n_samples: int = 1,
            mode: Optional[Mode] = None,
            gmm_mode: bool = False
    ):
        # If no label is provided, the model is assumed to be used for
        # prediction
        is_predicting = labels is None

        # Encode the data
        x, y = self.encoder(
            first_timesteps,
            inputs_st,
            labels_st,
            neighbors_data_st,
            neighbors_edge_value,
            encoded_robot_future,
            maps,
        )

        # Override the number of samples during training
        if not is_predicting:
            n_samples = 1

        # Generate the latent variable
        z = self.latent(x, y, n_samples, mode)

        # Select the right distribution
        if is_predicting:
            dist = self.latent.p
            n_components = self.latent.p.n_components
        else:
            dist = self.latent.q
            n_components = self.latent.q.n_components

        # Take the last node history state as the current state
        x_t0 = inputs_st[:, -1]

        # If robot states are available, extract thr current and future ones
        if encoded_robot_future is not None:
            x_r_t0 = robot_future_st[..., 0, :]
            y_r = robot_future_st[..., 1:, :]
        else:
            x_r_t0 = None
            y_r = None

        # Assuming positions account for the two first columns of the state,
        # and velocities for the next two
        # TODO: Make it more flexible
        initial_conditions = {
            "pos": inputs[:, -1, 0:2],
            "vel": inputs[:, -1, 2:4]
        }

        # Run the decoder
        y_dist, preds = self.decoder(
            x,
            x_r_t0,
            y_r,
            x_t0,
            z,
            dist,
            initial_conditions,
            prediction_horizon,
            n_samples,
            n_components,
            gmm_mode
        )

        return y_dist, preds
