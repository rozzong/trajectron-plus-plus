from typing import List


_robot_future_encoder_mapping = {
    "robot_future_encoder.weight_ih_l0": "robot_future_encoder.lstm.weight_ih_l0",
    "robot_future_encoder.weight_hh_l0": "robot_future_encoder.lstm.weight_hh_l0",
    "robot_future_encoder.bias_ih_l0": "robot_future_encoder.lstm.bias_ih_l0",
    "robot_future_encoder.bias_hh_l0": "robot_future_encoder.lstm.bias_hh_l0",
    "robot_future_encoder.weight_ih_l0_reverse": "robot_future_encoder.lstm.weight_ih_l0_reverse",
    "robot_future_encoder.weight_hh_l0_reverse": "robot_future_encoder.lstm.weight_hh_l0_reverse",
    "robot_future_encoder.bias_ih_l0_reverse": "robot_future_encoder.lstm.bias_ih_l0_reverse",
    "robot_future_encoder.bias_hh_l0_reverse": "robot_future_encoder.lstm.bias_hh_l0_reverse",
    "robot_future_encoder/initial_h.weight": "robot_future_encoder.state_initializers.h.weight",
    "robot_future_encoder/initial_h.bias": "robot_future_encoder.state_initializers.h.bias",
    "robot_future_encoder/initial_c.weight": "robot_future_encoder.state_initializers.c.weight",
    "robot_future_encoder/initial_c.bias": "robot_future_encoder.state_initializers.c.bias",

}


def _node_and_map_encoder_mapping(agent_type: str) -> dict:
    return {
        f"{agent_type}/node_history_encoder.weight_ih_l0": f"node_models.{agent_type}.encoder.node_history_encoder.lstm.weight_ih_l0",
        f"{agent_type}/node_history_encoder.weight_hh_l0": f"node_models.{agent_type}.encoder.node_history_encoder.lstm.weight_hh_l0",
        f"{agent_type}/node_history_encoder.bias_ih_l0": f"node_models.{agent_type}.encoder.node_history_encoder.lstm.bias_ih_l0",
        f"{agent_type}/node_history_encoder.bias_hh_l0": f"node_models.{agent_type}.encoder.node_history_encoder.lstm.bias_hh_l0",
        f"{agent_type}/node_future_encoder.weight_ih_l0": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.weight_ih_l0",
        f"{agent_type}/node_future_encoder.weight_hh_l0": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.weight_hh_l0",
        f"{agent_type}/node_future_encoder.bias_ih_l0": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.bias_ih_l0",
        f"{agent_type}/node_future_encoder.bias_hh_l0": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.bias_hh_l0",
        f"{agent_type}/node_future_encoder.weight_ih_l0_reverse": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.weight_ih_l0_reverse",
        f"{agent_type}/node_future_encoder.weight_hh_l0_reverse": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.weight_hh_l0_reverse",
        f"{agent_type}/node_future_encoder.bias_ih_l0_reverse": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.bias_ih_l0_reverse",
        f"{agent_type}/node_future_encoder.bias_hh_l0_reverse": f"node_models.{agent_type}.encoder.node_future_encoder.lstm.bias_hh_l0_reverse",
        f"{agent_type}/node_future_encoder/initial_h.weight": f"node_models.{agent_type}.encoder.node_future_encoder.state_initializers.h.weight",
        f"{agent_type}/node_future_encoder/initial_h.bias": f"node_models.{agent_type}.encoder.node_future_encoder.state_initializers.h.bias",
        f"{agent_type}/node_future_encoder/initial_c.weight": f"node_models.{agent_type}.encoder.node_future_encoder.state_initializers.c.weight",
        f"{agent_type}/node_future_encoder/initial_c.bias": f"node_models.{agent_type}.encoder.node_future_encoder.state_initializers.c.bias",
        f"{agent_type}/edge_influence_encoder.w1.weight": f"node_models.{agent_type}.encoder.edge_influence_encoder.combiner.w1.weight",
        f"{agent_type}/edge_influence_encoder.w2.weight": f"node_models.{agent_type}.encoder.edge_influence_encoder.combiner.w2.weight",
        f"{agent_type}/edge_influence_encoder.v.weight": f"node_models.{agent_type}.encoder.edge_influence_encoder.combiner.v.weight",
        f"{agent_type}/map_encoder.convs.0.weight": f"node_models.{agent_type}.encoder.map_encoder.convolutions.0.weight",
        f"{agent_type}/map_encoder.convs.0.bias": f"node_models.{agent_type}.encoder.map_encoder.convolutions.0.bias",
        f"{agent_type}/map_encoder.convs.1.weight": f"node_models.{agent_type}.encoder.map_encoder.convolutions.1.weight",
        f"{agent_type}/map_encoder.convs.1.bias": f"node_models.{agent_type}.encoder.map_encoder.convolutions.1.bias",
        f"{agent_type}/map_encoder.convs.2.weight": f"node_models.{agent_type}.encoder.map_encoder.convolutions.2.weight",
        f"{agent_type}/map_encoder.convs.2.bias": f"node_models.{agent_type}.encoder.map_encoder.convolutions.2.bias",
        f"{agent_type}/map_encoder.convs.3.weight": f"node_models.{agent_type}.encoder.map_encoder.convolutions.3.weight",
        f"{agent_type}/map_encoder.convs.3.bias": f"node_models.{agent_type}.encoder.map_encoder.convolutions.3.bias",
        f"{agent_type}/map_encoder.fc.weight": f"node_models.{agent_type}.encoder.map_encoder.fc.weight",
        f"{agent_type}/map_encoder.fc.bias": f"node_models.{agent_type}.encoder.map_encoder.fc.bias"
    }


def _edge_encoder_mapping(agent_type_1: str, agent_type_2: str) -> dict:
    return {
        f"{agent_type_1}->{agent_type_2}/edge_encoder.weight_ih_l0": f"node_models.{agent_type_1}.encoder.edge_state_encoders.{agent_type_1} -> {agent_type_2}.lstm.weight_ih_l0",
        f"{agent_type_1}->{agent_type_2}/edge_encoder.weight_hh_l0": f"node_models.{agent_type_1}.encoder.edge_state_encoders.{agent_type_1} -> {agent_type_2}.lstm.weight_hh_l0",
        f"{agent_type_1}->{agent_type_2}/edge_encoder.bias_ih_l0": f"node_models.{agent_type_1}.encoder.edge_state_encoders.{agent_type_1} -> {agent_type_2}.lstm.bias_ih_l0",
        f"{agent_type_1}->{agent_type_2}/edge_encoder.bias_hh_l0": f"node_models.{agent_type_1}.encoder.edge_state_encoders.{agent_type_1} -> {agent_type_2}.lstm.bias_hh_l0"
    }


def _latent_mapping(agent_type: str) -> dict:
    return {
        f"{agent_type}/p_z_x.weight": f"node_models.{agent_type}.latent.p_z_x.0.0.weight",
        f"{agent_type}/p_z_x.bias": f"node_models.{agent_type}.latent.p_z_x.0.0.bias",
        f"{agent_type}/hx_to_z.weight": f"node_models.{agent_type}.latent.p_z_x.1.weight",
        f"{agent_type}/hx_to_z.bias": f"node_models.{agent_type}.latent.p_z_x.1.bias",
        f"{agent_type}/hxy_to_z.weight": f"node_models.{agent_type}.latent.q_z_xy.1.weight",
        f"{agent_type}/hxy_to_z.bias": f"node_models.{agent_type}.latent.q_z_xy.1.bias"
    }


def _decoder_mapping(agent_type: str) -> dict:
    return {
        f"{agent_type}/decoder/state_action.0.weight": f"node_models.{agent_type}.decoder.state_action.weight",
        f"{agent_type}/decoder/state_action.0.bias": f"node_models.{agent_type}.decoder.state_action.bias",
        f"{agent_type}/decoder/rnn_cell.weight_ih": f"node_models.{agent_type}.decoder.rnn_cell.weight_ih",
        f"{agent_type}/decoder/rnn_cell.weight_hh": f"node_models.{agent_type}.decoder.rnn_cell.weight_hh",
        f"{agent_type}/decoder/rnn_cell.bias_ih": f"node_models.{agent_type}.decoder.rnn_cell.bias_ih",
        f"{agent_type}/decoder/rnn_cell.bias_hh": f"node_models.{agent_type}.decoder.rnn_cell.bias_hh",
        f"{agent_type}/decoder/initial_h.weight": f"node_models.{agent_type}.decoder.initial_h.weight",
        f"{agent_type}/decoder/initial_h.bias": f"node_models.{agent_type}.decoder.initial_h.bias",
        f"{agent_type}/decoder/proj_to_GMM_log_pis.weight": f"node_models.{agent_type}.decoder.proj_to_gmm.log_pis.weight",
        f"{agent_type}/decoder/proj_to_GMM_log_pis.bias": f"node_models.{agent_type}.decoder.proj_to_gmm.log_pis.bias",
        f"{agent_type}/decoder/proj_to_GMM_mus.weight": f"node_models.{agent_type}.decoder.proj_to_gmm.mus.weight",
        f"{agent_type}/decoder/proj_to_GMM_mus.bias": f"node_models.{agent_type}.decoder.proj_to_gmm.mus.bias",
        f"{agent_type}/decoder/proj_to_GMM_log_sigmas.weight": f"node_models.{agent_type}.decoder.proj_to_gmm.log_sigmas.weight",
        f"{agent_type}/decoder/proj_to_GMM_log_sigmas.bias": f"node_models.{agent_type}.decoder.proj_to_gmm.log_sigmas.bias",
        f"{agent_type}/decoder/proj_to_GMM_corrs.weight": f"node_models.{agent_type}.decoder.proj_to_gmm.corrs.weight",
        f"{agent_type}/decoder/proj_to_GMM_corrs.bias": f"node_models.{agent_type}.decoder.proj_to_gmm.corrs.bias",
        f"{agent_type}/unicycle_initializer.weight": f"node_models.{agent_type}.decoder.dynamical_model.p0_model.weight",
        f"{agent_type}/unicycle_initializer.bias": f"node_models.{agent_type}.decoder.dynamical_model.p0_model.bias"
    }


def _node_model_mapping(agent_type: str, edge_agent_types: List[str]) -> dict:
    # Add node and maps encoder mappings
    mapping = _node_and_map_encoder_mapping(agent_type)

    # Add edge encoder mappings
    for edge_agent_type in edge_agent_types:
        mapping.update(_edge_encoder_mapping(agent_type, edge_agent_type))

    # Add latent module mappings
    mapping.update(_latent_mapping(agent_type))

    # Add decoder mappings
    mapping.update(_decoder_mapping(agent_type))

    return mapping


def _build_mapping(agent_types: List[str]) -> dict:
    mapping = _robot_future_encoder_mapping
    for agent_type in agent_types:
        mapping.update(_node_model_mapping(agent_type, agent_types))

    return mapping


def check_if_original(state_dict: dict) -> bool:
    """Check whether a state dictionary of a Trajectron++ instance matches its
    original model structure or not.

    Parameters
    ----------
    state_dict : dict
        State dictionary of a Trajectron++ instance.

    Returns
    -------
    bool
        True if the state dictionary matches the original Trajectron++
        structure, else False
    """
    # Assume a "/" in dictionary keys means the dictionarty is original
    return any(["/" in key for key in state_dict])


def convert_state_dict(state_dict: dict) -> dict:
    """Convert the state dictionary of an original Trajectron++ instance into
    one compatible with this project's.

    Parameters
    ----------
    state_dict : dict
        State dictionary of an original Trajectron++ instance.

    Returns
    -------
    dict
        The state dictionary with the same parameter weights matching the new
        structure.
    """
    # TODO: Make it more flexible
    agent_types = {m.split("/")[0] for m in state_dict if "->" not in m}
    mapping = _build_mapping(list(agent_types))
    new_state_dict = {mapping[key]: val for key, val in state_dict.items()}

    return new_state_dict
