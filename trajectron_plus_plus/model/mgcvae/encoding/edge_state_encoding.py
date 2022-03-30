from abc import ABC, abstractmethod
from typing import Callable, Union

import torch
from torch import nn

from .utils import run_lstm_on_variable_length_seqs


# TODO: Specify types in signatures

class EdgeStateEncoder(nn.Module, ABC):

    def __init__(
            self,
            edge_type,
            input_size: int,
            output_size: int,
            p: float,
            use_dynamic_edges: bool,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.edge_type = edge_type
        self.input_size = input_size
        self.output_size = output_size
        self.p = p
        self.use_dynamic_edges = use_dynamic_edges

        self._build(*args, **kwargs)

    @abstractmethod
    def _build_combiner(self, *args, **kwargs) -> int:
        """Build the combiner.

        Returns
        -------
        int
            The size of the output of the combiner.
        """
        pass

    def _build(self, *args, **kwargs):
        self.combiner_output_size = self._build_combiner(*args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.combiner_output_size,
            hidden_size=self.output_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.p)

    @abstractmethod
    def combine(self, edge_states_list, neighbors_edge_value):
        pass

    def forward(
            self,
            node_history_st,
            neighbors,
            neighbors_edge_value,
            first_history_indices,
            states
    ):
        edge_states_list = []   # [Number of neighbors, max_ht, state_dim]
        for i, neighbor_states in enumerate(neighbors):
            if len(neighbor_states):
                edge_states_list.append(
                    torch.stack(neighbor_states, dim=0))
            else:
                len_neighbor_state = sum(
                    [len(axes) for axes in states[self.edge_type[1]].values()]
                )
                edge_states_list.append(
                    torch.zeros(
                        (1, node_history_st.shape[1], len_neighbor_state),
                        device=node_history_st.device
                    )
                )

        combined_neighbors, combined_edge_masks = self.combine(
            edge_states_list,
            neighbors_edge_value
        )

        joint_history = torch.cat(
            [combined_neighbors, node_history_st],
            dim=-1
        )

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.lstm,
            original_seqs=joint_history,
            lower_indices=first_history_indices
        )
        outputs = self.dropout(outputs)
        last_index_per_sequence = -(first_history_indices + 1)

        ret = outputs[
            torch.arange(len(last_index_per_sequence)),
            last_index_per_sequence
        ]

        if self.use_dynamic_edges:
            ret *= combined_edge_masks

        return ret


class ReducingEdgeStateEncoder(EdgeStateEncoder):

    def __init__(
            self,
            edge_type,
            input_size: int,
            output_size: int,
            p: float,
            use_dynamic_edges: bool,
            reduce: Union[Callable, str],
    ):
        super().__init__(
            edge_type,
            input_size,
            output_size,
            p,
            use_dynamic_edges,
            reduce
        )

    def _build_combiner(self, reduce) -> int:
        if isinstance(reduce, str):
            reduce = getattr(torch, reduce)
        self.reduce = reduce

        return self.input_size

    def combine(self, edge_states_list, neighbors_edge_value):
        op_applied_edge_states_list = []
        for neighbors_state in edge_states_list:
            op_applied_edge_states_list.append(
                self.reduce(neighbors_state, dim=0)
            )
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)

        combined_edge_masks = None
        if self.use_dynamic_edges:
            # Should now be (bs, time, 1)
            op_applied_edge_mask_list = []
            for edge_value in neighbors_edge_value:
                op_applied_edge_mask_list.append(
                    torch.clamp(
                        self.reduce(edge_value, dim=0, keepdim=True),
                        max=1
                    )
                )
            combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        return combined_neighbors, combined_edge_masks


class PointNetEdgeStateEncoder(EdgeStateEncoder):

    def __init__(
            self,
            edge_type,
            input_size: int,
            output_size: int,
            p: float,
            use_dynamic_edges: bool,
    ):
        super().__init__(
            edge_type,
            input_size,
            output_size,
            p,
            use_dynamic_edges,
        )

        raise NotImplementedError

    def _build_combiner(self) -> int:
        return 0

    def combine(self, edge_states_list, neighbors_edge_value):
        pass


class AttentionEdgeStateEncoder(EdgeStateEncoder):

    def __init__(
            self,
            edge_type,
            input_size: int,
            output_size: int,
            p: float,
            use_dynamic_edges: bool,
    ):
        super().__init__(
            edge_type,
            input_size,
            output_size,
            p,
            use_dynamic_edges,
        )

        raise NotImplementedError

    def _build_combiner(self) -> int:
        return 0

    def combine(self, edge_states_list, neighbors_edge_value):
        pass
