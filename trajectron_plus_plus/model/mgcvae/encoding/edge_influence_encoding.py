from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import torch
from torch import nn

from .additive_attention import AdditiveAttention
from ...model_utils import unpack_rnn_state  # TODO: Change location


class EdgeInfluenceEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _build(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(
            self,
            encoded_edges,
            encoded_history,
            batch_size: Optional[int] = None
    ):
        pass


class ReducingEdgeInfluenceEncoder(EdgeInfluenceEncoder):

    def __init__(self, reduce: Union[Callable, str]):
        super().__init__()

        if isinstance(reduce, str):
            reduce = getattr(torch, reduce)

        self.reduce = reduce

    def _build(self):
        pass

    def forward(
            self,
            encoded_edges,
            encoded_history,
            batch_size: Optional[int] = None
    ):
        stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
        combined_edges = self.reduce(stacked_encoded_edges, dim=0)

        return combined_edges


class BiRNNEdgeInfluenceEncoder(EdgeInfluenceEncoder):

    def __init__(self, input_dim: int, hidden_dim: int, p: float):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p

        self._build()

    def _build(self):
        self.output_dim = 4 * self.hidden_dim
        self.combiner = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.p)

    def forward(
            self,
            encoded_edges,
            encoded_history,
            batch_size: Optional[int] = None
    ):
        if len(encoded_edges):
            encoded_edges = torch.stack(encoded_edges, dim=1)
            _, state = self.combiner(encoded_edges)
            combined_edges = unpack_rnn_state(state)
            combined_edges = self.dropout(combined_edges)
        else:
            combined_edges = torch.zeros((batch_size, self.output_dim))

        return combined_edges


class AttentionEdgeInfluenceEncoder(EdgeInfluenceEncoder):

    def __init__(
            self,
            encoder_hidden_size: int,
            decoder_hidden_size: int,
            p: float
    ):
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.p = p

        self._build()

    def _build(self):
        self.output_dim = self.encoder_hidden_size
        self.combiner = AdditiveAttention(
            encoder_hidden_state_dim=self.encoder_hidden_size,
            decoder_hidden_state_dim=self.decoder_hidden_size,
        )
        self.dropout = nn.Dropout(self.p)

    def forward(
            self,
            encoded_edges,
            encoded_history,
            batch_size: Optional[int] = None
    ):
        if len(encoded_edges):
            encoded_edges = torch.stack(encoded_edges, dim=1)
            combined_edges, _ = self.combiner(
                encoded_edges,
                encoded_history
            )
            combined_edges = self.dropout(combined_edges)
        else:
            combined_edges = torch.zeros((batch_size, self.output_dim))

        return combined_edges
