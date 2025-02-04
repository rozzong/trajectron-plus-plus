from abc import ABC, abstractmethod

import torch
from torch import nn

from .utils import run_lstm_on_variable_length_seqs, unpack_rnn_state


class NodeEncoder(nn.Module, ABC):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            p: float,
            *args,
            **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p

        self.lstm = None
        self.dropout = None

        self._build(*args, **kwargs)

    @abstractmethod
    def _build(self, *args, **kwargs):
        pass


class NodeHistoryEncoder(NodeEncoder):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            p: float
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            p,
        )

    def _build(self):
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.p)

    def forward(
            self,
            history: torch.Tensor,
            first_timesteps: torch.Tensor
    ):
        outputs, _ = run_lstm_on_variable_length_seqs(
            self.lstm,
            history,
            first_timesteps
        )
        outputs = self.dropout(outputs)
        last_timesteps = -(first_timesteps + 1)
        last_outputs = outputs[
            torch.arange(len(first_timesteps)),
            last_timesteps
        ]

        return last_outputs


class NodeFutureEncoder(NodeEncoder):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            initializer_input_size: int,
            p: float
    ):
        self.initializer_input_size = initializer_input_size

        self.state_initializers = None

        super().__init__(
            input_dim,
            hidden_dim,
            p,
        )

    def _build(self):
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.p)
        self.keys = ("h", "c")
        self.state_initializers = nn.ModuleDict(
            {
                k: nn.Linear(
                    self.initializer_input_size,
                    self.hidden_dim
                ) for k in self.keys
            }
        )

    def forward(self, present, future):
        initial_state = dict()

        for k in self.keys:
            output = self.state_initializers[k](present)
            initial_state[k] = torch.stack(
                [output, torch.zeros_like(output)],
                dim=0
            )

        _, state = self.lstm(future, tuple(initial_state.values()))
        state = unpack_rnn_state(state)
        state = self.dropout(state)

        return state
