from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class Attention(nn.Module, ABC):
    """Attention module.
    """

    in_features: int = None
    out_features: int = None

    @abstractmethod
    def score(
            self,
            query: Optional[torch.Tensor] = None,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the score.

        Parameters
        ----------
        query : Optional[torch.Tensor], default: None
            The query vectors.
        key : Optional[torch.Tensor], default: None
            The key vectors.
        value : Optional[torch.Tensor], default: None
            The value vectors.

        Returns
        -------
        torch.Tensor
            Score (or energy).
        """
        pass

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get context vectors.

        Parameters
        ----------
        query : torch.Tensor
            The query vectors.
        key : torch.Tensor
            The key vectors.
        value : torch.Tensor
            The value vectors.

        Returns
        -------
        torch.Tensor
            Context vectors.
        torch.Tensor
            Attention weights.
        """
        score = self.score(query, key)
        attention = torch.softmax(score, dim=-1)
        context = torch.bmm(attention, value)

        return context, attention


class DotProductAttention(Attention):
    """Dot-product attention module.
    """

    def score(
            self,
            query: Optional[torch.Tensor] = None,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.bmm(query, key.transpose(-2, -1))


class ScaledDotProductAttention(DotProductAttention):
    """Scaled dot-product attention module.

    Parameters
    ---------
    scale : float
        The scale to divide the score with.
    """

    def __init__(self, scale: float):
        super().__init__()

        self.scale = scale

    def score(
            self,
            query: Optional[torch.Tensor] = None,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return super().score(query, key, value) / self.scale


class AdditiveAttention(Attention):
    """Additive attention module.

    Parameters
    ----------
    query_dim : int
        The query dimension.
    key_dim : int
        The key dimension.
    hidden_dim : Optional[int], default: None
        The hidden dimension. Set to `(query_dim + key_dim) // 2` if not
        specified.
    """

    def __init__(
            self,
            query_dim: int,
            key_dim: int,
            hidden_dim: Optional[int] = None
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = (query_dim + key_dim) // 2

        self.q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.Tensor(hidden_dim).uniform_(-0.1, 0.1))

    def score(
            self,
            query: Optional[torch.Tensor] = None,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        l_q, l_k = query.shape[1], key.shape[1]
        query_resized = query.unsqueeze(1).expand(-1, l_k, -1, -1)
        key_resized = key.unsqueeze(2).expand(-1, -1, l_q, -1)

        return torch.tanh(self.q(query_resized) + self.k(key_resized)) @ self.v
