from typing import Optional

import torch
from torch import nn

from . import attention


class StateAttention(nn.Module):
    """State attention module
    
    Compute contexts between input encodings and decoded states.
    
    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
    p : float, default: 0.5
        The probability of an element to be zeroed.
    attention_type : str, default: "AdditiveAttention"
        The type of attention to use to compute contexts.
    """
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            p: float = 0.5,
            attention_type: str = "AdditiveAttention"
    ) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.attention_type = attention_type
        
        self.attention: Optional[attention.Attention] = None
        self.dropout: Optional[nn.Dropout] = None

        self._build()
    
    def _build(self) -> None:
        self.attention = getattr(attention, self.attention_type)(
            self.in_features,
            self.out_features
        )
        self.dropout = nn.Dropout(self.p)
        
    def forward(
            self,
            inputs: torch.Tensor,
            state: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            The input encodings.
        state : torch.Tensor
            The decoded state.
        batch_size : int
            The size of th batch.
            
        Returns
        -------
        torch.Tensor
            Contexts between input encodings and decoded states.
        """
        # Reshape the annotations
        history_annotations = inputs.view(batch_size, -1, self.in_features)
        future_annotations = state.view(batch_size, -1, self.out_features)
        
        # Compute contexts
        contexts, _ = self.attention(
            history_annotations,
            future_annotations,
            history_annotations
        )
        
        contexts = torch.reshape(contexts, (-1, self.in_features))
        
        # Apply some dropout
        contexts = self.dropout(contexts)
        
        return contexts

