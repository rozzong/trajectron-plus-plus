from typing import Optional, Union

import torch
from torch import nn
from torch import distributions as td

from .mode import Mode
from .distributions import PDistribution, QDistribution


class DiscreteLatent(nn.Module):

    def __init__(
            self,
            n: int,
            k: int,
            x_size: int,
            y_size: int,
            p_z_x_mlp_dims: Optional[int] = None,
            q_z_xy_mlp_dims: Optional[int] = None,
            p_dropout: Optional[float] = None,
            kl_min: Optional[float] = None
    ):
        super().__init__()

        self.n = n
        self.k = k
        self.x_size = x_size
        self.y_size = y_size  # 4 * self.hyperparams['enc_rnn_dim_future']
        self.p_z_x_mlp_dims = p_z_x_mlp_dims
        self.q_z_xy_mlp_dims = q_z_xy_mlp_dims
        self.p_dropout = p_dropout
        self.kl_min = kl_min

        self._z_logit_clip = None
        self.p = None
        self.q = None
        self.p_z_x = None
        self.q_z_xy = None

        self._build()

    @property
    def kl_divergence(self) -> torch.Tensor:
        kl_separated = torch.atleast_2d(td.kl_divergence(self.q, self.p))
        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)
        if self.kl_min is not None and self.kl_min > 0:
            kl_minibatch = torch.clip(kl_minibatch, self.kl_min)

        return torch.sum(kl_minibatch)

    @property
    def z_logit_clip(self) -> Union[float, None]:
        return self._z_logit_clip if self.training else None

    @z_logit_clip.setter
    def z_logit_clip(self, value: Union[float, None]):
        self._z_logit_clip = value

    @property
    def z_size(self) -> int:
        return self.n * self.k

    def _build(self):
        self.p_z_x = nn.Sequential(
            nn.Sequential(
                nn.Linear(self.x_size, self.p_z_x_mlp_dims),
                nn.ReLU(),
                nn.Dropout(self.p_dropout)
            ) if self.p_z_x_mlp_dims is not None else nn.Identity(),
            nn.Linear(
                self.p_z_x_mlp_dims or self.x_size,
                self.z_size
            )
        )
        self.q_z_xy = nn.Sequential(
            nn.Sequential(
                nn.Linear(
                    self.x_size + self.y_size,
                    self.q_z_xy_mlp_dims
                ),
                nn.ReLU(),
                nn.Dropout(self.p_dropout)
            ) if self.q_z_xy_mlp_dims is not None else nn.Identity(),
            nn.Linear(
                self.q_z_xy_mlp_dims or (self.x_size + self.y_size),
                self.z_size
            )
        )

    def update_p(self, x: torch.Tensor) -> None:
        h = self.p_z_x(x)
        self.p = PDistribution.from_h(h, self.n, self.k, self.z_logit_clip)

    def update_q(self, x: torch.Tensor, y: torch.Tensor) -> None:
        xy = torch.cat([x, y], dim=1)
        h = self.q_z_xy(xy)
        self.q = QDistribution.from_h(h, self.n, self.k, self.z_logit_clip)

    def forward(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            n_samples: int = 1,
            mode: Optional[Mode] = None
    ):
        # If no label is provided, the model is assumed to be used for
        # prediction
        is_predicting = y is None

        # Update distributions
        if not is_predicting:
            self.update_q(x, y)
        self.update_p(x)

        # Sample from the latent space
        if self.training:
            z = self.q.sample_n(n_samples)
        else:
            self.p.mode = mode if mode is not None else Mode.FULL
            z = self.p.sample_n(n_samples)

        return z
