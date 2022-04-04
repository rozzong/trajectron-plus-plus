from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import distributions as td

from .mode import Mode


class LatentDistribution(ABC, td.OneHotCategorical):

    def __init__(
            self,
            probs: Optional[torch.Tensor] = None,
            logits: Optional[torch.Tensor] = None,
            validate_args=False
    ):
        super().__init__(probs, logits, validate_args)

    @property
    def k(self) -> int:
        return self.probs.shape[-1]

    @property
    def n(self) -> int:
        return self.probs.shape[-2]

    @property
    def z_dim(self) -> int:
        return self.n * self.k

    @property
    def mutual_inf(self) -> torch.Tensor:
        mean_dist = self.__class__(probs=self.probs.mean(dim=0))
        return (mean_dist.entropy() - self.entropy().mean(dim=0)).sum()

    @property
    @abstractmethod
    def n_components(self) -> int:
        pass

    @classmethod
    def from_h(
            cls,
            h: torch.Tensor,
            n: int,
            k: int,
            clip: Optional[float] = None,
            *args,
            **kwargs
    ):
        logits_separated = torch.reshape(h, (-1, n, k))
        logits_separated_mean_zero = logits_separated \
            - torch.mean(logits_separated, dim=-1, keepdim=True)

        # if self.training and self.z_logit_clip is not None:
        if clip is not None:
            logits_separated_mean_zero = torch.clip(
                logits_separated_mean_zero,
                min=-clip,
                max=clip
            )

        return cls(logits=logits_separated_mean_zero, *args, **kwargs)

    @staticmethod
    def all_one_hot_combinations(n, k):
        # TODO: Make it a torch.Tensor from the start
        return np.eye(k).take(
            np.reshape(np.indices([k] * n), [n, -1]).T,
            axis=0
        ).reshape(-1, n * k)  # [K**N, N*K]

    def log_prob(self, value):
        value_nk = torch.reshape(value, [len(value), -1, self.n, self.k])
        return torch.sum(super().log_prob(value_nk), dim=2)

    def sample_n(self, n):
        # TODO: Implement `sample` instead
        # TODO: Make it full torch
        batch_size = self.batch_shape[0]
        z_nk = torch.as_tensor(
            self.all_one_hot_combinations(self.n, self.k),
            dtype=torch.float,
            device=self.logits.device
        ).repeat(n, batch_size)
        return torch.reshape(
            z_nk,
            (n * self.n_components, -1, self.z_dim)
        )


class QDistribution(LatentDistribution):

    @property
    def n_components(self) -> int:
        return self.n * self.k


class PDistribution(LatentDistribution):

    def __init__(
            self,
            probs: Optional[torch.Tensor] = None,
            logits: Optional[torch.Tensor] = None,
            validate_args: Optional[torch.Tensor] = None,
            mode: Mode = Mode.MOST_LIKELY
    ):
        super().__init__(probs, logits, validate_args)

        self._mode = mode

    @property
    def n_components(self) -> int:
        if self.mode in (Mode.ALL_Z, Mode.MOST_LIKELY, Mode.DISTRIBUTION):
            return 1
        elif self.mode == Mode.FULL:
            return self.k ** self.n
        else:
            raise ValueError

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    def sample_n(self, n):
        # TODO: Implement `sample` instead
        # TODO: Make it full torch
        if self.mode == Mode.MOST_LIKELY:
            eye = torch.eye(self.event_shape[-1], device=self.logits.device)
            indices = torch.argmax(self.probs, dim=2)
            z_nk = torch.unsqueeze(eye[indices], dim=0).expand(n, -1, -1, -1)
            k = n
        elif self.mode in (Mode.FULL, Mode.ALL_Z):
            batch_size = self.batch_shape[0]
            k, r = (n * self.k ** self.n, n) \
                if self.mode == Mode.FULL \
                else (self.k ** self.n, 1)
            z_nk = torch.as_tensor(
                self.all_one_hot_combinations(self.n, self.k),
                dtype=torch.float,
                device=self.logits.device
            ).repeat(r, batch_size)
        else:
            z_nk = super().sample((n,))
            k = n

        return torch.reshape(z_nk, (k, -1, self.z_dim))
