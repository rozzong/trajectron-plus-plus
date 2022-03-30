from abc import ABC, abstractmethod
from typing import Mapping, Union

import torch
from torch import nn


class DynamicalModel(ABC, nn.Module):

    def __init__(
            self,
            dt: float,
            limits: Mapping[str, Union[int, float]],
            *args,
            **kwargs
    ):
        super().__init__()

        self.dt = dt
        self.limits = limits

        self._initial_conditions = None

        self.initialize(*args, **kwargs)

    def _apply(self, fn):
        super()._apply(fn)
        tensors = {
            name: val for name, val in vars(self).items()
            if isinstance(val, torch.Tensor)
        }
        for attr, val in tensors.items():
            setattr(self, attr, fn(val))

        return self

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, value):
        self._initial_conditions = value

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def integrate_samples(self, s, x):
        pass

    @abstractmethod
    def integrate_distribution(self, dist, x):
        pass
