from abc import ABC, abstractmethod
from typing import TypeVar

from numpy import exp


T = TypeVar("T")


class Scheduler(ABC):
    """Scheduler.

    Schedule the value a variable is bound to have according to a law.
    """

    def __init__(self):
        self._i_step = 0

    @property
    def i_step(self) -> int:
        return self._i_step

    @property
    def value(self) -> T:
        return self._rule(self._i_step)

    @abstractmethod
    def _rule(self, i_step: int) -> T:
        pass

    def reset(self) -> None:
        self._i_step = 0

    def step(self) -> None:
        self._i_step += 1


class ExponentialScheduler(Scheduler):
    """Exponential scheduler.

    Schedule the value a variable bound to have according to an exponential
    law.

    Parameters
    ----------
    start : T
        The initial value of the variable.
    stop : T
        The final value of the variable.
    rate : float
        The exponential rate.
    """

    def __init__(self, start: T, stop: T, rate: float):
        super().__init__()

        self.start = start
        self.stop = stop
        self.rate = rate

    def _rule(self, i_step: int) -> T:
        return self.stop - (self.stop - self.start) * self.rate ** i_step


class SigmoidScheduler(Scheduler):
    """Sigmoid scheduler.

    Schedule the value a variable bound to have according to a sigmoidal
    law.

    Parameters
    ----------
    start : T
        The initial value of the variable.
    stop : T
        The final value of the variable.
    center_step : int
        The step for which the sigmoidal law undergoes inflection.
    steps_low_to_high : float
        The steepness of the sigmoid slope.
    """

    def __init__(
            self,
            start: T,
            stop: T,
            center_step: int,
            steps_low_to_high: float
    ):
        super().__init__()

        self.start = start
        self.stop = stop
        self.center_step = center_step
        self.steps_low_to_high = steps_low_to_high

    @staticmethod
    def __sigmoid(x: float) -> float:
        return 1 / (1 + exp(-x))

    def _rule(self, i_step: int) -> T:
        return self.start + (self.stop - self.start) \
            * self.__sigmoid(
                (i_step - self.center_step) / self.steps_low_to_high
            )


class SchedulerDict(dict):
    """Scheduler dict.

    A dictionary holding schedulers and implementing `step` and `reset`
    methods.
    """

    def reset(self):
        for scheduler in self.values():
            scheduler.reset()

    def step(self):
        for scheduler in self.values():
            scheduler.step()
