from typing import Iterator, Sequence

from torch import nn


def parameters_but(
        module: nn.Module,
        to_exclude: Sequence[str],
        recurse: bool = True
) -> Iterator[nn.parameter.Parameter]:
    """Get all the parameters of a module not matching a selection of names.

    Parameters
    ----------
    module : nn.Module
        The module to yield parameters of.
    to_exclude : Sequence[str]
        The names of the parameters of the module to exclude.
    recurse : bool, default: True
        If True, look into child modules, else not.

    Yields
    ------
    Iterator[nn.parameter.Parameter]
        An iterator yielding parameters of the module different from selected
        names.
    """
    for name, parameter in module.named_parameters(recurse=recurse):
        if name not in to_exclude:
            yield parameter


def parameters_of(
        module: nn.Module,
        to_include: Sequence[str],
        recurse: bool = True
) -> Iterator[nn.parameter.Parameter]:
    """Get all the parameters of a module matching a selection of names.

    Parameters
    ----------
    module : nn.Module
        The module to yield parameters of.
    to_include : Sequence[str]
        The names of the parameters of the module to include.
    recurse : bool, default: True
        If True, look into child modules, else not.

    Yields
    ------
    Iterator[nn.parameter.Parameter]
        An iterator yielding parameters of the module matching selected names.
    """
    for name, parameter in module.named_parameters(recurse=recurse):
        if name in to_include:
            yield parameter
