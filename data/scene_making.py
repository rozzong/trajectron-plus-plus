from abc import ABC, abstractmethod
from itertools import islice, tee
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, TypeVar

from trajectron_plus_plus.data import NodeTypeEnum, Scene


RawScene = TypeVar("RawScene")


class SliceableIterator:
    """Iterator with a slice feature.

    Parameters
    ----------
    iterator : Iterator
        The iterator to make sliceable.
    """

    def __init__(self, iterator: Iterator):
        self.__iterator, self.__archive = tee(iterator)

    def __iter__(self) -> Iterator[Any]:
        return self.__iterator

    def __next__(self) -> Any:
        return next(self.__iterator)

    def __getitem__(self, item) -> Any:
        self.__archive, to_slice = tee(self.__archive)

        if isinstance(item, int) and item >= 0:
            return next(islice(to_slice, item, item + 1))
        elif isinstance(item, slice):
            return islice(to_slice, item.start, item.stop, item.step)
        else:
            raise KeyError()


class ReloadingIterator:
    """Macro-iterator that loops over the iterators returned by a loading
    function.

    Parameters
    ----------
    loader : Callable[[], Iterator]
        A function returning a new iterator.
    """

    def __init__(self, loader: Callable[[], Iterator]):
        self.__loader = loader
        self.__iterator = None
        self._load()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.__iterator)
        except StopIteration:
            self._load()
            raise StopIteration

    def _load(self):
        self.__iterator = self.__loader()


class DatasetSceneMaker(ABC):

    @classmethod
    @abstractmethod
    def raw_scene_iterator(cls, data_path: Path) -> Iterator[RawScene]:
        """

        Args:
            data_path:

        Yields:

        """
        pass

    @classmethod
    @abstractmethod
    def process_scene(
            cls,
            raw_scene: RawScene,
            dataset_config: Mapping[str, Any],
            node_type_enum: NodeTypeEnum,
            get_maps: bool = False
    ) -> Scene:
        """

        Args:
            raw_scene:
            dataset_config:
            node_type_enum:
            get_maps:

        Returns:

        """
        pass

    @classmethod
    def scene_iterator(
            cls,
            source_path: str,
            data: str,
            dataset_config: Mapping[str, Any],
            node_type_enum: NodeTypeEnum,
            get_maps: bool = False,
            start: Optional[int] = None,
            stop: Optional[int] = None,
            step: Optional[int] = None,
    ) -> Iterator[Scene]:
        """Iterate over the dataset scenes once.

        Args:
            source_path:
            data:
            dataset_config:
            node_type_enum:
            get_maps:
            start:
            stop:
            step:

        Yields:

        """
        # Infer the data path
        data_path = Path(source_path).resolve() / Path(data)

        # Get the selected raw scenes interator
        subset = slice(start, stop, step)
        raw_scenes = SliceableIterator(
            cls.raw_scene_iterator(data_path)
        )[subset]

        for raw_scene in raw_scenes:
            yield cls.process_scene(
                raw_scene,
                dataset_config,
                node_type_enum,
                get_maps
            )

    @classmethod
    def looping_scene_iterator(
            cls,
            source_path: str,
            data: str,
            dataset_config: Mapping[str, Any],
            node_type_enum: NodeTypeEnum,
            get_maps: bool = False,
            start: Optional[int] = None,
            stop: Optional[int] = None,
            step: Optional[int] = None,
    ) -> Iterator[Scene]:
        """Iterate over the dataset scenes multiple times.

        Args:
            source_path:
            data:
            dataset_config:
            node_type_enum:
            get_maps:
            start:
            stop:
            step:

        Yields:

        """
        return ReloadingIterator(
            lambda: cls.scene_iterator(
                source_path,
                data,
                dataset_config,
                node_type_enum,
                get_maps,
                start,
                stop,
                step
            )
        )
