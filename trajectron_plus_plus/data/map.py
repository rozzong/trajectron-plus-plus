from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Sequence, Tuple, Union

from kornia.geometry import crop_and_resize, rotate
import numpy as np
import torch


class Map(ABC):

    def __init__(
            self,
            data: torch.Tensor,
            homography: torch.Tensor,
            description: Optional[str] = None
    ):
        self.data = data
        self.homography = homography
        self.description = description

    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_cropped_patches(
            self,
            patch_size: Tuple[int, int, int, int],
            points: torch.Tensor,
            angles: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Crop and rotate the map with given positions and orientations.

        Parameters
        ----------
        patch_size : Tuple[int, int, int, int]
            Tuple of 4 integers defining the geometry of a patch relatively to
            the agent position, ordered as in the following drawing:

                ┌──────┬─────────────┐
                │      ┆             │
                │     [3]            │
                │      ┆             │
                │      ┆             │
                ├┄┄[0]┄┼┄┄┄┄┄[2]┄┄┄┄┄┤
                │      ┆             │
                │     [1]            │
                └──────┴─────────────┘

        points : torch.Tensor
            2D points in the world frame, with shape (n_points, 2).
        angles : Union[float, torch.Tensor], optional
            Angles in radians for the respective points, with shape
            (n_points,). A single value can also be provided, and if so it will
            be expanded to all points. If no value is passed, no rotation is
            applied.

        Returns
        -------
        torch.Tensor
            Map patches, cropped and rotated.
        """
        pass

    def to_map_points(self, scene_pts: np.ndarray) -> np.ndarray:
        """Transform points in the scene coordinates to points in the map
        coordinates.

        Parameters
        ----------
        scene_pts : np.ndarray
            Points in the scene frame.

        Returns
        -------
        np.ndarray
            Points in the map frame.
        """
        org_shape = None
        if len(scene_pts.shape) > 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))

        n, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, n))
        points_with_one[:dims] = scene_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims]

        if org_shape is not None:
            map_points = map_points.reshape(org_shape)

        return map_points


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography
    must transform a point in scene coordinates to the respective point in map
    coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(
            self,
            data: torch.Tensor,
            homography: torch.Tensor,
            description: Optional[str] = None
    ):
        super().__init__(data, homography, description)

        self._last_padding = None
        self._last_padded_map = None
        self._torch_map = None

    @property
    def image(self) -> np.ndarray:
        # We have to transpose x and y to rows and columns.
        # Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return np.transpose(self.data, (2, 1, 0)).astype(np.uint)

    def torch_map(self, device: Union[torch.device, str]) -> torch.Tensor:
        if self._torch_map is None:
            self._torch_map = torch.tensor(
                self.data,
                dtype=torch.uint8,
                device=device
            )
        return self._torch_map

    @staticmethod
    def crop_rotate_maps(
            maps: Sequence["GeometricMap"],
            patch_size: Tuple[int, int, int, int],
            points: torch.Tensor,
            angles: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Crop and rotate maps with given positions and orientations.

        Parameters
        ----------
        maps : Sequence[GeometricMap]
            A sequence of Trajectron++ geometric maps.
        patch_size : Tuple[int, int, int, int]
            Tuple of 4 integers defining the geometry of a patch relatively to
            the agent position, ordered as in the following drawing:

                ┌──────┬─────────────┐
                │      ┆             │
                │     [3]            │
                │      ┆             │
                │      ┆             │
                ├┄┄[0]┄┼┄┄┄┄┄[2]┄┄┄┄┄┤
                │      ┆             │
                │     [1]            │
                └──────┴─────────────┘

        points : torch.Tensor
            2D points in the world frame, with shape (n_points, 2).
        angles : Union[float, torch.Tensor], optional
            Angles in radians for the respective points, with shape
            (n_points,). A single value can also be provided, and if so it will
            be expanded to all points. If no value is passed, no rotation is
            applied.

        Returns
        -------
        torch.Tensor
            Map patches, cropped and rotated.
        """
        batch_size = len(points)
        device = points.device

        # Compute context paddings
        lat_size = 2 * np.max((patch_size[0], patch_size[2]))
        long_size = 2 * np.max((patch_size[1], patch_size[3]))
        context_padding = torch.tensor(
            np.ceil(np.sqrt(2) * np.array([lat_size, long_size])),
            device=device
        ).int()
        points = torch.hstack(
            (points, torch.ones((batch_size, 1), device=device)))

        # Iterate over unique maps
        first_index = 0
        patches = []
        size = (
            int(patch_size[0] + patch_size[2]),
            int(patch_size[1] + patch_size[3])
        )

        for unique_map, counts in Counter(maps).items():
            # Compute the padded map and repeat it
            repeated_padded_map = unique_map.get_padded_map(
                *context_padding,
                device
            ).float().unsqueeze(0).expand(counts, -1, -1, -1)

            # Compute patch boxes
            map_transforms = torch.tensor(
                unique_map.homography,
                device=device
            ).float().unsqueeze(0).expand(counts, -1, -1)
            maps_centers = torch.einsum(
                "bij,bj->bi",
                map_transforms,
                points[first_index:first_index + counts]
            )[:, :-1].int()
            padded_map_centers = (maps_centers + context_padding).unsqueeze(1)
            x_min = padded_map_centers[..., [0]] - context_padding[0]
            x_max = padded_map_centers[..., [0]] + context_padding[0]
            y_min = padded_map_centers[..., [1]] - context_padding[1]
            y_max = padded_map_centers[..., [1]] + context_padding[1]
            boxes = torch.cat(
                (
                    torch.cat((x_min, y_min), dim=2),
                    torch.cat((x_max, y_min), dim=2),
                    torch.cat((x_max, y_max), dim=2),
                    torch.cat((x_min, y_max), dim=2),
                ),
                dim=1
            )

            # Crop the padded map in patches
            patches.append(
                crop_and_resize(
                    repeated_padded_map,
                    boxes,
                    size
                )
            )

            first_index += counts

        # Concatenate all patches
        patches = torch.cat(patches)

        # Rotate all patches
        if angles is not None:
            if isinstance(angles, float):
                angles = torch.full((batch_size,), angles, device=device)
            angles = angles.float()
            patches = rotate(patches, angles)

        return patches

    def get_padded_map(
            self,
            padding_x: int,
            padding_y: int,
            device: Union[str, torch.device] = "cpu"
    ) -> torch.Tensor:
        """Get the padded features of a map.

        Parameters
        ----------
        padding_x : int
            The padding to apply to the map on the x-axis.
        padding_y : int
            The padding to apply to the map on the y-axis.
        device : Union[str, torch.device], default: "cpu"
            The device to work with.

        Returns
        -------
        torch.Tensor
            The padded map features.
        """
        if not self._last_padding == (padding_x, padding_y):
            self._last_padding = (padding_x, padding_y)
            padded_map = torch.full(
                (
                    self.data.shape[0],
                    self.data.shape[1] + 2 * padding_x,
                    self.data.shape[2] + 2 * padding_y
                ),
                0,
                dtype=torch.uint8,
                device=device
            )
            padded_map[..., padding_x:-padding_x, padding_y:-padding_y] \
                = self.torch_map(device)
            self._last_padded_map = padded_map

        return self._last_padded_map

    def get_cropped_patches(
            self,
            patch_size: Tuple[int, int, int, int],
            points: torch.Tensor,
            angles: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Crop and rotate the map with given positions and orientations.

        Parameters
        ----------
        patch_size : Tuple[int, int, int, int]
            Tuple of 4 integers defining the geometry of a patch relatively to
            the agent position, ordered as in the following drawing:

                ┌──────┬─────────────┐
                │      ┆             │
                │     [3]            │
                │      ┆             │
                │      ┆             │
                ├┄┄[0]┄┼┄┄┄┄┄[2]┄┄┄┄┄┤
                │      ┆             │
                │     [1]            │
                └──────┴─────────────┘

        points : torch.Tensor
            2D points in the world frame, with shape (n_points, 2).
        angles : Union[float, torch.Tensor], optional
            Angles in radians for the respective points, with shape
            (n_points,). A single value can also be provided, and if so it will
            be expanded to all points. If no value is passed, no rotation is
            applied.

        Returns
        -------
        torch.Tensor
            Map patches, cropped and rotated.
        """
        return self.crop_rotate_maps(
            [self] * len(points),
            patch_size,
            points,
            angles
        )


class ImageMap(Map):
    # TODO: Implement for image maps -> watch flipped coordinate system
    pass
