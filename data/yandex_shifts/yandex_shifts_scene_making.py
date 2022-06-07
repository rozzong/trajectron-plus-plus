from itertools import chain
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Optional, Union
import warnings

import numpy as np
import pandas as pd
from mergedeep import merge
import transforms3d as t3d

# Filter Numba performance warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ysdc_dataset_api.features import FeatureRenderer
    from ysdc_dataset_api.utils import get_file_paths, scenes_generator
    from ysdc_dataset_api.proto import Scene as ShiftsScene

from .. import DatasetSceneMaker

sys.path.append("../../trajectron_plus_plus")
from trajectron_plus_plus.data import Scene, Node, NodeTypeEnum, GeometricMap, \
    derivative_of


class YandexShiftsSceneMaker(DatasetSceneMaker):

    columns = {
        "vehicle": pd.MultiIndex.from_product(
            [['position', 'velocity', 'acceleration'], ['x', 'y']]
        ).append(pd.MultiIndex.from_tuples([('yaw', 'rad')])),
        "pedestrian": pd.MultiIndex.from_product(
            [["position", "velocity", "acceleration"], ["x", "y"]]
        )
    }

    state = {
        "vehicle": {
            "position": ["x", "y"],
            "velocity": ["x", "y"],
            "acceleration": ["x", "y"],
            "yaw": ["rad"],
        },
        "pedestrian": {
            "position": ["x", "y"],
            "velocity": ["x", "y"],
            "acceleration": ["x", "y"],
        }
    }

    renderer_params = {
        "feature_map_params": {"rows": 400, "cols": 400, "resolution": 0.25},
        "padding": 50,
        "renderers_groups": [
            {
                "renderers": [
                    {
                        "road_graph": [
                            "crosswalk_occupancy",
                            "crosswalk_availability",
                            "lane_availability",
                            "lane_direction",
                            "lane_occupancy",
                            "lane_priority",
                            "lane_speed_limit",
                            "road_polygons",
                        ]
                    }
                ],
                "time_grid_params": {"start": 0, "step": 1, "stop": 0},
            },
        ],
    }

    @classmethod
    def raw_scene_iterator(cls, data_path: Path) -> Iterator[ShiftsScene]:
        file_paths = get_file_paths(str(data_path))
        raw_scenes = scenes_generator(file_paths)

        return raw_scenes

    @staticmethod
    def _get_motion(
            scene: ShiftsScene,
            track_id: Union[int, float],
            agent_type: str,
            quantity: str,
            axis: Optional[str] = None
    ) -> np.ndarray:
        """Get the motion properties of an agent.

        Parameters
        ----------
        scene : ShiftsScene
            The Yandex Shifts scene to find motion data in.
        track_id : Union[int, float]
            The track ID of the agent to extract motion of.
        agent_type : str
            The agent type of the agent to extract motion of.
        quantity : str
            The physical quantity to obtain.
        axis : Optional[str]
            The axis along which to obtain the physical quantity, if
            applicable.
        Returns
        -------
        np.ndarray
            The requested motion data.
        """
        motion_values = chain.from_iterable(
            [
                [
                    (
                            getattr(getattr(frame, quantity), axis)
                            or (np.nan if quantity == "position" else 0)
                    ) if axis is not None
                    else getattr(frame, quantity)
                    for frame in getattr(scene, f"{time}_ego_track")
                ] for time in ("past", "future")
            ]
        ) if track_id == -1 else chain.from_iterable(
            [
                value[0] if len(value) else np.nan for value in [
                    [
                        (
                                getattr(getattr(track, quantity), axis)
                                or (np.nan if quantity == "position" else 0)
                        ) if axis is not None
                        else getattr(track, quantity)
                        for track in frame.tracks
                        if track.track_id == track_id
                    ] for frame in getattr(
                        scene,
                        f"{time}_{agent_type}_tracks"
                    )
                ]
            ] for time in ("past", "future")
        )

        return np.array(list(motion_values))

    @classmethod
    def _build_node(
            cls,
            scene: ShiftsScene,
            node_type_enum: NodeTypeEnum,
            agent_type: str,
            track_id: Optional[int] = -1,
    ):
        """Build one AgentTrack instance for an agent in specific scene.

        Parameters
        ----------
        scene : ShiftsScene
            The Yandex Shifts scene to find motion data in.
        node_type_enum : NodeTypeEnum

        agent_type : str
            The agent type of the agent to extract motion of.
        track_id : Union[int, float]
            The track ID of the agent to extract motion of.
        Returns
        -------
        Node
            Trajectron++ node.
        """
        # Gather motion data
        state = cls.state[agent_type]
        motion = {}
        for quantity, axes in state.items():
            for axis in axes:
                motion[(quantity, axis)] = cls._get_motion(
                    scene,
                    track_id,
                    agent_type,
                    "linear_" + quantity
                    if quantity in ("velocity", "acceleration")
                    else quantity,
                    None if quantity == "yaw" else axis
                ) if not (
                        agent_type == "pedestrian"
                        and quantity == "acceleration"
                ) else derivative_of(motion[("velocity", axis)])

        # Get the first and last frame indices
        non_nan_frames = np.where(~np.isnan(list(motion.values())[0]))[0]
        first_frame, last_frame = non_nan_frames[0], non_nan_frames[-1]

        node_data = pd.DataFrame(
            motion,
            columns=cls.columns[agent_type]
        )[first_frame:last_frame + 1]

        # Handle the missing data case, with linear interpolation
        if node_data.isna().values.any():
            node_data.interpolate(inplace=True)

        # Create the node
        node = Node(
            node_type=getattr(node_type_enum, agent_type),
            node_id=str(int(track_id)),
            data=node_data,
            first_timestep=first_frame,
            is_robot=track_id == -1,
        )

        # Compute the node extrema
        extrema = {
            "x": {
                "min": np.nanmin(motion[("position", "x")]),
                "max": np.nanmax(motion[("position", "x")])
            },
            "y": {
                "min": np.nanmin(motion[("position", "y")]),
                "max": np.nanmax(motion[("position", "y")])
            }
        }

        return node, extrema

    @staticmethod
    def _update_extrema(extrema_ref: dict, extrema_new: dict) -> dict:
        """Take an agent's position extrema, compare them to reference
        position extrema, and update those if necessary.
        """
        for axis, axis_extrema in extrema_ref.items():
            for extremum, value in axis_extrema.items():
                if (
                        extremum == "min"
                        and extrema_new[axis][extremum] < value
                ) or (
                        extremum == "max"
                        and extrema_new[axis][extremum] > value
                ):
                    extrema_ref[axis][extremum] = \
                        extrema_new[axis][extremum]

        return extrema_ref

    @classmethod
    def process_scene(
            cls,
            shifts_scene: ShiftsScene,
            dataset_config: Mapping[str, Any],
            node_type_enum: NodeTypeEnum,
            get_maps: bool = True
    ) -> Scene:
        ego_node = None
        nodes = []
        extrema = None

        # Build the ego-node
        if dataset_config["ego_agent_type"] in node_type_enum:
            ego_node, extrema = cls._build_node(
                shifts_scene,
                node_type_enum,
                dataset_config["ego_agent_type"],
                -1,
            )
            nodes.append(ego_node)

        # Get all track IDs by agent type
        scene_track_ids = {}
        for agent_type in dataset_config["agent_types"]:
            scene_track_ids[agent_type] = np.unique(
                np.concatenate(
                    list(
                        chain.from_iterable(
                            [
                                [track.track_id for track in frame.tracks]
                                for frame in getattr(
                                    shifts_scene,
                                    f"{time}_{agent_type}_tracks"
                                )
                            ] for time in ("past", "future")
                        )
                    )
                )
            )

        # Get all agent types by track ID
        scene_agent_types = {}
        for agent_type, track_ids in scene_track_ids.items():
            scene_agent_types.update(
                {track_id: agent_type for track_id in track_ids}
            )

        # Iterate over valid prediction requests of the current scene
        for track_id, agent_type in scene_agent_types.items():
            node, agent_extrema = cls._build_node(
                shifts_scene,
                node_type_enum,
                agent_type,
                track_id
            )

            if extrema is None:
                extrema = agent_extrema

            # Add the agent's node
            nodes.append(node)

            # Update extrema if necessary
            if get_maps:
                extrema = cls._update_extrema(
                    extrema,
                    agent_extrema
                )

        # Create a Trajectron++ scene
        scene = Scene(
            timesteps=dataset_config["n_timesteps_per_scene"],
            dt=dataset_config["dt"],
            name=str(shifts_scene.id)
        )

        # Attach previously built nodes to the scene
        scene.robot = ego_node
        scene.nodes = nodes

        # Build maps, if asked
        if get_maps and len(nodes):
            # Update feature maps parameters
            scene_rast_size_m = np.array(
                [
                    extrema["x"]["max"] - extrema["x"]["min"],
                    extrema["y"]["max"] - extrema["y"]["min"]
                ],
            )
            # FIXME: Correct offset maps when the raster is not squared
            scene_rast_size_m = np.repeat(
                scene_rast_size_m.max(keepdims=True),
                2
            ) + 2 * cls.renderer_params["padding"]
            resolution = \
                cls.renderer_params["feature_map_params"]["resolution"]
            scene_rast_size_px = (
                    scene_rast_size_m / resolution
            ).astype(int)
            feature_map_params = {
                "rows": scene_rast_size_px[1],
                "cols": scene_rast_size_px[0],
                "resolution": resolution
            }

            # Wrap up the renderer configuration
            renderer_config = merge(
                {},
                {"feature_map_params": feature_map_params},
                {"renderers_groups": cls.renderer_params["renderers_groups"]}
            )

            # Create a feature renderer
            feature_renderer = FeatureRenderer(renderer_config)

            # Compute the transform for the current scene
            transform = np.eye(4)
            center = np.array(
                [
                    (extrema["x"]["min"] + extrema["x"]["max"]) / 2,
                    (extrema["y"]["min"] + extrema["y"]["max"]) / 2,
                ]
            )
            transform[:2, -1] = center
            transform = np.linalg.inv(transform).astype(np.float32)

            map_data = feature_renderer._create_feature_maps()
            slice_start = 0
            for renderer in feature_renderer._renderers:
                slice_end = slice_start + renderer.num_channels \
                    * renderer.n_history_steps
                renderer.render(
                    map_data[slice_start:slice_end, :, :],
                    shifts_scene,
                    transform
                )
                slice_start = slice_end

            # Create Trajectron++ geometric maps from the rasterized map
            # TODO: Investigate agent conditioned maps
            # TODO: Investigate the use of COO sparse matrices for maps
            t, r, z, _ = t3d.affines.decompose44(
                feature_renderer.to_feature_map_tf @ transform
            )
            homography = r * np.diag(z) + np.vstack((np.zeros((2, 3)), t)).T
            scene.map = {
                agent_type: GeometricMap(
                    data=map_data,
                    homography=homography,
                    description="_".join((scene.name, agent_type))
                ) for agent_type in dataset_config["agent_types"]
            }

        return scene
