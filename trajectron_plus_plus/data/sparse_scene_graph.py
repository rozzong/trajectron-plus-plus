from sparse import COO

from .scene_graph import SceneGraph, TemporalSceneGraph


class SparseSceneGraph(SceneGraph):
    """Sparse Scene Graph

    A sparse version of the original implementation of scene graphs,
    especially useful for pickling.
    """

    _SPARSE_ATTRS = (
        "adj_mat",
        "edge_scaling",
        "weight_mat"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the attribute sparse
        for attr in self._SPARSE_ATTRS:
            if not isinstance(getattr(self, attr), COO):
                setattr(self, attr, COO(getattr(self, attr)))

    @classmethod
    def from_scene_graph(
            cls,
            scene_graph: SceneGraph
    ) -> "SparseSceneGraph":
        """Convert dense arrays of a scene graph into sparse arrays.

        Parameters
        ----------
        scene_graph : SceneGraph
            The scene graph to convert arrays of.

        Returns
        -------
        SparseSceneGraph
            The same scene graph with COO sparse matrices.
        """
        return cls(
            edge_radius=scene_graph.edge_radius,
            nodes=scene_graph.nodes,
            adj_mat=scene_graph.adj_mat,
            weight_mat=scene_graph.weight_mat,
            node_type_mat=scene_graph.node_type_mat,
            node_index_lookup=scene_graph.node_index_lookup,
            edge_scaling=scene_graph.edge_scaling
        )

    def get_connection_mask(self, node_index):
        return self.adj_mat[node_index] > 0 \
            if self.edge_scaling is None \
            else self.edge_scaling.todense()[node_index] > 1e-2


class SparseTemporalSceneGraph(TemporalSceneGraph):
    """Sparse Temporal Scene Graph

    A sparse version of the original implementation of temporal scene graphs,
    especially useful for pickling.
    """

    _SPARSE_ATTRS = (
        "adj_cube",
        "adj_mat",
        "edge_scaling",
        "weight_cube"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the attribute sparse
        for attr in self._SPARSE_ATTRS:
            if not isinstance(getattr(self, attr), COO):
                setattr(self, attr, COO(getattr(self, attr)))

    @classmethod
    def from_temporal_scene_graph(
            cls,
            temporal_scene_graph: TemporalSceneGraph
    ) -> "SparseTemporalSceneGraph":
        """Convert dense arrays of a temporal scene graph into sparse arrays.

        Parameters
        ----------
        temporal_scene_graph : TemporalSceneGraph
            The temporal scene graph to convert arrays of.

        Returns
        -------
        SparseTemporalSceneGraph
            The same temporal scene graph with COO sparse matrices.
        """
        return cls(
            edge_radius=temporal_scene_graph.edge_radius,
            nodes=temporal_scene_graph.nodes,
            adj_cube=temporal_scene_graph.adj_cube,
            weight_cube=temporal_scene_graph.weight_cube,
            node_type_mat=temporal_scene_graph.node_type_mat,
            edge_scaling=temporal_scene_graph.edge_scaling
        )

    def to_scene_graph(
            self,
            t: int,
            t_hist: int = 0,
            t_fut: int = 0
    ) -> SparseSceneGraph:
        lower_t = max(0, t - t_hist)
        higher_t = min(t + t_fut + 1, len(self.adj_cube) + 1)
        adj_mat = self.adj_cube[lower_t:higher_t].max(axis=0)
        weight_mat = self.weight_cube[lower_t:higher_t].max(axis=0)

        return SparseSceneGraph(
            self.edge_radius,
            self.nodes,
            adj_mat,
            weight_mat,
            self.node_type_mat,
            self.node_index_lookup,
            self.edge_scaling[t]
        )
