import os

import numpy as np
from sklearn.neighbors import KDTree

from pointcloud import PointcloudFromDepthMaps, Pointcloud,\
    PointcloudFromDepthMapsWithConsistency
from utils.geometry import keep_points_in_aabbox


class FiltersFactory(object):
    def __init__(self, filters):
        self.filters = filters

    @property
    def has_filters(self):
        return len(self.filters) > 0

    def filter(self, X):
        # Apply all filters one by one to the input data
        for f in self.filters:
            X = f.filter(X)

        return X


class VoxelMask(object):
    """A simple mask to filter points in a voxel grid.

    Arguments
    ---------
        bbox: (1, 6) The 3D bounding box
        mask: (A, B, C) The mask defining the grid shape and filters
    """
    def __init__(self, bbox, mask, output_directory=None):
        assert bbox.shape == (1, 6)
        assert np.all(bbox[0, :3] < bbox[0, 3:])
        self._bbox_min = bbox[0, :3, np.newaxis]
        self._bbox_max = bbox[0, 3:, np.newaxis]
        self._grid_shape = np.array(mask.shape).reshape(3, 1)
        self._mask = mask
        # Compute the size of each voxel in the voxel grid in each dimension
        self._steps = (self._bbox_max - self._bbox_min) / self._grid_shape

        self.output_directory = output_directory

    def _to_file(self, X):
        # If we have specified an output directory
        if self.output_directory is not None:
            p = Pointcloud(X)
            p.save_ply(
                os.path.join(self.output_directory, "pc_inside_voxel_mask.ply")
            )

    def filter(self, X):
        """Return a subset of X that is filtered by the voxel mask"""
        assert X.shape[0] == 3

        # Decimate all points that are outside the bbox
        points = keep_points_in_aabbox(X, self._bbox_min, self._bbox_max)

        # Associate every 3D point to a voxel index. First subtract from all
        # points the bbox_min so that the (0, 0, 0) of the voxel coordinate
        # sysem coincides with the bbox_min. Divide with the size of each voxel
        # to associte with the correct index
        idxs = np.round((points - self._bbox_min - self._steps / 2) / self._steps).astype(int)
        points = points[:, self._mask[idxs[0], idxs[1], idxs[2]] == 1]

        print "Filter out %d out of %d points" % (
            X.shape[1] - points.shape[1], X.shape[1])

        # Save to file, if we have specified an output directory
        self._to_file(points)

        return points


class ReduceDensity(object):
    """A wrapper to perform density reduction based on a minimum distance
    """
    def __init__(self, min_dist, output_directory=None):
        self._min_dist = min_dist

        self.output_directory = output_directory

    def _to_file(self, X):
        # If we have specified an output directory
        if self.output_directory is not None:
            p = Pointcloud(X)
            p.save_ply(
                os.path.join(self.output_directory, "pc_after_density_reduction.ply")
            )

    def filter(self, X):
        """Return a subset of X that is filtered based on the density of the
        points
        """
        assert X.shape[0] == 3
        index_set = np.ones(X.shape[1], dtype=np.bool)
        # Randomly shuffle the points (because they also do it in the
        # evaluation code for the DTU dataset)
        rand_ord = np.arange(X.shape[1])
        np.random.shuffle(rand_ord)

        # Create a KDTree with the points
        t = KDTree(X.T)
        # Get the indices of all points that have distance smaller than the
        # self._min_dist
        idx = t.query_radius(X[:, rand_ord].T, self._min_dist)

        # For every query point get all the points that have distance smaller
        # than self._min_dist
        for _id, i in zip(idx, rand_ord):
            # If the query point belongs in the set remove all the points from
            # the set and add again the query point because we just removed it
            # :-)
            if index_set[i]:
                index_set[_id] = 0
                index_set[i] = 1

        print "Filter out %d out of %d points" % (
            X.shape[1] - X[:, index_set].shape[1], X.shape[1])

        # Save to file, if we have specified an output directory
        self._to_file(X[:, index_set])

        return X[:, index_set]


class Metric(object):
    def compute(self, scene, frame_idxs, depthmaps, predicted_pointcloud):
        raise NotImplementedError()


class PerPixelMeanDepthError(Metric):
    def __init__(self, borders=40):
        self.borders = borders

    def compute(self, scene, frame_idxs, depthmaps, predicted_pointcloud):
        metric = np.zeros((len(frame_idxs),))

        H, W = scene.image_shape
        bordersH = slice(self.borders, H-self.borders)
        bordersW = slice(self.borders, W-self.borders)
        for i, (fi, d) in enumerate(zip(frame_idxs, depthmaps)):
            G = scene.get_depth_map(fi)[bordersH, bordersW]
            D = np.load(d)[bordersH, bordersW]
            pixels = G != 0

            metric[i] = np.abs(G[pixels] - D[pixels]).mean()

        return metric, None


class Accuracy(Metric):
    def __init__(
        self,
        filter_factory=None,
        truncate=float("inf"),
        borders=40,
        use_pc_from_depthmap=False
    ):
        self.filter_factory = filter_factory
        self.truncate = truncate
        self.borders = borders
        self.use_pc_from_depthmap = use_pc_from_depthmap

    def compute(self, scene, frame_idxs, depthmaps, predicted_pointcloud):
        # Load the point clouds and filter them
        if self.use_pc_from_depthmap:
            # Estimate the gt point cloud from a set of gt images
            gt_depthmaps = [
                scene.get_depthmap_file(i)
                for i in frame_idxs
            ]
            ground_truth_pc = PointcloudFromDepthMaps(
                scene,
                frame_idxs,
                gt_depthmaps,
                self.borders
            )
        else:
            # Load the gt-pointcloud as it
            ground_truth_pc = scene.get_pointcloud()

        if self.filter_factory.has_filters:
            ground_truth_pc.filter(self.filter_factory)
            predicted_pointcloud.filter(self.filter_factory)

        ground_truth_pc.index()
        distances, indexes = ground_truth_pc.nearest_neighbors(
            predicted_pointcloud.points
        )

        return np.minimum(distances, self.truncate), predicted_pointcloud.points


class Completeness(Metric):
    def __init__(
        self,
        filter_factory=None,
        truncate=float("inf"),
        borders=40,
        use_pc_from_depthmap=False
    ):
        self.filter_factory = filter_factory
        self.truncate = truncate
        self.borders = borders
        self.use_pc_from_depthmap = use_pc_from_depthmap

    def compute(self, scene, frame_idxs, depthmaps, predicted_pointcloud):
        # Load the point clouds and filter them
        if self.use_pc_from_depthmap:
            gt_depthmaps = [
                scene.get_depthmap_file(i)
                for i in frame_idxs
            ]
            ground_truth_pc = PointcloudFromDepthMaps(
                scene,
                frame_idxs,
                gt_depthmaps,
                self.borders
            )
        else:
            ground_truth_pc = scene.get_pointcloud()

        if self.filter_factory.has_filters:
            ground_truth_pc.filter(self.filter_factory)
            predicted_pointcloud.filter(self.filter_factory)

        predicted_pointcloud.index()
        distances, indexes = predicted_pointcloud.nearest_neighbors(
            ground_truth_pc.points
        )

        return np.minimum(distances, self.truncate), ground_truth_pc.points
