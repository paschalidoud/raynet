"""Create a point cloud from depth images and other point cloud related
functions"""

import sys

import numpy as np
from sklearn.neighbors import KDTree

from matplotlib.cm import get_cmap

from .utils.geometry import project


class Pointcloud(object):
    """A collection of ND (usually 3D) points that can be searched over and
    saved into a file."""
    def __init__(self, points):
        self._points = points

    @property
    def points(self):
        return self._points

    def save_ply(self, file):
        N = self.points.shape[1]
        with open(file, "w") as f:
            f.write(("ply\nformat binary_%s_endian 1.0\ncomment Raynet"
            " pointcloud!\nelement vertex %d\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n") % (sys.byteorder, N))
            self.points.T.astype(np.float32).tofile(f)

    def save_colored_ply(self, file, intensities, colormap="jet"):
        # Get the colormap based on the input
        cmap = get_cmap(colormap)
        # Based on the selected colormap get the the colors for every point
        intensities = intensities / 2
        colors = cmap(intensities.ravel())[:, :-1]
        # The color values need to be uchar
        colors = (colors * 255).astype(np.uint8)

        N = self.points.shape[1]
        # idxs = np.arange(N)[intensities.ravel() < 1.0]
        idxs = np.arange(N)
        with open(file, "w") as f:
            f.write(("ply\nformat binary_%s_endian 1.0\ncomment Raynet"
            " pointcloud!\nelement vertex %d\nproperty float x\n"
            "property float y\nproperty float z\nproperty uchar red\n"
            "property uchar green\nproperty uchar blue\nend_header\n") % (sys.byteorder, len(idxs)))
            cnt = 0
            # for point, color in zip(self.points.T, colors):
            for i in idxs:
                point = self.points.T[i]
                color = colors[i]
                point.astype(np.float32).tofile(f)
                color.tofile(f)
                cnt += 1

    def save(self, file):
        np.save(file, self.points)

    def filter(self, mask):
        self._points = mask.filter(self.points)

    def index(self, leaf_size=40, metric="minkowski"):
        if hasattr(self, "_index"):
            return

        # NOTE: scikit-learn expects points (samples, features) while we use
        # the more traditional (features, samples)
        self._index = KDTree(self.points.T, leaf_size, metric)

    def nearest_neighbors(self, X, k=1, return_distances=True):
        return self._index.query(X.T, k, return_distances)


class PointcloudFromDepthMaps(Pointcloud):
    """Create a point cloud from the depth prediction files and a scene.

    Arguments
    ---------
        scene: The scene for which we are creating a pointcloud
        frame_idxs: The image idx for the corresponding depthmap
        depthmaps: The files containing the predicted depthmaps
    """
    def __init__(self, scene, frame_idxs, depthmaps, borders=40):
        self._scene = scene
        self._frame_idxs = frame_idxs
        self._depthmaps = depthmaps
        self._borders = borders

        self._points = None

    def _remove_unwanted_points(self, G, D, R):
        """Remove unwanted points from the depth prediction.

        Arguments
        ---------
            G: (H, W) The ground truth depth
            D: (H, W) The predicted depth map
            R: (4, H*W) The rays from every pixel

        Return
        -------
            D: (1, N) filtered depth values
            R: (4, N) filtered rays
        """
        H, W = G.shape
        idxs = np.arange(H*W).reshape(W, H).T
        bordersH = slice(self._borders, H-self._borders)
        bordersW = slice(self._borders, W-self._borders)

        # Remove borders
        G = G[bordersH, bordersW]
        D = D[bordersH, bordersW]
        idxs = idxs[bordersH, bordersW]

        # Remove points where there is no ground truth
        Gmask = G != 0
        D = D[Gmask]
        idxs = idxs[Gmask]

        return D.reshape(1, -1), R[:, idxs.ravel()]

    def _generate_points_per_image(self, frame, predicted_depth_file):
        # Load the image and the depthmap and compute the rays
        image = self._scene.get_image(frame)
        depth = np.load(predicted_depth_file)

        # TODO: Should this happening? Maybe I should make sure that
        # this will never happen
        depth[np.isnan(depth)] = depth[~np.isnan(depth)].min()
        camera_center, rays = image.rays()

        # Remove unwanted rays that are too close to the borders or
        # that have no ground truth
        depth, rays = self._remove_unwanted_points(
            self._scene.get_depth_map(frame),
            depth,
            rays
        )

        # Compute the normalized directions of the rays
        directions = rays - camera_center
        norms = np.sqrt((directions**2).sum(axis=0, keepdims=True))
        assert np.all(directions[-1, :] == 0)

        return camera_center + depth * directions / norms

    @property
    def points(self):
        if self._points is None:
            _points = [
                self._generate_points_per_image(i, d)
                for i, d in zip(self._frame_idxs, self._depthmaps)
            ]
            _points = np.hstack(_points)
            self._points = _points[:-1, :]

        return self._points


class PointcloudFromDepthMapsWithConsistency(PointcloudFromDepthMaps):
    def __init__(self, scene, frame_idxs, depthmaps, borders=40,
                 consistency_threshold=0.75, n_neighbors=5):

        self._consistency_threshold = consistency_threshold
        self._n_neighbors = n_neighbors
        self._camera_neighbors = None
        self._frame_idxs_map = dict(zip(frame_idxs, range(len(frame_idxs))))

        super(PointcloudFromDepthMapsWithConsistency, self).__init__(
            scene,
            frame_idxs,
            depthmaps,
            borders
        )

    def _neighbor_frames(self, frame):
        """Return frame idxs and predicted depthmap files given a frame"""
        if self._camera_neighbors is None:
            camera_centers = np.hstack([
                self._scene.get_image(i).camera.center
                for i in self._frame_idxs
            ])
            a = camera_centers
            distances = 2*(a*a).sum(axis=0) - 2*(a.T.dot(a))
            self._camera_neighbors = distances.argsort()[:, 1:self._n_neighbors+1]

        return [
            (self._frame_idxs[i], self._depthmaps[i])
            for i in self._camera_neighbors[self._frame_idxs_map[frame]]
        ]

    def _update_tau(self, tau, predicted_depths, depths, mask):
        if tau is None:
            tau = np.abs(predicted_depths - depths)
        else:
            tau = np.maximum(
                np.abs(predicted_depths - depths),
                tau
            )
        tau[~mask] = float("inf")

        return tau

    def _generate_points_per_image(self, frame, predicted_depth_file):
        # Get the parent class bound to this object for calling parent functions
        _parent = super(PointcloudFromDepthMapsWithConsistency, self)

        # Get the points without any consistency check
        _points = _parent._generate_points_per_image(
            frame,
            predicted_depth_file
        )

        # Create a variable to hold the difference/variance between the
        # projections
        tau = None

        for i, d in self._neighbor_frames(frame):
            # Get the depths predicted from the i-th frame
            image = self._scene.get_image(i)
            pixels = project(
                image.camera.P,
                _points
            ).T
            x = np.round(pixels[0]).astype(np.int32)
            y = np.round(pixels[1]).astype(np.int32)
            valid = np.logical_and.reduce([
                0 <= x, x < image.width,
                0 <= y, y < image.height
            ])
            x[~valid] = 0
            y[~valid] = 0
            predicted_depths = np.load(d)
            predicted_depths = predicted_depths[y, x]

            # Compute the distances of the _points to the camera center
            depths = np.sqrt(((_points - image.camera.center)**2).sum(axis=0))

            # Update tau
            tau = self._update_tau(tau, predicted_depths, depths, valid)

        print (tau >= self._consistency_threshold).sum()
        return _points[:, tau < self._consistency_threshold]


def get_pointcloud(
    scene,
    frame_idxs,
    depthmaps,
    with_consistency,
    **kwargs
):
    if with_consistency:
        return PointcloudFromDepthMapsWithConsistency(
            scene,
            frame_idxs,
            depthmaps,
            kwargs["borders"],
            kwargs["consistency_threshold"],
            kwargs["n_neighbors"]
        )
    else:
        return PointcloudFromDepthMaps(
            scene,
            frame_idxs,
            depthmaps,
            kwargs["borders"]
        )
