import os
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import numpy as np
from scipy.io import loadmat

from image import Image
from parse_input_data import parse_scene_info, parse_gt_mesh,\
    parse_gt_data, parse_scene_info_dtu_dataset, parse_stl_file_to_pointcloud

from ..pointcloud import Pointcloud
from ..utils.geometry import project, distance
from ..utils.generic_utils import get_voxel_grid
from ..utils.oct_tree import OctTree
from ..utils.training_utils import get_adjacent_frames_idxs,\
    get_ray_meshes_first_intersection


class Scene(object):
    """Scene class is a wrapper for every scene in a dataset.

    A scene is defined as a collection of images, camera_poses, ground-truth
    data and a bounding box that specifies the borders of the scene.
    """
    def __init__(self, select_neighbors_based_on="filesystem"):
        self._voxel_grid = None
        # Keep track of the neighboring cameras in space in terms of distance
        self._camera_neighbors = None
        self._select_neighbors_based_on = select_neighbors_based_on

    @staticmethod
    def _load_sorted_files(basepath, dir, condition=None):
        path = os.path.join(basepath, dir)
        return [
            os.path.join(path, f)
            for f in sorted(filter(condition, os.listdir(path)))
        ]

    def _get_neighbor_idxs(self, i, neighbors):
        if self._select_neighbors_based_on == "distance":
            return self._get_adjacent_camera_centers(
                neighbors,
                dict(zip(range(self.n_images), range(self.n_images))),
                0
            )[i]
        elif self._select_neighbors_based_on == "filesystem":
            return get_adjacent_frames_idxs(
                i,
                self.n_images,
                neighbors,
                0
            )
        else:
            raise NotImplementedError()

    def _get_adjacent_camera_centers(
        self,
        neighbors,
        frame_idxs=None,
        skip=0
    ):
        if self._camera_neighbors is None:
            camera_centers = np.hstack([
                self.get_image(i).camera.center
                for i in frame_idxs
            ])
            a = camera_centers
            distances = \
                ((a.T[:, :, np.newaxis] - a[np.newaxis])**2).sum(axis=1)
            self._camera_neighbors = \
                distances.argsort()[:, 1:neighbors+1:skip+1]

        return self._camera_neighbors

    @property
    def bbox(self):
        raise NotImplementedError()

    @property
    def n_images(self):
        raise NotImplementedError()

    @property
    def image_shape(self):
        im = self.get_image(0)
        return im.height, im.width

    @property
    def observation_mask(self):
        return None

    @property
    def gt_depth_range(self):
        D = self.get_depth_map(0)
        return np.min(D[D != 0]), np.max(D)

    def get_image(self, i):
        raise NotImplementedError()

    def get_images(self):
        return [self.get_image(i) for i in range(self.n_images)]

    def get_random_image(self):
        img_idx = np.random.choice(np.arange(0, self.n_images))
        return self.get_image(img_idx)

    def get_image_with_neighbors(self, i, neighbors=4):
        # First return the image specified with index i and then its neighbors
        return [self.get_image(i)] + [
            self.get_image(n)
            for n in self._get_neighbor_idxs(i, neighbors)
        ]

    def get_depth_for_pixel(self, i, y, x):
        raise NotImplementedError()

    def get_depth_map(self, i):
        image_shape = self.image_shape
        dm = np.zeros(image_shape, dtype=np.float32)
        for x in range(image_shape[1]):
            for y in range(image_shape[0]):
                dm[y, x] = self.get_depth_for_pixel(i, y, x)

    def get_depth_maps(self):
        return [self.get_depth_map(i) for i in range(self.n_images)]

    def get_depthmap_file(self, i):
        return None

    def get_pointcloud(self):
        raise NotImplementedError()

    def voxel_grid(self, grid_shape):
        if self._voxel_grid is None:
            if self.bbox is None:
                raise Exception("bbox needs to be different than None")
            self._voxel_grid = get_voxel_grid(self.bbox, grid_shape)
        return self._voxel_grid.astype(np.float32)


class RestrepoScene(Scene):
    """Wrapper for a RestrepoScene
    """
    def __init__(self, basepath, select_neighbors_based_on="filesystem"):
        super(RestrepoScene, self).__init__(select_neighbors_based_on)
        # Path to the directory containing
        self._basepath = basepath
        self._image_paths = self._load_sorted_files(self._basepath, "imgs")
        self._cam_paths = self._load_sorted_files(self._basepath, "cams_krt")
        self._bbox_path = os.path.join(self._basepath, "scene_info.xml")

        self._bbox = None
        self._oct_tree = None
        self._cache = [None]*len(self._image_paths)
        # TODO: Add caching for the depth map computation
        # self._cache_depth_maps = [None]*len(self._image_paths)

    @property
    def n_images(self):
        return len(self._image_paths)

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = parse_scene_info(self._bbox_path)
        return self._bbox

    def get_image(self, i):
        if self._cache[i] is None:
            self._cache[i] = Image.from_file(
                self._image_paths[i],
                self._read_camera_poses(i)
            )
        return self._cache[i]

    def _has_gt_depth(self, i):
        gt_dir = os.path.join(self._basepath, "gt")
        gt_file = os.path.join(gt_dir, "gt_depth_%d.npy" % (i,))
        return (
            os.path.isdir(gt_dir) and
            os.path.isfile(gt_file)
        )

    def get_depth_for_pixel(self, i, y, x):
        # Get the image with index i
        im = self.get_image(i)
        # Compute the origin and the destination of the ray passing through the
        # camera center and the pixel (y, x)
        origin, destination = im.ray(np.array([[x, y, 1.]], dtype=np.int32).T)
        # Compute the 3D point that corresponds to this pixel
        target_point = get_ray_meshes_first_intersection(
            origin,
            destination,
            self._get_oct_tree()
        )
        if target_point is None:
            return None

        return distance(target_point[:-1], im.camera.center[:-1])

    def get_depthmap_file(self, i):
        if not self._has_gt_depth(i):
            return None

        gt_dir = os.path.join(self._basepath, "gt")
        gt_file = os.path.join(gt_dir, "gt_depth_%d.npy" % (i,))

        return gt_file

    def get_depth_map(self, i):
        if not self._has_gt_depth(i):
            super(RestrepoScene, self).get_depth_map(i)

        return np.load(self.get_depthmap_file(i))

    def _read_camera_poses(self, i):
        """Read the camera poses saved in the Restrepo format and export the K,
        R, t matrices as a dictionary."""
        with open(self._cam_paths[i]) as f:
            lines = f.readlines()

        l = [x.strip().split(" ") for x in lines if x != "\n"]

        # Make a dictionary that will hold the camera posses
        camera_poses = {}

        # K is the 3x3 instrinsic camera matrix
        K = np.array(l[0:3]).astype(np.float32)
        camera_poses["K"] = K

        # R is the 3x3 rotation matrix
        R = np.array(l[3:-1]).astype(np.float32)
        camera_poses["R"] = R

        # t is the 3x1 translation matrix
        t = np.array(l[-1]).astype(np.float32).reshape(-1, 1)
        camera_poses["t"] = t

        return camera_poses

    def _get_oct_tree(self):
        if self._oct_tree is None:
            triangles = parse_gt_mesh(self._basepath)
            print "Building the octree for the current scene. Be patient..."
            self._oct_tree = OctTree(triangles)

        return self._oct_tree

    def get_pointcloud(self):
        points, _, _ = parse_gt_data(self._basepath)
        return Pointcloud(points.T)


class DTUScene(Scene):
    """Wrapper for a DTUScene
    """
    def __init__(
        self,
        basepath,
        scene_idx,
        illumination="max",
        select_neighbors_based_on="filesystem"
    ):
        super(DTUScene, self).__init__(select_neighbors_based_on)
        # Path to the directory containing
        self._basepath = basepath

        self._image_paths = self._load_sorted_files(
            basepath,
            os.path.join("Rectified", "scan%03d" % (scene_idx,)),
            lambda i: illumination in i
        )
        # Filter out frames that are larger than 49, because we only have depth
        # maps for 49 frames
        # TODO: Generate gt depth maps for all images
        self._image_paths = [
            ip
            for ip in self._image_paths
            if int(ip.split("/")[-1].split(".")[0].split("_")[1]) <= 49
        ]
        self._cam_paths = self._load_sorted_files(
            basepath,
            "SampleSet/MVS_Data/Calibration/cal18",
            lambda i: "pos" in i
        )
        self._cam_intrinsic_path = os.path.join(
            basepath,
            "SampleSet/MVS_Data/Calibration/cal18/intrinsic.txt"
        )
        self._bbox_path = os.path.join(
            basepath,
            "SampleSet/MVS_Data/ObsMask/",
            "ObsMask%d_10.mat" % (scene_idx,)
        )
        self._depth_map_paths = self._load_sorted_files(
            basepath,
            os.path.join("Depth", "scan%03d" % (scene_idx,)),
            lambda i: i.endswith("npy")
        )
        self._gt_stl_path = os.path.join(
            basepath,
            "Points/stl/stl%03d_total.ply" % (scene_idx,)
        )

        self._bbox = None
        self._cache = [None]*len(self._image_paths)
        self._cache_depth_maps = [None]*len(self._image_paths)

    @property
    def n_images(self):
        return len(self._image_paths)

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = parse_scene_info_dtu_dataset(
                self._bbox_path
            ).astype(np.float32)
        return self._bbox

    @property
    def observation_mask(self):
        return loadmat(self._bbox_path)["ObsMask"]

    def get_image(self, i):
        if self._cache[i] is None:
            self._cache[i] = Image.from_file(
                self._image_paths[i],
                self._read_camera_poses(i)
            )
        return self._cache[i]

    def _read_camera_poses(self, i):
        """Read the camera poses saved in the DTU format and export the K,
        R, t matrices as a dictionary.

        Returns:
        -------
        camera_poses: dictionary
            Dictionary containing the K, R and t matrices
        """
        # Read the intrinsic camera matrix
        lines = []
        with open(self._cam_intrinsic_path) as f:
            lines = f.readlines()

        l = [x.strip().split(" ") for x in lines]
        K = np.array(l[0:3]).astype(np.float32)

        # Make a dictionary that will hold the camera posses
        camera_poses = {}
        camera_poses["K"] = K

        # Read the projection matrix and calculate the Rotation and translation
        lines = []
        with open(self._cam_paths[i]) as f:
            lines = f.readlines()
        l = [x.strip().split(" ") for x in lines]
        P = np.array(l[0:4]).astype(np.float32)

        Rt = np.dot(np.linalg.inv(K), P)
        # Extract the rotation and translation matrix
        R = Rt[:, :3]
        camera_poses["R"] = R

        t = Rt[:, -1].reshape(-1, 1)
        camera_poses["t"] = t

        return camera_poses

    @lru_cache(maxsize=8)
    def get_gt_depth_map(self, i):
        # Load the ground-truth depth map
        return np.load(self._depth_map_paths[i])

    def _get_depth_map(self, i):
        if self._cache_depth_maps[i] is None:
            image = self.get_image(i)
            gt_depth_map = self.get_gt_depth_map(i)

            # Bring pixels in camera coordinates
            H, W, _ = image.image.shape
            pixels = np.array([
                [u, v, 1.] for u in range(W) for v in range(H)
            ], dtype=np.float32).T
            K = image.camera.K
            K_inv = np.linalg.inv(K)
            p_cc = np.dot(K_inv, pixels)

            p_cc = p_cc * gt_depth_map.T.reshape(1, -1)
            # Bring point in homogenous coordinates
            p_cc = np.vstack([
                p_cc,
                np.ones(p_cc.shape[1], dtype=np.float32)
            ])

            # Bring the point in world coordinates
            P = np.vstack([
                np.hstack([image.camera.R, image.camera.t]),
                np.array([0., 0., 0., 1.])
            ])
            target = project(np.linalg.inv(P), p_cc)

            # Compute the distance from the camera center
            D = np.sqrt(((target - image.camera.center.T)**2).sum(axis=-1))
            D = D.reshape(W, H).T
            D *= (gt_depth_map != 0)
            self._cache_depth_maps[i] = D.astype(np.float32)
        return self._cache_depth_maps[i]

    def get_depth_map(self, i):
        return self._get_depth_map(i)

    def get_depth_for_pixel(self, i, y, x):
        # depth_map = self._get_depth_map(i)
        # print depth_map[y, x]
        # return depth_map[y, x]

        # Load the ground-truth depth map
        gt_depth_map = self.get_gt_depth_map(i)
        depth_value = gt_depth_map[y, x]
        if depth_value == 0:
            return None

        # Get the image with index i
        im = self.get_image(i)
        # Bring pixel (patch center) in camera coordinates
        K = im.camera.K
        K_pinv = np.linalg.inv(K)
        p_cc = np.dot(K_pinv, np.array([[x, y, 1]], dtype=np.int32).T)

        p_cc = p_cc * depth_value
        # Bring point in homogenous coordinates
        p_cc = np.vstack((p_cc, np.array([1])))

        # Bring the point in world coordinates
        P = np.vstack([
            np.hstack([im.camera.R, im.camera.t]),
            np.array([0., 0., 0., 1.])
        ])
        target_point = project(np.linalg.inv(P), p_cc)
        if target_point is None:
            return None

        return distance(target_point[:-1], im.camera.center[:-1])

    def get_pointcloud(self):
        points = parse_stl_file_to_pointcloud(self._gt_stl_path)
        return Pointcloud(points.T)
