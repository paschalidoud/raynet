import os
import sys

import numpy as np

from collections import namedtuple
from itertools import combinations

from ..utils.generic_utils import point_from_depth, pixel_to_ray,\
    point_to_voxel
from ..utils.geometry import point_in_aabbox
from ..ray_marching.ray_tracing import voxel_traversal


SampleFromImage = namedtuple(
    "SampleFromImage",
    ["img_idx", "patch_x", "patch_y", "points", "target"]
)
Sample = namedtuple(
    "Sample",
    ["scene_idx", "img_idx", "patch_x", "patch_y", "points", "X", "y"]
)
RayNetSample = namedtuple(
    "RayNetSample",
    ["scene_idx", "img_idx", "patch_x", "patch_y", "points", "X", "y", "Nr",
     "ray_voxel_indices", "camera_center"]
)


def create_combinations_of_patches(patches, n_pairs=2):
    """Just N choose n_pairs from the patches list."""
    # return list(map(list, combinations(patches, n_pairs)))
    return [
        list(p)
        for p in combinations(patches, n_pairs)
    ]


def is_empty(x):
    """Checks if x is all filled with -1"""
    return x.sum() == -np.prod(x.shape)


class SampleGenerator(object):
    """Class used to create the input for Keras models."""
    def __init__(
        self,
        sampling_scheme,
        generation_params,
        scenes_range,
        input_shapes,
        output_shapes,
        repeat_from_same_scene=1000
    ):
        self._sampling_scheme = sampling_scheme
        self._generation_params = generation_params
        # The range of scenes that will be used to sample points from
        self._scenes_range = scenes_range
        # Lists of lists (or tuples of tuples, who cares?) that contain the
        # shape of each input and output (without the batch size). The number
        # of inputs is simply len(input_shapes) etc.
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

        # Number of times to take samples from the same scene
        self._repeat_from_same_scene = repeat_from_same_scene
        # Initialize counter to something huge to make sure that the first time
        # the get_sample() is called it will be able to select a scene.
        self._cnt_same_scenes = sys.maxint

        # The state of the sample generator
        self._scene_idx = 0

    def set_sampling_scheme(self, sampling_scheme):
        self._sampling_scheme = sampling_scheme

    def compute_X(self, images, points, y, target):
        raise NotImplementedError()

    def compute_y(self, points, target):
        return self._generation_params.target_distribution_factory(
            np.vstack([target, [1]]),
            points
        )

    @property
    def generation_params(self):
        return self._generation_params

    def _compute_patches_from_point(self, images, point):
        expand_patch = self._generation_params.expand_patch
        # Project the point specified by the point_index in all images and get
        # the corresponding patches
        patches = [
            im.patch_from_3d(
                point.reshape(-1, 1),
                self._generation_params.patch_shape[:2],
                expand_patch
            )
            for im in images
        ]

        # Check if we need to bail because some point is projected outside
        # of some image
        if not expand_patch and any(map(is_empty, patches)):
            return None

        return patches

    def _compute_patches_from_points(self, images, points):
        """Given a set of images and a set of points project the points to the
        image planes and take the corresponding patches.
        """
        # Create some local variables for convenience
        patch_shape = self._generation_params.patch_shape
        views = self._generation_params.neighbors + 1
        depth_planes = self._generation_params.depth_planes

        # Specify the dimensionality of the input
        shape = (views, depth_planes) + patch_shape

        X = np.empty(shape=shape, dtype=np.float32)

        # Fill the return buffer depth plane by depth plane (which means one
        # point at a time)
        for im_idx, im in enumerate(images):
            patches = im.patches_from_3d_points(
                points,
                self._generation_params.patch_shape[:2]
            )
            # Check if we need to bail because some point is projected outside
            # of some image
            if patches is None:
                return None

            X[im_idx] = np.array(patches).reshape(
                (depth_planes,) + patch_shape
            )

        return X

    def _get_sample_from_image_idx(self, scene, img_idx):
        """Given a scene_idx and a image_idxe generate a sample
        """
        # Compute the multi-view Image object
        images = scene.get_image_with_neighbors(
            img_idx,
            self._generation_params.neighbors
        )
        # Randomly select a pixel from the reference image
        ref_img = images[0]
        # Extract the pixel coordinates from the (3, 1) array
        px, py, _ = ref_img.random_pixel()[:, 0]
        # print "scene_idx:", scene_idx, " img_idx:", img_idx
        # print px, py
        bs = self._get_sample_from_patch_idx(scene, ref_img, img_idx, px, py)
        return bs, images

    def _get_sample_from_patch_idx(self, scene, ref_img, img_idx, px, py):
        # Get the depth value for this pixel, namely the distance from the
        # camera center
        depth = scene.get_depth_for_pixel(img_idx, py, px)
        # Check whether there is empty space there
        if depth is None or depth == 0:
            return SampleFromImage(
                img_idx=img_idx,
                patch_x=px,
                patch_y=py,
                points=None,
                target=None
            )

        # Associate this depth value with a 3D point in space
        origin, direction = ref_img.ray(np.vstack([px, py, [1]]))
        target = point_from_depth(
            origin[:-1],
            direction[:-1] - origin[:-1],
            depth
        )
        # print "target:", target
        # Make sure that the target point is inside the bounding box.
        # TODO: This should be removed because it should not be happening
        bbox = scene.bbox
        if not point_in_aabbox(
            target,
            bbox[0, :3].reshape(-1, 1),
            bbox[0, 3:].reshape(-1, 1)
        ):
            return SampleFromImage(
                img_idx=img_idx,
                patch_x=px,
                patch_y=py,
                points=None,
                target=None
            )

        # Sample the 3D points that will discretize the viewing ray
        points = self._sampling_scheme.sample_points_across_ray(
            scene,
            img_idx,
            py,
            px
        )
        # print "points:", points
        if points is None:
            return SampleFromImage(
                img_idx=img_idx,
                patch_x=px,
                patch_y=py,
                points=None,
                target=target
            )

        return SampleFromImage(
            img_idx=img_idx,
            patch_x=px,
            patch_y=py,
            points=points,
            target=target
        )

    def get_sample(self, dataset):
        # Pick randomly a scene from the dataset
        if self._cnt_same_scenes > self._repeat_from_same_scene:
            self._scene_idx = np.random.choice(self._scenes_range)
            self._cnt_same_scenes = 0

        scene = dataset.get_scene(self._scene_idx)
        self._cnt_same_scenes += 1
        # Pick randomly an image from the scene
        img_idx = np.random.choice(np.arange(2, scene.n_images))

        # Keep selecting pixels from the same image until we have a valid
        # sample. This is done to make the sample-generation a bit faster
        bs, images = self._get_sample_from_image_idx(
            scene,
            img_idx
        )
        # In case the target point is None we want to ignore this point
        if bs.target is None:
            return Sample(
                scene_idx=self._scene_idx,
                img_idx=img_idx,
                patch_x=bs.patch_x,
                patch_y=bs.patch_y,
                points=bs.points,
                X=None,
                y=None
            )

        # Compute the target distribution based on a factory
        y = self.compute_y(bs.points, bs.target)

        # Project points to images and get the surrounding patches
        X = self.compute_X(images, bs.points, y, bs.target)

        return Sample(
            scene_idx=self._scene_idx,
            img_idx=img_idx,
            patch_x=bs.patch_x,
            patch_y=bs.patch_y,
            points=bs.points,
            X=X,
            y=[y]
        )


class DefaultSampleGenerator(SampleGenerator):
    def compute_X(self, images, points, y, target):
        # Reproject and extract patches for images, points
        # patches.shape == (views, depth planes) + patch shape
        patches = self._compute_patches_from_points(images, points)
        # Check for valid patches
        if patches is None:
            return None

        X = np.array(create_combinations_of_patches(
            patches,
            len(self.input_shapes)
        )).transpose([1, 2, 0, 3, 4, 5])

        return list(X)


class CompareWithReferenceSampleGenerator(SampleGenerator):
    def compute_X(self, images, points, y, target):
        # Reproject and extract patches for images, points
        # patches.shape == (views, depth planes) + patch shape
        patches = self._compute_patches_from_points(images, points)
        # Check for valid patches
        if patches is None:
            return None

        X = np.array([
            [patches[0], p]
            for p in patches[1:]
        ]).transpose([1, 2, 0, 3, 4, 5])

        return list(X)


class HartmannSampleGenerator(SampleGenerator):
    """Class used to generate samples according to the Hartmann et al. paper.
    For more information please refer to
    https://arxiv.org/abs/1703.08836
    """
    def _get_positive_index(self, target_distribution):
        return np.argmax(target_distribution)

    def _get_negative_index(self, target_distribution):
        # Get the index of the point that is the correct one based on the gt
        pos_idx = self._get_positive_index(target_distribution)
        total_depths = np.arange(self._generation_params.depth_planes)

        # Remove the index for the gt depth
        step_depth = self._generation_params.step_depth
        depth_planes = self._generation_params.depth_planes
        new_depths = np.delete(
            total_depths,
            range(
                max(0, -step_depth + pos_idx),
                min(step_depth + pos_idx, depth_planes)
            )
        )

        return np.random.choice(new_depths)

    def compute_y(self, points, target):
        # Randomly choose to generate a positive or a negative training sample
        if np.random.random() > 0.5:
            return np.array([1., 0.], dtype=np.float32).reshape(1, 1, 2)
        else:
            return np.array([0., 1.], dtype=np.float32).reshape(1, 1, 2)

    def compute_X(self, images, points, y, target):
        target_distribution =\
            self._generation_params.target_distribution_factory(
                np.vstack([target, [1]]),
                points
            )
        if y[0, 0, 0] == 1:
            idx = self._get_positive_index(target_distribution)
        else:
            idx = self._get_negative_index(target_distribution)

        X = self._compute_patches_from_point(
            images,
            points[idx]
        )
        if X is None:
            return None
        else:
            return np.array(X)


class RayNetSampleGenerator(SampleGenerator):
    """Class used to generate samples for training our RayNet model.

    TODO: Document the arguments
    """
    def __init__(
        self,
        sampling_scheme,
        generation_params,
        scenes_range,
        input_shapes,
        output_shapes,
        n_rays=10000,
        window=4
    ):
        super(RayNetSampleGenerator, self).__init__(
            sampling_scheme,
            generation_params,
            scenes_range,
            input_shapes,
            output_shapes
        )

        # Generate samples from the same reference image
        self._window = window
        self._n_rays = n_rays
        self._rays_cnt = 0

        # The state of the sample generator
        self._scene_idx = 0
        self._img_idx = 2

    def compute_X(self, images, points, y, target):
        return self._compute_patches_from_points(images, points)

    def compute_y(self, points, target):
        raise NotImplementedError()

    def _get_sample(self, scene, scene_idx, img_idx):
        # Create local variables for convenience
        bbox = scene.bbox
        bs, images = self._get_sample_from_image_idx(
            scene,
            img_idx
        )
        # In case the target point is None we want to ignore this point
        if bs.target is None:
            return RayNetSample(
                scene_idx=scene_idx,
                img_idx=img_idx,
                patch_x=bs.patch_x,
                patch_y=bs.patch_y,
                points=bs.points,
                X=None,
                y=None,
                Nr=None,
                ray_voxel_indices=None,
                camera_center=images[0].camera.center
            )

        # Project points to images and get the surrounding patches
        X = self.compute_X(images, bs.points, None, None)
        if X is None:
            return RayNetSample(
                scene_idx=scene_idx,
                img_idx=img_idx,
                patch_x=bs.patch_x,
                patch_y=bs.patch_y,
                points=bs.points,
                X=None,
                y=None,
                Nr=None,
                ray_voxel_indices=None,
                camera_center=images[0].camera.center
            )

        # Allocate space to hold the ray_voxel indices for the current ray
        ray_voxel_indices = np.zeros(
            (self._generation_params.max_number_of_marched_voxels, 3),
            dtype=np.int32
        )
        # Bring the grid shape in the proper format
        grid_shape = np.array(
            self._generation_params.grid_shape
        ).astype(np.int32)
        Nr = voxel_traversal(
            bbox.ravel(),
            grid_shape,
            ray_voxel_indices,
            bs.points[0, :-1],
            bs.points[-1, :-1]
        )
        # If the ray is not intersection with the voxel grid ignore this sample
        # TODO: Is this supposed to happen?
        if Nr == 0:
            return RayNetSample(
                scene_idx=scene_idx,
                img_idx=img_idx,
                patch_x=bs.patch_x,
                patch_y=bs.patch_y,
                points=bs.points,
                X=None,
                y=None,
                Nr=None,
                ray_voxel_indices=None,
                camera_center=images[0].camera.center
            )

        # Generate the target distribution in voxel space to train the network
        bin_size = (bbox[0, 3:].T - bbox[0, :3].T) / grid_shape
        # Transform the 3D point to a point in the voxel_coordinate system and
        # get the voxel, where this point lies
        v = point_to_voxel(bs.target, bbox[:, :3].T, bin_size.reshape(-1, 1))
        # Compute the closest voxel center from the voxels through which the
        # ray intersects
        voxel_idx = np.abs(ray_voxel_indices - v.T).sum(axis=-1).argmin()
        y = np.zeros(
            (self._generation_params.max_number_of_marched_voxels,),
            dtype=np.float32
        )
        # Give 1.0 to the correct depth
        y[voxel_idx] = 1.0

        # Keep track of the generated samples
        self._rays_cnt += 1

        return RayNetSample(
            scene_idx=scene_idx,
            img_idx=img_idx,
            patch_x=bs.patch_x,
            patch_y=bs.patch_y,
            points=bs.points,
            X=X,
            y=y,
            Nr=Nr,
            ray_voxel_indices=ray_voxel_indices,
            camera_center=images[0].camera.center
        )
    
    def get_sample(self, dataset):
        scene_idx = self._scenes_range[self._scene_idx]
        scene = dataset.get_scene(scene_idx)

        img_idx_start = self._img_idx
        # Scene and img are such that the following always exist (this is
        # enforced in the end of the function while we update the indices)
        img_idx = img_idx_start + int(np.random.rand()*self._window)

        raynet_sample = self._get_sample(scene, scene_idx, img_idx)

        if self._rays_cnt >= self._n_rays:
            self._rays_cnt = 0
            # Proceed to the next image from the same scene
            self._img_idx += 2
            # If we have passed through all the images from the scene start
            # generating samples from the next scene
            if self._img_idx >= scene.n_images - self._window:
                self._img_idx = 2
                self._scene_idx += 1
            if self._scene_idx >= len(self._scenes_range):
                self._scene_idx = 0

        return raynet_sample


class RayNetRandomSampleGenerator(RayNetSampleGenerator):
    """Class used to generate samples for training our RayNet model.

    TODO: Document the arguments
    """
    def __init__(
        self,
        sampling_scheme,
        generation_params,
        scenes_range,
        input_shapes,
        output_shapes,
        n_rays=10000,
        window=4
    ):
        super(RayNetRandomSampleGenerator, self).__init__(
            sampling_scheme,
            generation_params,
            scenes_range,
            input_shapes,
            output_shapes
        )

        # Generate samples from the same reference image
        self._window = window
        self._n_rays = n_rays
        self._rays_cnt = 0

        # The state of the sample generator
        self._scene_idx = 0
        self._img_idx = 2

    def get_sample(self, dataset):
        scene_idx = self._scenes_range[self._scene_idx]
        scene = dataset.get_scene(scene_idx)

        img_idx = np.random.choice(
            np.arange(2, scene.n_images - self._window)
        )

        raynet_sample = self._get_sample(scene, scene_idx, img_idx)
        if self._rays_cnt >= self._n_rays:
            self._rays_cnt = 0

            # Randomly select the new scene for the next batch
            self._scene_idx = np.random.choice(
                np.arange(len(self._scenes_range)
            ))

        return raynet_sample
