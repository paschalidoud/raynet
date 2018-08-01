import os
from itertools import product
import numpy as np
import imageio

from camera import Camera
from ..utils.geometry import project


class Image(object):
    """Image contains the image buffer and the camera that produced it.
    """
    def __init__(self, camera, image_data, normalize=True):
        self._camera = camera
        self._image = image_data
        if len(self._image.shape) == 2:
            self._image = self._image[:, :, np.newaxis]

        # Normalize the patch so that the values are in [0,1]
        if normalize:
            self._image = self._image.astype(np.float32) / np.float32(255.)

    @classmethod
    def from_file(cls, image_file, camera_poses, mode="RGB"):
        """
        Arguments:
        ---------
        image_file: Path to the file containing the image data
        camera_poses: Dictionary containing the camera poses
        """
        image = imageio.imread(image_file)
        camera = Camera(
            K=camera_poses["K"],
            R=camera_poses["R"],
            t=camera_poses["t"]
        )

        return cls(camera, image)

    @property
    def image(self):
        return self._image

    @property
    def camera(self):
        return self._camera

    @property
    def width(self):
        return self._image.shape[1]

    @property
    def height(self):
        return self._image.shape[0]

    @property
    def channels(self):
        return self._image.shape[2]

    def random_pixel(self):
        return np.array([[
            np.random.randint(0, self.width),
            np.random.randint(0, self.height),
            1
        ]]).T

    def random_pixel_in_range(self, cropp):
        return np.array([[
            np.random.randint(cropp, self.width-cropp),
            np.random.randint(cropp, self.height-cropp),
            1
        ]]).T

    def rgb2gray(self):
        return Image(
            self.camera,
            np.dot(self.image[..., :3], [0.299, 0.587, 0.114])
        )

    def project(self, point):
        """Perform the projection of the 3d point on the image plane and get
        the corresponding 2D pixel in homogenous coordinate"""
        return np.round(project(self.camera.P, point)).astype(int)

    def patch_from_3d(self, point, patch_size, expand_patch=True):
        """Project the 3d point on the image plane and call the patch()
        method."""
        patch_center = self.project(point)

        return self.patch(patch_center, patch_size, expand_patch)

    def patch(self, patch_center, patch_size, expand_patch=True):
        """Return the image content around the patch_center.

        Arguments:
        ----------
            patch_center: array of shape (C+1, 1)
                The pixel coordinates
            patch_size: tuple
                The patch dimensions
            expand_patch: bool
                When set return also patches that are partially or
                entirely black if the point is outside the image
                boundaries
        """
        # x-axis corresponds to the columns of the image
        # y-axis corresponds to the rows of the image
        padding_x = int(patch_size[1]/2)
        padding_y = int(patch_size[0]/2)

        min_x = patch_center[0, 0] - padding_x
        max_x = patch_center[0, 0] + padding_x + patch_size[1] % 2
        min_y = patch_center[1, 0] - padding_y
        max_y = patch_center[1, 0] + padding_y + patch_size[0] % 2

        # Initialize the patch with 0.0
        patch = np.zeros(patch_size + self._image.shape[2:], dtype=np.float32)

        # Save some space by creating local copies with single letter names
        h, w = self.height, self.width

        # If the patch is inside the image boundaries return it as it is
        if min_x >= 0 and min_y >= 0 and max_x <= w and max_y <= h:
            patch[:, :] = self._image[min_y:max_y, min_x:max_x]

        # otherwise copy part (or nothing) from the image into the empty patch
        elif expand_patch:
            p_min_x = min(w, max(0, min_x))
            p_max_x = max(0, min(w, max_x))
            p_min_y = min(h, max(0, min_y))
            p_max_y = max(0, min(h, max_y))

            s_min_x = min(patch_size[1], max(0, 0 - min_x))
            s_max_x = max(0, min(patch_size[1], patch_size[1] + w - max_x))
            s_min_y = min(patch_size[0], max(0, 0 - min_y))
            s_max_y = max(0, min(patch_size[0], patch_size[0] + h - max_y))

            patch[s_min_y:s_max_y, s_min_x:s_max_x] = \
                self._image[p_min_y:p_max_y, p_min_x:p_max_x]
        else:
            patch.fill(-1.)

        return patch

    def patches_from_3d_points(self, points, patch_size):
        """Project an array of points to the image and take the patch around
        the projected patch.

        Arguments:
        ---------
            points: np.array(N, 4)
                The points to be projected
            patch_size: tuple
                The patch dimensions

        Return:
        ------
            A list of the patches
        """
        patch_centers = np.round(project(self.camera.P, points.T)).astype(int)

        patches = self.patches(patch_centers, patch_size)
        return patches

    def patches(self, patch_centers, patch_size):
        """Return the image content around the patch_center.

        Arguments:
        ----------
            patch_centers: array of shape (N, C+1)
                The pixel coordinates
            patch_size: tuple
                The patch dimensions
        """
        assert patch_centers.shape[0] > patch_centers.shape[1]
        # x-axis corresponds to the columns of the image
        # y-axis corresponds to the rows of the image
        padding_x = int(patch_size[1]/2)
        padding_y = int(patch_size[0]/2)

        min_x = patch_centers[:, 0] - padding_x
        max_x = patch_centers[:, 0] + padding_x + patch_size[1] % 2
        min_y = patch_centers[:, 1] - padding_y
        max_y = patch_centers[:, 1] + padding_y + patch_size[0] % 2

        # Save some space by creating local copies with single letter names
        h, w = self.height, self.width
        # Get the patch_centers that are inside the image boundaries
        patches_inside_boundaries = np.logical_and(
            np.logical_and(min_x >= 0, min_y >= 0),
            np.logical_and(max_x <= w, max_y <= h)
        )

        # If a single patch is outside the boundaries return None to avoid
        # useless computations
        if ~np.all(patches_inside_boundaries):
            return None

        # Initialize the patch with 0.0
        N = patch_centers.shape[0]
        patch_shape = (N,) + patch_size + self._image.shape[2:]
        patches = np.ones(patch_shape, dtype=np.float32)

        idxs = np.arange(N)
        for pi in idxs[patches_inside_boundaries]:
            patches[pi] = self._image[min_y[pi]:max_y[pi], min_x[pi]:max_x[pi]]

        return patches

    def ray(self, pixel):
        """Returns the ray connecting the camera center C and the point p.

        Given a 2D point p in an image, there exists a collection of 3D points
        that are mapped and projected onto the same point p. This collection of
        3D points constitutes a ray connecting the camera center C=(C_x, C_y,
        C_z) and the point p=(x, y, 1).

        We compute the 2D pixel coordinate of a 3D point using the projection
        matrix P = K[R|t]. In order to backproject a pixel to a 3D ray we have
        to use the pseudo inverse of the P matrix.

        Parameters:
        -----------
            pixel: Numpy array column vector of dimensions 2x1 or 3x1
                   correspoding to the pixel coordinates normal or homogenous
        Return:
        -------
            origin: 4x1, numpy array
                    The starting point of the ray
            destination: 4x1, numpy array
                         The  destination of the ray
        """
        # Ensure pixel is in homogenous coordinates
        if len(pixel) == 2:
            pixel = np.vstack((pixel, [1]))

        ray = project(self._camera.P_pinv, pixel.astype(np.float32))
        assert ray.shape == (4, 1)

        return self._camera.center, ray

    def rays(self):
        """Returns the set of rays for every pixel in the image

        Return:
        -------
            origin: 4x1, numpy array
                    The starting point of the ray
            rays: 4xN, numpy array, where N is the propduct of the width and
                  the height of the image
        """
        pixels = np.array([
            [u, v, 1.]
            for u, v in product(range(self.width), range(self.height))
        ], dtype=np.int32).T
        rays = project(self.camera.P_pinv, pixels)

        return self._camera.center, rays.T
