import numpy as np

from ..utils.geometry import ray_aabbox_intersection, ray_ray_intersection
from ..utils.generic_utils import get_voxel_grid
from ..ray_marching.ray_tracing import voxel_traversal
from ..tf_implementations.sampling_schemes import build_sample_points_in_bbox,\
    build_sample_points_in_range


class SamplingScheme(object):
    def __init__(self, generation_params):
        self.sampling_type = generation_params.sampling_type
        self.n_points = generation_params.depth_planes

    def _get_ray_from_pixel(self, scene, i, y, x):
        # Pixel should be in homogenous coordinates and we always use columns
        # as the the first cordinate (namely along the x-axis (width)) and rows
        # as the second coordinate (namely along the y-axis (height))
        pixel = np.array([[
            x,
            y,
            1
        ]]).T
        origin, destination = scene.get_image(i).ray(pixel)

        # Make sure that both the origin and the ray have size 4x1
        assert origin.shape == (4, 1)
        assert destination.shape == (4, 1)

        return origin, destination

    def _get_points_in_line(self, start, end, t):
        n = len(t)
        points = (start + t * (end - start)).T

        # Make sure that points have shape(n, 4)
        assert points.shape == (n, 4)
        # Return the sampled points as np.float32 to avoid numeric issues
        return points.astype(np.float32)

    def sample_points_across_ray(self, scene, i, y, x):
        """Sample n_points 3D points across the ray that emanates from the
        camera center and passes through the pixel defined as (x, y)

        Arguments:
        ----------
        scene: Scene object
            Scene object
        i: int
          Image index
        y: int
           The pixel value across columns
        x: int
           The pixel value across rows

        Returns:
        -------
        points: np.array(n_points, 4), dtype=np.float32
                The sampled points in homogenous coordinates
        """
        raise NotImplementedError()

    def sample_points_across_rays(self, scene, i):
        """Sample n_points 3D points across all rays from the image specified
        with index i in a vectorized manner

        Arguments:
        ----------
        scene: Scene object
            Scene object
        i: int
          Image index

        Returns:
        -------
        points: np.array(4, N, 32), dtype=np.float32, where N is the number of
                rays The sampled points in homogenous coordinates
        """
        raise NotImplementedError()

    def sample_points_across_rays_batched(self, scene, i, batch):
        """Sample n_points 3D points across all rays from the image specified
        with index i in a vectorized manner

        Arguments:
        ----------
        scene: Scene object
        i: int, Image index
        batch: slice, A slice object to define across which rays we will sample

        Returns:
        -------
        points: np.array(4, N, 32), dtype=np.float32, where N is the number of
                rays selected. The sampled points in homogenous coordinates.
        """
        raise NotImplementedError()


class SamplingInBboxScheme(SamplingScheme):
    def sample_points_across_ray(self, scene, i, y, x):
        n_points = self.n_points
        bbox = scene.bbox
        origin, destination = self._get_ray_from_pixel(scene, i, y, x)

        # Get the point of entry and exit to the bbox as `origin + t_(near|far)
        # * (destination - origin)`
        t_near, t_far = ray_aabbox_intersection(
            origin,
            destination,
            bbox[0, :3],
            bbox[0, 3:]
        )

        # assert t_near is not None, "The ray doesn't intersect with the bbox"
        if t_near is None or t_far is None:
            return None

        # Sample uniformly between t_near and t_far
        t = np.linspace(t_near, t_far, n_points, dtype=np.float32)
        return self._get_points_in_line(origin, destination, t)

    def _sample_points_across_rays(self, origin, directions, bbox):
        """Given a set of rays, defined through a origin and directions sample
           points in a range.

         Arguments:
         ---------
            origin: np.array((4, 1), dtype=np.float32)
            directions: np.array((4, N), dtype=np.float32)
            bbox: np.array((6,1 ), dtype=np.float32)
        """
        # Get the number of rays
        N = directions.shape[1]

        # Intersect the rays with the bbox
        t_near = np.ones(N) * float("-inf")
        t_far = np.ones(N) * float("inf")
        for i in range(3):
            t1 = (bbox[i, 0] - origin[i, 0]) / directions[i]
            t2 = (bbox[i+3, 0] - origin[i, 0]) / directions[i]
            np.maximum(np.minimum(t1, t2), t_near, out=t_near)
            np.minimum(np.maximum(t1, t2), t_far, out=t_far)

        # Sample points uniformly along the rays and inside the bbox
        points = np.array([
            origin +
            directions[:, i:i+1] * np.linspace(
                t_near[i],
                t_far[i],
                self.n_points
            ) for i in range(N)
        ])
        points = np.transpose(points, [1, 0, 2])

        return points.astype(np.float32)

    def sample_points_across_rays(self, scene, i):
        # Based on the image index get the camera center and transform all
        # pixels in the image to rays
        camera_center, rays = scene.get_image(i).rays()
        directions = rays - camera_center

        bbox = scene.bbox.T
        return self._sample_points_across_rays(camera_center, directions, bbox)

    def sample_points_across_rays_batched(self, scene, i, batch):
        # Based on the image index get the camera center and transform all
        # pixels in the image to rays
        camera_center, rays = scene.get_image(i).rays()
        directions = rays - camera_center

        bbox = scene.bbox.T
        # Extract the batch of directions
        directions = directions[:, batch]
        return self._sample_points_across_rays(camera_center, directions, bbox)


class SamplingInRangeScheme(SamplingScheme):
    def __init__(self, generation_params):
        super(SamplingInRangeScheme, self).__init__(generation_params)
        self._range = generation_params.depth_range

    def sample_points_across_ray(self, scene, i, y, x):
        t = np.linspace(
            self._range[0],
            self._range[1],
            self.n_points,
            dtype=np.float32
        )
        origin, destination = self._get_ray_from_pixel(scene, i, y, x)
        # Normalize the distance between the origin and the destination
        d = (destination - origin)
        d /= np.sqrt(np.sum(d**2))
        p = origin + t*d
        return p.T

    def _sample_points_across_rays(self, origin, directions):
        """Given a set of rays, defined through a origin and directions sample
           points in a range.

         Arguments:
         ---------
            origin: np.array((4, 1), dtype=np.float32)
            directions: np.array((4, N), dtype=np.float32)
        """
        # Sample the points
        sampling_range = np.linspace(
            self._range[0],
            self._range[1],
            self.n_points
        )[np.newaxis, np.newaxis, :].astype(np.float32)

        points = directions[:, :, np.newaxis] * sampling_range
        points += origin[:, :, np.newaxis]

        return points

    def sample_points_across_rays(self, scene, i):
        # Based on the image index get the camera center and transform all
        # pixels in the image to rays
        H, W = scene.image_shape
        camera_center, rays = scene.get_image(i).rays()
        directions = rays - camera_center
        directions /= np.sqrt(np.sum(directions**2, axis=0))

        return self._sample_points_across_rays(camera_center, directions)

    def sample_points_across_rays_batched(self, scene, i, batch):
        # Extract the normalized rays from the image
        camera_center, rays = scene.get_image(i).rays()
        directions = rays - camera_center
        directions /= np.sqrt(np.sum(directions**2, axis=0))

        # Extract the batch of directions
        directions = directions[:, batch]

        return self._sample_points_across_rays(camera_center, directions)


class SamplingInDisparityScheme(SamplingScheme):
    def sample_points_across_ray(self, scene, i, y, x):
        bbox = scene.bbox
        origin, destination = self._get_ray_from_pixel(scene, i, y, x)

        # Get the point of entry and exit to the bbox as `origin + t_(near|far)
        # * (destination - origin)`
        t_near, t_far = ray_aabbox_intersection(
            origin,
            destination,
            bbox[0, :3],
            bbox[0, 3:]
        )
        if t_near is None or t_far is None:
            return None

        # Compute the start and end point of the 3D line segment
        direction = destination - origin
        p_near = (origin + t_near * direction).T
        p_far = (origin + t_far * direction).T

        images = scene.get_image_with_neighbors(i)

        # Project the 3D line segment in the farmost view and get a 2D line
        # segment
        pixel_near = images[-1].project(p_near.T)[:-1]
        pixel_far = images[-1].project(p_far.T)[:-1]

        # Sample n_points pixels in the line segment
        t = np.linspace(start=0, stop=1, num=self.n_points, dtype=np.float32)
        pixels = (pixel_near + t*(pixel_far - pixel_near)).T
        # Transform pixels in homogenous coordinates
        pixels = np.hstack((pixels, np.ones((self.n_points, 1))))

        # For each of these pixels compute the corresponding ray
        points = []
        for p in pixels:
            # Get the new ray passing through the random pixel
            n_origin, n_destination = images[-1].ray(p.reshape(-1, 1))
            # Compute the intersection of each ray with the viewing ray and get
            # the 3D points that correspond to the depth planes
            n_direction = n_destination - n_origin

            # The arguments SHOULD NOT be in homogenous coordinates
            point = ray_ray_intersection(
                origin[:-1],
                direction[:-1],
                n_origin[:-1],
                n_direction[:-1]
            )
            # Transform the point in homogenous cordinates before appending it
            # in the list
            points.append(np.hstack((point[0], [1.])))

        points = np.array(points, dtype=np.float32)
        # Make sure that points have shape(n, 4)
        assert points.shape == (self.n_points, 4)
        return points


class SamplingInVoxelSpaceScheme(SamplingScheme):
    def __init__(self, generation_params):
        super(SamplingInVoxelSpaceScheme, self).__init__(generation_params)
        self._grid_shape = generation_params.grid_shape
        self.n_points = generation_params.max_number_of_marched_voxels

    def sample_points_across_ray(self, scene, i, y, x):
        bbox = scene.bbox
        origin, destination = self._get_ray_from_pixel(scene, i, y, x)

        # Get the point of entry and exit to the bbox as `origin + t_(near|far)
        # * (destination - origin)`
        t_near, t_far = ray_aabbox_intersection(
            origin,
            destination,
            bbox[0, :3],
            bbox[0, 3:]
        )
        if t_near is None or t_far is None:
            return None

        # Compute the start and end point of the 3D line segment
        direction = destination - origin
        p_near = (origin + t_near * direction)
        p_far = (origin + t_far * direction)

        ray_voxel_indices = np.zeros((self.n_points, 3), dtype=np.int32)
        Nr = voxel_traversal(
            bbox.reshape(-1,),
            np.array(self._grid_shape, dtype=np.int32),
            ray_voxel_indices,
            p_near[:-1].ravel(),
            p_far[:-1].ravel()
        )
        idxs = ray_voxel_indices[:Nr]
        # Transform the indices in the voxel_space to actual points
        points = scene.voxel_grid(
            self._grid_shape
        )[:, idxs[:, 0], idxs[:, 1], idxs[:, 2]].T
        points = np.hstack([points, np.ones((points.shape[0], 1))])

        # Make sure that points have shape(n, 4)
        assert points.shape == (Nr, 4)
        return points.astype(np.float32)


class TFSamplingInBboxScheme(SamplingInBboxScheme):
    def __init__(self, generation_params):
        super(TFSamplingInBboxScheme, self).__init__(generation_params)
        self.generation_params = generation_params
        self._sp = None

    def _build_sampler(self, scene):
        if self._sp is None:
            self._sp = build_sample_points_in_bbox(
                scene.image_shape[0],
                scene.image_shape[1],
                self.generation_params
            )
        return self._sp

    def sample_points_across_rays(self, scene, i):
        # Based on the i index compute the multi-view Image objects
        images = scene.get_image_with_neighbors(i)

        # Prepare the inputs
        inputs = []
        # Add the camera center of the reference view
        inputs.append(images[0].camera.center)
        # Add the pseudo inverse projection matrix for the reference image
        inputs.append(images[0].camera.P_pinv)
        # Add the bounding box that encloses the scene
        inputs.append(scene.bbox.reshape(6, 1))

        points = self._build_sampler(scene)(inputs)

        return points


class TFSamplingInRangeScheme(SamplingInRangeScheme):
    def __init__(self, generation_params):
        super(TFSamplingInRangeScheme, self).__init__(generation_params)
        self.generation_params = generation_params
        self._sp = None

    def _build_sampler(self, scene):
        if self._sp is None:
            self._sp = build_sample_points_in_range(
                scene.image_shape[0],
                scene.image_shape[1],
                self.generation_params
            )
        return self._sp

    def sample_points_across_rays(self, scene, i):
        # Based on the i index compute the multi-view Image objects
        images = scene.get_image_with_neighbors(i)

        # Prepare the inputs
        inputs = []
        # Add the camera center of the reference view
        inputs.append(images[0].camera.center)
        # Add the pseudo inverse projection matrix for the reference image
        inputs.append(images[0].camera.P_pinv)
        # Add the bounding box that encloses the scene
        inputs.append(scene.bbox.reshape(6, 1))

        points = self._build_sampler(scene)(inputs)

        return points


class DummySamplingScheme(object):
    def __init__(self, generation_params):
        self.sampling_type = generation_params.sampling_type


def get_sampling_scheme(name):
    return {
        "sample_in_bbox": SamplingInBboxScheme,
        "sample_in_disparity": SamplingInDisparityScheme,
        "sample_in_range": SamplingInRangeScheme,
        "tf_sample_in_bbox": TFSamplingInBboxScheme,
        "tf_sample_in_range": TFSamplingInRangeScheme,
        "full_tf_sample_in_bbox": DummySamplingScheme,
        "full_tf_sample_in_range": DummySamplingScheme
    }[name]
