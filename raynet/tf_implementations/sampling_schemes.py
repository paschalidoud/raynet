from itertools import product

from keras import backend as K
import numpy as np


def rays_bbox_intersection(bbox, origin, directions):
    """Compute the intersections of the rays with an axis-aligned bbox.

    Arguments
    ---------
    bbox: tensor(shape=(6, 1), float32), The bounding box to intersect with
    origin: tensor(shape=(4, 1), float32), The origin of the rays
    directions: tensor(shape=(4, N), float32), The direction vectors defining
                the rays
    """
    # Variables that will hold the nearest and furthest intersection (namely
    # the entrance point and exit point of the box)
    N = K.int_shape(directions)[1]
    t_near = K.variable(
        np.ones(N)*float("-inf"),
        dtype="float32",
        name="t_near"
    )
    t_far = K.variable(np.ones(N)*float("inf"), dtype="float32", name="t_far")

    for i in range(3):
        # Intersection at axis i
        t1 = (bbox[i] - origin[i]) / directions[i]
        t2 = (bbox[i + 3] - origin[i]) / directions[i]

        t_near = K.maximum(K.minimum(t1, t2), t_near)
        t_far = K.minimum(K.maximum(t1, t2), t_far)

    # Swap far and near in case of negative values
    near_mask = K.cast(K.abs(t_near) < K.abs(t_far), K.floatx())
    t_near_actual = t_near * near_mask + t_far * (1 - near_mask)
    t_far_actual = (1 - near_mask) * t_near + near_mask * t_far

    return t_near_actual, t_far_actual


def sample_points_in_bbox(bbox, origin, directions, n_points):
    """Sample uniformly points inside the bounding box that encloses the scene.

    Arguments
    ---------
    bbox: tensor(shape=(6, 1), float32), The bounding box that encloses the
          scene
    origin: tensor(shape=(4, 1), float32), The origin of the rays
    directions: tensor(shape=(4, N), float32), The direction vectors defining
                the rays
    n_points: int, The number of points to be sampled
    """
    # How many rays do we have?
    N = K.shape(directions)[1]

    # Calculate the intersections with the bounding box
    t_near, t_far = rays_bbox_intersection(bbox, origin, directions)

    # Sample points uniformly on the ray in the bbox
    points = K.map_fn(
        lambda i: origin + directions[:, i:i+1] * K.tf.linspace(t_near[i], t_far[i], n_points),
        K.tf.range(N),
        dtype="float32"
    )

    return K.permute_dimensions(
        K.reshape(points, (N, 4, n_points)),
        (1, 0, 2)
    )


def sample_points_in_range(min_range, max_range, origin, directions, n_points):
    """Sample uniformly depth planes in a depth range set to [min_range,
       max_range]

    Arguments
    ---------
    min_range: int, The minimum depth range
    max_range: int, The maximum depth range
    origin: tensor(shape=(4, 1), float32), The origin of the rays
    directions: tensor(shape=(4, N), float32), The direction vectors defining
                the rays
    n_points: int, The number of points to be sampled
    """
    # How many rays do we have?
    N = K.shape(directions)[1]

    directions /= K.sqrt(K.sum(directions**2, axis=0))

    # Sample points uniformly on the ray in the bbox
    points = K.map_fn(
        lambda i: origin + directions[:, i:i+1] * K.tf.linspace(min_range, max_range, n_points),
        K.tf.range(N),
        dtype="float32"
    )

    return K.permute_dimensions(
        K.reshape(points, (N, 4, n_points)),
        (1, 0, 2)
    )


def compute_rays_from_pixels(pixels, P_inv, camera_center):
    """Compute the rays and the directions starting from camera_center with an
    inverse projection matrix P_inv for an image ith H,W pixels.

    Arguments
    ---------
        pixels: tensor(shape=(3, N), float32), The pixels
        P_inv: tensor(shape=(4, 3), float32), The inverse projection matrix for
               a camera
        camera_center: tensor(shape=(3, 1), float32), The camera position (all
                       rays start from there)
    """
    # Compute the rays that start from camera center
    rays = K.dot(P_inv, pixels)
    rays /= rays[-1:, :]  # normalize the homogenous coordinates

    # Compute the directions so that the ray is camera_center + t * direction
    # NOTE: The homogenous coordinate of the directions tensor is 0!
    directions = rays - camera_center

    return rays, directions


def compute_rays(H, W, P_inv, camera_center):
    """Compute the rays and the directions starting from camera_center with an
    inverse projection matrix P_inv for an image ith H,W pixels.

    Arguments
    ---------
        H: int, The height of the image in pixels
        W: int, The width of the image in pixels
        P_inv: tensor(shape=(4, 3), float32), The inverse projection matrix for
               a camera
        camera_center: tensor(shape=(3, 1), float32), The camera position (all
                       rays start from there)
    """
    # Create the pixels array
    pixels = K.variable(np.array([
        [u, v, 1.] for u, v in product(range(W), range(H))
    ]).T, dtype="float32", name="pixels")

    rays, directions = compute_rays_from_pixels(pixels, P_inv, camera_center)

    return pixels, rays, directions


def build_rays_bbox_intersections(H, W):
    camera_center = K.placeholder(
        shape=(4, 1),
        dtype="float32",
        name="camera_center"
    )
    P_inv = K.placeholder(
        shape=(4, 3),
        dtype="float32",
        name="inv_proj_matrix"
    )
    bbox = K.placeholder(shape=(6, 1), dtype="float32", name="bbox")

    pixels, rays, directions = compute_rays(H, W, P_inv, camera_center)
    t_near, t_far = rays_bbox_intersection(bbox, camera_center, directions)
    points_near = camera_center + t_near*directions
    points_far = camera_center + t_far*directions

    return K.function(
        [camera_center, P_inv, bbox],
        [points_near, points_far]
    )


def build_sample_points_in_bbox(H, W, generation_params):
    camera_center = K.placeholder(
        shape=(4, 1),
        dtype="float32",
        name="camera_center"
    )
    P_inv = K.placeholder(
        shape=(4, 3),
        dtype="float32",
        name="inv_proj_matrix"
    )
    bbox = K.placeholder(shape=(6, 1), dtype="float32", name="bbox")

    pixels, rays, directions = compute_rays(H, W, P_inv, camera_center)

    n_points = generation_params.depth_planes
    points = sample_points_in_bbox(bbox, camera_center, directions, n_points)

    return K.function([camera_center, P_inv, bbox], [points])


def build_sample_points_in_range(H, W, generation_params):
    camera_center = K.placeholder(
        shape=(4, 1),
        dtype="float32",
        name="camera_center"
    )
    P_inv = K.placeholder(
        shape=(4, 3),
        dtype="float32",
        name="inv_proj_matrix"
    )
    bbox = K.placeholder(shape=(6, 1), dtype="float32", name="bbox")

    pixels, rays, directions = compute_rays(H, W, P_inv, camera_center)
    n_points = generation_params.depth_planes
    points = sample_points_in_range(
        generation_params.depth_range[0],
        generation_params.depth_range[1],
        camera_center,
        directions,
        generation_params.depth_planes
    )

    return K.function([camera_center, P_inv, bbox], [points])
