import numpy as np

from .ray_tracing import voxel_traversal
from ..tf_implementations.sampling_schemes import build_rays_bbox_intersections
from .ray_tracing_cuda import perform_ray_marching as perform_ray_marching_cuda


def get_ray_voxel_indices(
    bbox,
    grid_shape,
    points_start,
    points_end,
    rays_idxs,
    ray_voxel_indices,
    ray_voxel_count
):
    """Perform the ray marching given a scene and a set of entry and out points in
    the voxel grid
    Atributes:
    ---------
        grid_shape: np.array((D1, D2, D3))
                    The same of the grid
    """
    # Make sure that everything has the right shape
    assert ray_voxel_indices.shape[0] > ray_voxel_indices.shape[1]
    assert ray_voxel_indices.shape[2] == 3
    assert ray_voxel_indices.shape[0] == ray_voxel_count.shape[0]

    # Export some important dimensions
    N, M, _ = ray_voxel_indices.shape

    # Start iterationg over the batch of rays
    for r, ray_idx in enumerate(rays_idxs):
        Nr = voxel_traversal(
            bbox.ravel(),
            grid_shape,
            ray_voxel_indices[r],
            points_start[:, ray_idx][:-1],
            points_end[:, ray_idx][:-1]
        )
        if Nr >= M:
            raise ValueError("Nr=%d > M=%d" % (Nr, M))
        ray_voxel_count[r] = Nr

    return ray_voxel_indices, ray_voxel_count


def perform_ray_marching(scene, img_idx, M, rays_idxs, grid_shape):
    H, W = scene.image_shape
    # Get the reference camera
    ref_camera = scene.get_image(img_idx).camera

    # Compute the start and end points for the current image
    g = build_rays_bbox_intersections(H, W)
    points_start, points_end = g(
        [ref_camera.center, ref_camera.P_pinv, scene.bbox.reshape(-1, 1)]
    )
    # Make sure that the input points have the right size
    assert points_start.shape[1] == H*W
    assert points_end.shape[1] == H*W
    assert points_start.shape[0] == 4
    assert points_start.shape[0] == points_end.shape[0]

    # Export the discretization steps in the voxel grid
    N = rays_idxs.shape[0]  # Number of rays used for the ray marching

    ray_voxel_indices = np.zeros((N, M, 3), dtype=np.int32)
    ray_voxel_count = np.zeros((N,), dtype=np.int32)

    # Do the ray marching for the given rays
    ray_voxel_indices, ray_voxel_count = get_ray_voxel_indices(
        scene.bbox,
        grid_shape,
        points_start,
        points_end,
        rays_idxs,
        ray_voxel_indices,
        ray_voxel_count
    )

    return ray_voxel_indices, ray_voxel_count


def get_voxel_traversal_backend(name):
    if name == "cython":
        return perform_ray_marching
    elif name == "cuda":
        return perform_ray_marching_cuda
    else:
        raise NotImplementedError()
