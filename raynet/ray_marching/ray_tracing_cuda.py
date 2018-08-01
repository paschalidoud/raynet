from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from ..tf_implementations.sampling_schemes import build_rays_bbox_intersections
from ..cuda_implementations.utils import all_arrays_to_gpu


def batch_voxel_traversal(M, bbox, grid_shape):
    cu_file_path = "../raynet/cuda_implementations/ray_tracing.cu"
    with open(cu_file_path, "r") as f:
        cu_file = f.read()
    tpl = Template(cu_file)

    mod = SourceModule(tpl.substitute(
        max_voxels=M,
        bbox_min_x=bbox[0],
        bbox_min_y=bbox[1],
        bbox_min_z=bbox[2],
        bbox_max_x=bbox[3],
        bbox_max_y=bbox[4],
        bbox_max_z=bbox[5],
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2]
    ))

    cuda_vtr = mod.get_function("batch_voxel_traversal")
    cuda_vtr.prepare("i" + "P"*4)

    @all_arrays_to_gpu
    def vtr(
        points_start,
        points_end,
        ray_voxel_indices,
        ray_voxel_count,
        threads=1024
    ):
        # Assert everything in the right size, shape and dtype
        assert ray_voxel_indices[1] == M
        assert np.float32 == points_start.dtype
        assert np.float32 == points_end.dtype
        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(ray_voxel_count)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_vtr.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            n_rays,
            points_start.gpudata,
            points_end.gpudata,
            ray_voxel_indices.gpudata,
            ray_voxel_count.gpudata
        )

    return vtr


def voxel_traversal(
    bbox,
    grid_shape,
    ray_voxel_indices,
    ray_start,
    ray_end
):
    N, M = ray_voxel_indices.shape

    # Move everything to the gpu
    ray_voxel_count = to_gpu(np.zeros((1,), dtype=np.int32))
    ray_voxel_indices_out = to_gpu(ray_voxel_indices)

    vtr = batch_voxel_traversal(N, bbox, grid_shape)
    vtr(
        to_gpu(ray_start),
        to_gpu(ray_end),
        ray_voxel_indices_out,
        ray_voxel_count,
        threads=1
    )
    ray_voxel_indices[:, :] = ray_voxel_indices_out.get()

    return ray_voxel_count.get()[0]


def perform_ray_marching(
    scene,
    img_idx,
    M,
    rays_idxs,
    grid_shape,
    batch_size=40000
):
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

    # Number of rays used for the ray marching
    N = rays_idxs.shape[0]

    # Allocate memory for the results (final and intermediate)
    ray_voxel_indices = np.zeros((N, M, 3), dtype=np.int32)
    ray_voxel_count = to_gpu(np.zeros((N,), dtype=np.int32))
    ray_voxel_indices_gpu = to_gpu(
        np.zeros((batch_size, M, 3), dtype=np.int32)
    )
    points_start_gpu = to_gpu(points_start[:-1, rays_idxs].T)
    points_end_gpu = to_gpu(points_end[:-1, rays_idxs].T)

    vtr = batch_voxel_traversal(M, scene.bbox.ravel(), grid_shape)
    # Start iterationg over the batch of rays
    for r in range(0, len(rays_idxs), batch_size):
        ray_voxel_indices_gpu.fill(0)
        vtr(
            points_start_gpu[r:r+batch_size],
            points_end_gpu[r:r+batch_size],
            ray_voxel_indices_gpu,
            ray_voxel_count[r:r+batch_size]
        )

        # Check if the number of intersected voxels is larger than M
        # if np.any(ray_voxel_indices.get() > M):
        #    raise ValueError("Nr=%d > M=%d" % (Nr, M))
        ray_voxel_indices[r:r+batch_size, :, :] = ray_voxel_indices_gpu.get()

    return ray_voxel_indices, ray_voxel_count.get()
