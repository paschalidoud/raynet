from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from ..cuda_implementations.utils import all_arrays_to_gpu


def batch_depth_to_voxels_mapping(M, D, grid_shape):
    cu_file_path = "../raynet/cuda_implementations/planes_voxels_mapping.cu"
    with open(cu_file_path, "r") as f:
        cu_file = f.read()
    tpl = Template(cu_file)

    mod = SourceModule(tpl.substitute(
        max_voxels=M,
        depth_planes=D,
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2]
    ))
    cuda_pvm = mod.get_function("batch_planes_voxels_mapping")
    cuda_pvm.prepare("i" + "P"*7)

    @all_arrays_to_gpu
    def pvm(
        voxel_grid,
        ray_voxel_indices,
        ray_voxel_count,
        ray_start,
        ray_end,
        S,
        S_new,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S.shape[1] == D
        assert S_new.shape[1] == M
        assert len(ray_voxel_count.shape) == 1
        assert np.float32 == S.dtype
        assert np.float32 == S_new.dtype
        assert np.int32 == ray_voxel_count.dtype
        assert np.float32 == ray_start.dtype
        assert np.float32 == ray_end.dtype

        # Determine the grid and block arguments
        n_rays = len(S)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_pvm.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            voxel_grid.gpudata,
            ray_voxel_indices.gpudata,
            ray_voxel_count.gpudata,
            ray_start.gpudata,
            ray_end.gpudata,
            S.gpudata,
            S_new.gpudata
        )

        return S_new

    return pvm


def depth_to_voxels(
    ray_voxel_count,
    ray_voxel_indices,
    rays_idxs,
    voxel_grid,
    points,
    S,
    S_new,
    batch_size=20000
):
    """Compute the depth probability of each voxel based on S the probability
    distribution on points.

    Arguments
    ---------
        ray_voxel_count: array(shape=(N,), int), The number of voxels
                         intersected by each ray
        ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                           voxel grid per ray
        rays_idxs: array(shape=(N,), int), The indices of the valid rays
        voxel_grid: array(shape=(3, D1, D2, D3)), The coordinates of the
                    centers of all voxels in the voxel grid of size
                    (D1, D2, D3)
        points: array(shape=(4, N, D), float32), D points sampled on each of
                the N rays
        S: array(shape=(N, D), float32), A depth probability distribution for
           each of the N rays
    """
    # Extract the numbers N, M, D
    N, M, _ = ray_voxel_indices.shape
    _, _, D = points.shape

    # Fill the output array to 0
    S_new.fill(0)
    # Move to GPU to save some time frome copying
    points_start_gpu = to_gpu(points[:-1, rays_idxs, 0].T)
    points_end_gpu = to_gpu(points[:-1, rays_idxs, -1].T)
    ray_voxel_count = to_gpu(ray_voxel_count[rays_idxs])

    pvm = batch_depth_to_voxels_mapping(M, D, np.array(voxel_grid.shape[1:]))
    voxel_grid = voxel_grid.transpose(1, 2, 3, 0).ravel()
    # Start iterationg over the batch of rays
    for i in range(0, len(rays_idxs), batch_size):
        s = pvm(
            voxel_grid,
            ray_voxel_indices[rays_idxs[i:i+batch_size]],
            ray_voxel_count[i:i+batch_size],
            points_start_gpu[i:i+batch_size],
            points_end_gpu[i:i+batch_size],
            S[rays_idxs[i:i+batch_size]],
            S_new[rays_idxs[i:i+batch_size]]
        )
        S_new[rays_idxs[i:i+batch_size]] = s.get()

    return S_new
