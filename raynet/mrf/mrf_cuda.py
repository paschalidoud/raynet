from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu

from ..cuda_implementations.utils import all_arrays_to_gpu,\
    parse_cu_files_to_string


def batch_ray_belief_propagation(M, grid_shape):
    cu_source_code = parse_cu_files_to_string([
        "../raynet/cuda_implementations/mrf_bp.cu",
    ])
    tpl = Template(cu_source_code)

    mod = SourceModule(tpl.substitute(
        max_voxels=M,
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2]
    ))

    cuda_bp = mod.get_function("batch_belief_propagation")
    cuda_bp.prepare("i" + "P"*7)

    mod_de = SourceModule(tpl.substitute(
        max_voxels=M,
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2]
    ))
    cuda_de = mod_de.get_function("batch_depth_estimation")
    cuda_de.prepare("i" + "P"*6)

    @all_arrays_to_gpu
    def bp(
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon,
        ray_to_occupancy_accumulated_out_pon,
        threads=1024
    ):
        # Assert everything is the right size, shape and dtype
        assert S.shape[1] == M
        assert ray_voxel_indices.shape[1:] == (M, 3)
        assert len(ray_voxel_count.shape) == 1
        assert len(ray_voxel_count) == len(S) == len(ray_voxel_indices)
        assert len(ray_voxel_count) == len(ray_to_occupancy_messages_pon)
        assert S.shape[1] == ray_to_occupancy_messages_pon.shape[1]
        assert ray_to_occupancy_accumulated_pon.shape == tuple(grid_shape)
        assert ray_to_occupancy_accumulated_out_pon.shape == tuple(grid_shape)
        assert np.float32 == S.dtype
        assert np.float32 == ray_to_occupancy_messages_pon.dtype
        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(S)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_bp.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            S.gpudata,
            ray_voxel_indices.gpudata,
            ray_voxel_count.gpudata,
            ray_to_occupancy_accumulated_pon.gpudata,
            ray_to_occupancy_messages_pon.gpudata,
            ray_to_occupancy_accumulated_out_pon.gpudata,
            ray_to_occupancy_messages_pon.gpudata
        )

        return ray_to_occupancy_accumulated_out_pon, \
            ray_to_occupancy_messages_pon

    @all_arrays_to_gpu
    def de(
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon,
        S_new,
        threads=1024
    ):
        # Assert everything is the right size, shape and dtype
        assert S.shape[1] == M
        assert S_new.shape[1] == M
        assert ray_voxel_indices.shape[1:] == (M, 3)
        assert len(ray_voxel_count.shape) == 1
        assert len(ray_voxel_count) == len(S) == len(ray_voxel_indices)
        assert len(ray_voxel_count) == len(ray_to_occupancy_messages_pon)
        assert S.shape[1] == ray_to_occupancy_messages_pon.shape[1]
        assert ray_to_occupancy_accumulated_pon.shape == tuple(grid_shape)
        assert np.float32 == S.dtype
        assert np.float32 == S_new.dtype
        assert np.float32 == ray_to_occupancy_messages_pon.dtype
        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(S)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_de.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            S.gpudata,
            ray_voxel_indices.gpudata,
            ray_voxel_count.gpudata,
            ray_to_occupancy_accumulated_pon.gpudata,
            ray_to_occupancy_messages_pon.gpudata,
            S_new.gpudata
        )

        return S_new

    return bp, de


def belief_propagation(
    S,
    ray_voxel_indices,
    ray_voxel_count,
    ray_to_occupancy_messages_pon,
    grid_shape,
    gamma=0.05,
    bp_iterations=3,
    batch_size=50000
):
    """Run the belief propagation for a set of rays

    Arguments:
    ---------
        S: array(shape=(N, M), float32), A depth probability distribution for
           each of the N rays
        ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                           voxel grid per ray
        ray_voxel_count: array(shape=(N,1), int) The number of voxels
                         intersected by each ray
        ray_to_occupancy_messages_pon: array(shape=(N, M) float32), Holds the
                                       ray_to_occupancy messages between bp
                                       iterations
        grid_shape: array(shape=(3,), int), The number of voxels for each axis
        gamma: float32, Prior probabilitiy that the ith voxel is occupied
        bp_iterations: int, Number of belief-propagation iterations
        batch_size: int, Number of rays per batch size
    """
    # Extract the number of rays
    N, M = S.shape

    # Initialize the ray to occupancy messages to uniform
    ray_to_occupancy_messages_pon.fill(0)

    # Initialize the ray_to_occupancy accumulators as GPU arrays
    ray_to_occupancy_accumulated_pon = np.ones(
        tuple(grid_shape),
        dtype=np.float32
    ) * (np.log(gamma) - np.log(1 - gamma))
    ray_to_occupancy_accumulated_pon = \
        to_gpu(ray_to_occupancy_accumulated_pon)

    ray_to_occupancy_accumulated_out_pon = np.ones(
        tuple(grid_shape),
        dtype=np.float32
    ) * (np.log(gamma) - np.log(1 - gamma))
    ray_to_occupancy_accumulated_out_pon = \
        to_gpu(ray_to_occupancy_accumulated_out_pon)

    bp, _ = batch_ray_belief_propagation(M, grid_shape)

    # Iterate over the rays multiple times
    for it in xrange(bp_iterations):
        print "Iteration %d " % (it,)
        for i in range(0, N, batch_size):
            _, msgs = bp(
                S[i:i+batch_size],
                ray_voxel_indices[i:i+batch_size],
                ray_voxel_count[i:i+batch_size],
                ray_to_occupancy_accumulated_pon,
                ray_to_occupancy_messages_pon[i:i+batch_size],
                ray_to_occupancy_accumulated_out_pon
            )
            ray_to_occupancy_messages_pon[i:i+batch_size] = msgs.get()

        # Swap the accumulators for the next bp iteration
        ray_to_occupancy_accumulated_out_pon, ray_to_occupancy_accumulated_pon = \
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_accumulated_out_pon
        ray_to_occupancy_accumulated_out_pon.fill(np.log(gamma) - np.log(1 - gamma))

    return ray_to_occupancy_accumulated_pon.get(), ray_to_occupancy_messages_pon


def compute_depth_distribution(
    S,
    ray_voxel_indices,
    ray_voxel_count,
    ray_to_occupancy_messages_pon,
    ray_to_occupancy_accumulated_pon,
    S_new,
    grid_shape,
    batch_size=50000
):
    """Perform the depth estimation

    Arguments:
    ---------
        S: array(shape=(N, M), float32), A depth probability distribution for
           each of the N rays
        ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                           voxel grid per ray
        ray_voxel_count: array(shape=(N,1), int) The number of voxels
                         intersected by each ray
        ray_to_occupancy_messages_pon: array(shape=(N, M) float32), Holds the
                                       ray_to_occupancy messages between bp
                                       iterations
        ray_to_occupancy_accumulated_pon: array(shape=(D1, D2, D3) float32)
                                          The accumulation of the positive
                                          over negative ray-to-occupancy msgs
        S_new: array(shape=(N, M), float32), A refined depth probability
               distribution for each of the N rays
        grid_shape: array(shape=(3,), int), The number of voxels for each axis
        batch_size: int, Number of rays per batch size
    """
    # Extract the number of rays
    N, M = S.shape

    # Fill the output array to 0
    S_new.fill(0)

    _, de = batch_ray_belief_propagation(M, grid_shape)

    # Iterate over the rays multiple times
    for i in range(0, N, batch_size):
        s = de(
            S[i:i+batch_size],
            ray_voxel_indices[i:i+batch_size],
            ray_voxel_count[i:i+batch_size],
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_messages_pon[i:i+batch_size],
            S_new[i:i+batch_size]
        )
        S_new[i:i+batch_size] = s.get()

    return S_new

