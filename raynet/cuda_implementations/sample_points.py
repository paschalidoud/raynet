from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from ..cuda_implementations.utils import all_arrays_to_gpu, \
    parse_cu_files_to_string


def batch_sample_points(D, H, W, bbox, sampling_scheme):
    file_paths = ["sampling_schemes.cu"]
    cu_source_code = parse_cu_files_to_string(file_paths)

    tpl = Template(cu_source_code)
    mod = SourceModule(tpl.substitute(
        depth_planes=D,
        width=W,
        height=H,
        bbox_min_x=bbox[0],
        bbox_min_y=bbox[1],
        bbox_min_z=bbox[2],
        bbox_max_x=bbox[3],
        bbox_max_y=bbox[4],
        bbox_max_z=bbox[5],
        sampling_scheme=sampling_scheme
    ))
    cuda_sp = mod.get_function("batch_sample_points_in_bbox")
    cuda_sp.prepare("i" + "P"*4)

    @all_arrays_to_gpu
    def sp(
        ray_idxs,
        P_inv,
        camera_center,
        points,
        threads=2048
    ):
        # Determine the grid and block arguments
        n_rays = len(ray_idxs)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_sp.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            ray_idxs.gpudata,
            P_inv.gpudata,
            camera_center.gpudata,
            points.gpudata
        )

    return sp


def sample_points(
    ray_idxs,
    P_inv,
    camera_center,
    points,
    H,
    W,
    bbox,
    sampling_scheme="sample_in_bbox",
    batch_size=80000
):
    # Extract the number of depth planes
    _, D, _ = points.shape
    #  Make sure that points has the right size
    assert points.shape == (H*W, D, 4)

    # Move to GPU
    ray_idxs = to_gpu(ray_idxs.astype(np.int32))
    P_inv_gpu = to_gpu(P_inv.ravel())
    camera_center_gpu = to_gpu(camera_center)
    points_gpu = to_gpu(np.zeros((batch_size, D, 4), dtype=np.float32))

    sp = batch_sample_points(D, H, W, bbox, sampling_scheme)
    # Start iterationg over the batch of rays
    for i in range(0, len(ray_idxs), batch_size):
        points_gpu.fill(0)
        sp(
            ray_idxs[i:i+batch_size],
            P_inv_gpu,
            camera_center_gpu,
            points_gpu
        )
        points[i:i+batch_size, :, :] = points_gpu.get()

    return points


def compute_depth_from_distribution(
    ray_idxs,
    P_inv,
    camera_center,
    H,
    W,
    bbox,
    S,
    depth_map,
    sampling_scheme="sample_in_bbox",
    batch_size=80000
):
    # Extract the number of depth planes
    _, D, = S.shape

    # Move to GPU
    ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
    P_inv_gpu = to_gpu(P_inv.ravel())
    camera_center_gpu = to_gpu(camera_center)
    points_gpu = to_gpu(np.zeros((batch_size, D, 4), dtype=np.float32))

    sp = batch_sample_points(D, H, W, bbox, sampling_scheme)
    # Start iterationg over the batch of rays
    for i in range(0, len(ray_idxs), batch_size):
        points_gpu.fill(0)
        sp(
            ray_idxs_gpu[i:i+batch_size],
            P_inv_gpu,
            camera_center_gpu,
            points_gpu
        )
        idxs = ray_idxs[i:i+batch_size]
        pts = points_gpu.get()[:len(idxs)].transpose(2, 0, 1)
        pts = pts[:-1, np.arange(len(idxs)), S[idxs].argmax(axis=-1)]
        depth_map[idxs] = np.sqrt(
            np.sum((camera_center[:-1] - pts)**2, axis=0)
        )

    return depth_map
