from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from .utils import all_arrays_to_gpu, parse_cu_files_to_string


def perform_multi_view_cnn_forward_pass(
    D,
    N,
    F,
    H,
    W,
    padding,
    bbox,
    sampling_scheme
):
    """Compile the CUDA kernel that given the features and the camera matrices
    estimates the similarities between the features and convert them into a
    per-pixel depth distribution

    Arguments:
    ----------
        D: int, depth planes (discretization steps)
        N: int, number of views
        F: int, feature size (from the Multi-View CNN)
        H: int, image height
        W: int, image width,
        padding: int, the number of zero-padded pixels around the image to
                 estimate the features from the Multi-View CNN
        bbox: np.array((6,), dtype=np.float32), the coordinates of the bbox
              that enclose the scene
        sampling_scheme: string, specification of the sampling scheme
    """
    # Set the paths to the files that will be used to construct the cuda kernel
    file_paths = [
        "feature_similarities.cu",
        "sampling_schemes.cu"
    ]
    cu_source_code = parse_cu_files_to_string(file_paths)

    tpl = Template(cu_source_code + """
        __global__ void batch_multi_view_cnn_forward_pass(
            int n_rays,
            int * ray_idxs,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * S
        ) {
            // Compute the thread
            int r = threadIdx.x + blockDim.x * blockIdx.x;
            if (r >= n_rays)
                return;

            // Estimate the ray_start and ray_end for the current pixel
            float ray_start[3], ray_end[3];
            $sampling_scheme(
                ray_idxs[r],
                P_inv,
                camera_center,
                ray_start,
                ray_end
            );

            // Compute the similarities between features
            compute_similarities_per_ray(
                features,
                P,
                ray_start,
                ray_end,
                S + r*$depth_planes
            );

        }
    """)

    mod = SourceModule(tpl.substitute(
        depth_planes=D,
        n_views=N,
        padding=padding,
        features_dimensions=F,
        width=W,
        height=H,
        bbox_min_x=bbox[0],
        bbox_min_y=bbox[1],
        bbox_min_z=bbox[2],
        bbox_max_x=bbox[3],
        bbox_max_y=bbox[4],
        bbox_max_z=bbox[5],
        sampling_scheme=sampling_scheme,
    ))
    cuda_mvcnnfp = mod.get_function("batch_multi_view_cnn_forward_pass")
    cuda_mvcnnfp.prepare("i" + "P"*6)

    @all_arrays_to_gpu
    def mvcnnfp(
        ray_idxs,
        features,
        P,
        P_inv,
        camera_center,
        S,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S.shape[1] == D
        assert np.float32 == S.dtype

        # Determine the grid and block arguments
        n_rays = len(S)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_mvcnnfp.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            ray_idxs.gpudata,
            features.gpudata,
            P.gpudata,
            P_inv.gpudata,
            camera_center.gpudata,
            S.gpudata
        )

    return mvcnnfp


def perform_multi_view_cnn_forward_pass_with_depth_estimation(
    D,
    N,
    F,
    H,
    W,
    padding,
    bbox,
    sampling_scheme
):
    """Compile the CUDA kernel that given the features and the camera matrices
    estimates the similarities between the features and converts the per-pixel
    depth estimation into a depth map

    Arguments:
    ----------
        D: int, depth planes (discretization steps)
        N: int, number of views
        F: int, feature size (from the Multi-View CNN)
        H: int, image height
        W: int, image width,
        padding: int, the number of zero-padded pixels around the image to
                 estimate the features from the Multi-View CNN
        bbox: np.array((6,), dtype=np.float32), the coordinates of the bbox
              that enclose the scene
        sampling_scheme: string, specification of the sampling scheme
    """
    # Set the paths to the files that will be used to construct the cuda kernel
    file_paths = [
        "feature_similarities.cu",
        "sampling_schemes.cu"
    ]
    cu_source_code = parse_cu_files_to_string(file_paths)

    tpl = Template(cu_source_code + """
        __global__ void batch_multi_view_cnn_forward_pass_with_depth(
            int n_rays,
            int * ray_idxs,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * S,
            float * points,
            float * depth_map
        ) {
            // Compute the thread
            int r = threadIdx.x + blockDim.x * blockIdx.x;
            if (r >= n_rays)
                return;

            // Estimate the ray_start and ray_end for the current pixel
            float ray_start[3], ray_end[3];
            $sampling_scheme(
                ray_idxs[r],
                P_inv,
                camera_center,
                ray_start,
                ray_end
            );

            // Compute the similarities between features
            compute_similarities_per_ray(
                features,
                P,
                ray_start,
                ray_end,
                S + r*$depth_planes
            );

            float * Sr = S + r*$depth_planes;
            float max = -INFINITY;
            int max_idx = 0;

            int offset = r * $depth_planes * 4;
            // Get the rest of the uniformly sampled points and at the same
            // time also find the argmax of S
            for (int k=0; k<$depth_planes; k++) {
                points[offset + 4*k] = ray_start[0] + k*(ray_end[0] - ray_start[0])/($depth_planes - 1);
                points[offset + 4*k + 1] = ray_start[1] + k*(ray_end[1] - ray_start[1])/($depth_planes - 1);
                points[offset + 4*k + 2] = ray_start[2] + k*(ray_end[2] - ray_start[2])/($depth_planes - 1);
                points[offset + 4*k + 3] = 1.0;

                // Find the index with the highest value/probability in S in
                // order to compute the most probable 3D point afterwards
                if (Sr[k] > max) {
                    max_idx = k;
                    max = Sr[k];
                }
            }

            // Get the distance from the camera center
            float sum = 0.0;
            for (int i=0; i<3; i++) {
                sum += pow(points[offset + 4*max_idx + i] - camera_center[i], 2);
            }
            depth_map[r] = sqrt(sum);
        }
    """)

    mod = SourceModule(tpl.substitute(
        depth_planes=D,
        n_views=N,
        padding=padding,
        features_dimensions=F,
        width=W,
        height=H,
        bbox_min_x=bbox[0],
        bbox_min_y=bbox[1],
        bbox_min_z=bbox[2],
        bbox_max_x=bbox[3],
        bbox_max_y=bbox[4],
        bbox_max_z=bbox[5],
        sampling_scheme=sampling_scheme,
    ))
    cuda_mvcnnfp = mod.get_function("batch_multi_view_cnn_forward_pass_with_depth")
    cuda_mvcnnfp.prepare("i" + "P"*8)

    @all_arrays_to_gpu
    def mvcnnfp(
        ray_idxs,
        features,
        P,
        P_inv,
        camera_center,
        S,
        points,
        depth_map,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S.shape[1] == D
        assert np.float32 == S.dtype

        # Determine the grid and block arguments
        n_rays = len(S)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_mvcnnfp.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            ray_idxs.gpudata,
            features.gpudata,
            P.gpudata,
            P_inv.gpudata,
            camera_center.gpudata,
            S.gpudata,
            points.gpudata,
            depth_map.gpudata
        )

    return mvcnnfp


def multi_view_cnn_fp(
    ray_idxs,
    features,
    P,
    P_inv,
    camera_center,
    bbox,
    S,
    padding,
    batch_size=80000,
    sampling_scheme="sample_in_bbox"
):
    # Extract the number of depth planes and the number of views
    _, D = S.shape
    N, Fh, Fw, F = features.shape
    H = Fh - padding - 1
    W = Fw - padding - 1
    # Make sure that P is a list
    assert len(P) == N

    # Move to GPU to save some time frome copying
    features_gpu = to_gpu(features.ravel())
    s_gpu = to_gpu(
        np.zeros((batch_size, D), dtype=np.float32)
    )
    ray_idxs = to_gpu(ray_idxs.astype(np.int32))
    P_gpu = to_gpu(np.array(P).ravel())
    P_inv_gpu = to_gpu(P_inv.ravel())
    camera_center_gpu = to_gpu(camera_center)

    sim = perform_multi_view_cnn_forward_pass(
        D,
        N,
        F,
        H,
        W,
        padding,
        bbox,
        sampling_scheme
    )

    # Start iterationg over the batch of rays
    for i in range(0, len(ray_idxs), batch_size):
        sim(
            ray_idxs[i:i+batch_size],
            features_gpu,
            P_gpu,
            P_inv_gpu,
            camera_center_gpu,
            s_gpu
        )
        S[i:i+batch_size] = s_gpu.get()

    return S


def multi_view_cnn_fp_with_depth_estimation(
    ray_idxs,
    features,
    P,
    P_inv,
    camera_center,
    bbox,
    padding,
    batch_size=80000,
    sampling_scheme="sample_in_bbox"
):
    # Extract the number of depth planes and the number of views
    _, D = S.shape
    N, Fh, Fw, F = features.shape
    H = Fh - padding - 1
    W = Fw - padding - 1
    # Make sure that P is a list
    assert len(P) == N

    # Move to GPU to save some time frome copying
    features_gpu = to_gpu(features.ravel())
    s_gpu = to_gpu(
        np.zeros((batch_size, D), dtype=np.float32)
    )
    ray_idxs = to_gpu(ray_idxs.astype(np.int32))
    P_gpu = to_gpu(np.array(P).ravel())
    P_inv_gpu = to_gpu(P_inv.ravel())
    camera_center_gpu = to_gpu(camera_center)
    points_gpu = to_gpu(
        np.zeros((batch_size, D, 4), dtype=np.float32)
    )
    depth_map = to_gpu(
        np.zeros((H*W), dtype=np.float32)
    )

    sim = perform_multi_view_cnn_forward_pass_with_depth_estimation(
        D,
        N,
        F,
        H,
        W,
        padding,
        bbox,
        sampling_scheme
    )

    # Start iterationg over the batch of rays
    for i in range(0, len(ray_idxs), batch_size):
        s_gpu.fill(0)
        points_gpu.fill(0)

        sim(
            ray_idxs[i:i+batch_size],
            features_gpu,
            P_gpu,
            P_inv_gpu,
            camera_center_gpu,
            s_gpu,
            points_gpu,
            depth_map
        )

    return depth_map.get().reshape(W, H).T
