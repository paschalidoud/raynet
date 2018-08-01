from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from .utils import all_arrays_to_gpu, parse_cu_files_to_string


def batch_mvcnn_voxel_traversal_with_ray_marching(
    M,
    D,
    N,
    F,
    H,
    W,
    padding,
    bbox,
    grid_shape,
    sampling_scheme
):
    """Compile the CUDA kernel that given the features and the camera matrices
    estimates the similarities between the features, performs the marched
    voxels along its ray and does the mapping from depth planes to voxel
    centers.

    Arguments:
    ----------
        M: int, maximum number of marched voxels along ray
        D: int, depth planes (discretization steps)
        N: int, number of views
        F: int, feature size (from the Multi-View CNN)
        H: int, image height
        W: int, image width,
        padding: int, the number of zero-padded pixels around the image to
                 estimate the features from the Multi-View CNN
        bbox: np.array((6,), dtype=np.float32), the coordinates of the bbox
              that enclose the scene
        grid_shape: np.array((3,), dtype=np.int32), the dimensionality of the
                    voxel grid
        sampling_scheme: string, specification of the sampling scheme
    """
    # Set the paths to the files that will be used to construct the cuda kernel
    file_paths = [
        "ray_tracing.cu",
        "utils.cu",
        "planes_voxels_mapping.cu",
        "feature_similarities.cu",
        "sampling_schemes.cu"
    ]

    cu_source_code = parse_cu_files_to_string(file_paths)

    tpl = Template(cu_source_code + """
        __global__ void batch_mvcnn_planes_voxels_with_ray_marching(
            int n_rays,
            int * ray_idxs,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * voxel_grid,
            int * ray_voxel_indices,
            int * ray_voxel_count,
            float * S_new
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
            float S[$depth_planes];
            compute_similarities_per_ray(
                features,
                P,
                ray_start,
                ray_end,
                S
            );

            // Estimate the ray_voxel_indices and the ray_voxel_count
            voxel_traversal(
                ray_start,
                ray_end,
                ray_voxel_indices + r*$max_voxels*3,
                ray_voxel_count + r
            );

            // Map the depth planes to voxel centers
            planes_voxels_mapping(
                voxel_grid,
                ray_voxel_indices + 3*$max_voxels*r,
                ray_voxel_count + r,
                ray_start,
                ray_end,
                S,
                S_new + $max_voxels*r
            );
        }
    """)

    mod = SourceModule(tpl.substitute(
        max_voxels=M,
        depth_planes=D,
        n_views=N,
        padding=padding,
        features_dimensions=F,
        width=W,
        height=H,
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2],
        bbox_min_x=bbox[0],
        bbox_min_y=bbox[1],
        bbox_min_z=bbox[2],
        bbox_max_x=bbox[3],
        bbox_max_y=bbox[4],
        bbox_max_z=bbox[5],
        sampling_scheme=sampling_scheme
    ))
    cuda_fp = mod.get_function("batch_mvcnn_planes_voxels_with_ray_marching")
    cuda_fp.prepare("i" + "P"*9)

    @all_arrays_to_gpu
    def fp(
        ray_idxs,
        features,
        P,
        P_inv,
        camera_center,
        voxel_grid,
        ray_voxel_indices,
        ray_voxel_count,
        S_new,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S_new.shape[1] == M
        assert len(ray_voxel_count.shape) == 1
        assert np.float32 == S_new.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(S_new)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_fp.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            ray_idxs.gpudata,
            features.gpudata,
            P.gpudata,
            P_inv.gpudata,
            camera_center.gpudata,
            voxel_grid.gpudata,
            ray_voxel_indices.gpudata,
            ray_voxel_count.gpudata,
            S_new.gpudata
        )

    return fp


def batch_mvcnn_voxel_traversal_with_ray_marching_with_depth_estimation(
    M,
    D,
    N,
    F,
    H,
    W,
    padding,
    bbox,
    grid_shape,
    sampling_scheme
):
    """Compile the CUDA kernel that given the features and the camera matrices
    estimates the similarities between the features, performs the marched
    voxels along its ray and does the mapping from depth planes to voxel
    centers. Finally directly convert the per voxel depth distribution to a depth map

    Arguments:
    ----------
        M: int, maximum number of marched voxels along ray
        D: int, depth planes (discretization steps)
        N: int, number of views
        F: int, feature size (from the Multi-View CNN)
        H: int, image height
        W: int, image width,
        padding: int, the number of zero-padded pixels around the image to
                 estimate the features from the Multi-View CNN
        bbox: np.array((6,), dtype=np.float32), the coordinates of the bbox
              that enclose the scene
        grid_shape: np.array((3,), dtype=np.int32), the dimensionality of the
                    voxel grid
        sampling_scheme: string, specification of the sampling scheme
    """
    # Set the paths to the files that will be used to construct the cuda kernel
    file_paths = [
        "ray_tracing.cu",
        "utils.cu",
        "planes_voxels_mapping.cu",
        "feature_similarities.cu",
        "sampling_schemes.cu"
    ]

    cu_source_code = parse_cu_files_to_string(file_paths)

    tpl = Template(cu_source_code + """
        __global__ void batch_mvcnn_planes_voxels_with_ray_marchingi_with_depth(
            int n_rays,
            int * ray_idxs,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * voxel_grid,
            int * ray_voxel_indices,
            int * ray_voxel_count,
            float * S_new,
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
            float S[$depth_planes];
            compute_similarities_per_ray(
                features,
                P,
                ray_start,
                ray_end,
                S
            );

            // Estimate the ray_voxel_indices and the ray_voxel_count
            voxel_traversal(
                ray_start,
                ray_end,
                ray_voxel_indices + r*$max_voxels*3,
                ray_voxel_count + r
            );

            // Map the depth planes to voxel centers
            planes_voxels_mapping(
                voxel_grid,
                ray_voxel_indices + 3*$max_voxels*r,
                ray_voxel_count + r,
                ray_start,
                ray_end,
                S,
                S_new + $max_voxels*r
            );

            // We need to find the voxel center with the highest probability
            // based on the S_new
            float * Sr = S_new + r*$max_voxels;
            float max = -INFINITY;
            int max_idx = 0;

            for (int i=0; i<$max_voxels; i++) {
                if (Sr[i] > max) {
                    max_idx = i;
                    max = Sr[i];
                }
            }

            // Associate the voxel_center with id max_idx with a 3D point in
            // world coordinates
            int idx_x, idx_y, idx_z;
            int dim_x = 3*$grid_y*$grid_z;
            int dim_y = 3*$grid_z;
            int dim_z = 3;
            idx_x = ray_voxel_indices[3*$max_voxels*r + 3*max_idx];
            idx_y = ray_voxel_indices[3*$max_voxels*r + 3*max_idx + 1];
            idx_z = ray_voxel_indices[3*$max_voxels*r + 3*max_idx + 2];

            float point[3];
            for (int i=0; i<3; i++) {
                point[i] = voxel_grid[idx_x*dim_x + idx_y*dim_y + idx_z*dim_z + i];
            }

            // Get the distance from the camera center
            float sum = 0.0;
            for (int i=0; i<3; i++) {
                sum += pow(point[i] - camera_center[i], 2);
            }
            depth_map[r] = sqrt(sum);
        }
    """)

    mod = SourceModule(tpl.substitute(
        max_voxels=M,
        depth_planes=D,
        n_views=N,
        padding=padding,
        features_dimensions=F,
        width=W,
        height=H,
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2],
        bbox_min_x=bbox[0],
        bbox_min_y=bbox[1],
        bbox_min_z=bbox[2],
        bbox_max_x=bbox[3],
        bbox_max_y=bbox[4],
        bbox_max_z=bbox[5],
        sampling_scheme=sampling_scheme
    ))
    cuda_fp = mod.get_function("batch_mvcnn_planes_voxels_with_ray_marchingi_with_depth")
    cuda_fp.prepare("i" + "P"*10)

    @all_arrays_to_gpu
    def fp(
        ray_idxs,
        features,
        P,
        P_inv,
        camera_center,
        voxel_grid,
        ray_voxel_indices,
        ray_voxel_count,
        S_new,
        depth_map,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S_new.shape[1] == M
        assert len(ray_voxel_count.shape) == 1
        assert np.float32 == S_new.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(S_new)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_fp.prepared_call(
            (threads, 1),
            (blocks, 1, 1),
            np.int32(n_rays),
            ray_idxs.gpudata,
            features.gpudata,
            P.gpudata,
            P_inv.gpudata,
            camera_center.gpudata,
            voxel_grid.gpudata,
            ray_voxel_indices.gpudata,
            ray_voxel_count.gpudata,
            S_new.gpudata,
            depth_map.gpudata
        )

    return fp

def perform_mvcnn_with_ray_marching_and_voxel_mapping(
    ray_idxs,
    features,
    P,
    P_inv,
    camera_center,
    bbox,
    voxel_grid,
    ray_voxel_indices,
    ray_voxel_count,
    S_new,
    padding,
    depth_planes,
    batch_size=80000,
    sampling_scheme="sample_in_bbox"
):
    # Extract the numbers of views (N), the maximum number of marched voxels
    # (M), the depth planes (D), the image height and the image width
    _, M, _ = ray_voxel_indices.shape
    D = depth_planes
    N, Fh, Fw, F = features.shape
    H = Fh - padding - 1
    W = Fw - padding - 1
    # Make sure that P is a list
    assert len(P) == N

    # Move to GPU to save some time frome copying
    features_gpu = to_gpu(features.ravel())
    ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
    P_gpu = to_gpu(np.array(P).ravel())
    P_inv_gpu = to_gpu(P_inv.ravel())
    camera_center_gpu = to_gpu(camera_center)

    s_gpu = to_gpu(
        np.zeros((batch_size, M), dtype=np.float32)
    )
    ray_voxel_count_gpu = to_gpu(
        np.zeros((batch_size,), dtype=np.int32)
    )
    ray_voxel_indices_gpu = to_gpu(
        np.zeros((batch_size, M, 3), dtype=np.int32)
    )

    fp = batch_mvcnn_voxel_traversal_with_ray_marching(
        M,
        D,
        N,
        F,
        H,
        W,
        padding,
        bbox.ravel(),
        np.array(voxel_grid.shape[1:]),
        sampling_scheme
    )
    voxel_grid = voxel_grid.transpose(1, 2, 3, 0).ravel()
    # Start iterationg over the batch of rays
    for i in range(0, len(ray_idxs), batch_size):
        ray_voxel_indices_gpu.fill(0)
        ray_voxel_count_gpu.fill(0)
        s_gpu.fill(0)

        fp(
            ray_idxs_gpu[i:i+batch_size],
            features_gpu,
            P_gpu,
            P_inv_gpu,
            camera_center_gpu,
            voxel_grid,
            ray_voxel_indices_gpu,
            ray_voxel_count_gpu,
            s_gpu,
        )

        idxs = ray_idxs[i:i+batch_size]
        ray_voxel_indices[idxs] = ray_voxel_indices_gpu.get()[:len(idxs)]
        ray_voxel_count[idxs] = ray_voxel_count_gpu.get()[:len(idxs)]
        S_new[idxs] = s_gpu.get()[:len(idxs)]


def perform_mvcnn_with_ray_marching_and_voxel_mapping(
    ray_idxs,
    features,
    P,
    P_inv,
    camera_center,
    bbox,
    voxel_grid,
    padding,
    depth_planes,
    batch_size=80000,
    sampling_scheme="sample_in_bbox"
):
    # Extract the numbers of views (N), the maximum number of marched voxels
    # (M), the depth planes (D), the image height and the image width
    _, M, _ = ray_voxel_indices.shape
    D = depth_planes
    N, Fh, Fw, F = features.shape
    H = Fh - padding - 1
    W = Fw - padding - 1
    # Make sure that P is a list
    assert len(P) == N

    # Move to GPU to save some time frome copying
    features_gpu = to_gpu(features.ravel())
    ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
    P_gpu = to_gpu(np.array(P).ravel())
    P_inv_gpu = to_gpu(P_inv.ravel())
    camera_center_gpu = to_gpu(camera_center)

    s_gpu = to_gpu(
        np.zeros((batch_size, M), dtype=np.float32)
    )
    ray_voxel_count_gpu = to_gpu(
        np.zeros((batch_size,), dtype=np.int32)
    )
    ray_voxel_indices_gpu = to_gpu(
        np.zeros((batch_size, M, 3), dtype=np.int32)
    )
    depth_map = to_gpu(
        np.zeros(H*W, dtype-np.float32)
    )

    fp = batch_mvcnn_voxel_traversal_with_ray_marching(
        M,
        D,
        N,
        F,
        H,
        W,
        padding,
        bbox.ravel(),
        np.array(voxel_grid.shape[1:]),
        sampling_scheme
    )
    voxel_grid = voxel_grid.transpose(1, 2, 3, 0).ravel()
    # Start iterationg over the batch of rays
    for i in range(0, len(ray_idxs), batch_size):
        ray_voxel_indices_gpu.fill(0)
        ray_voxel_count_gpu.fill(0)
        s_gpu.fill(0)

        fp(
            ray_idxs_gpu[i:i+batch_size],
            features_gpu,
            P_gpu,
            P_inv_gpu,
            camera_center_gpu,
            voxel_grid,
            ray_voxel_indices_gpu,
            ray_voxel_count_gpu,
            s_gpu,
            depth_map
        )
    return depth_map.get().reshape(W, H).T
