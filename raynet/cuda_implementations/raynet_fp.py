from string import Template

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from .utils import all_arrays_to_gpu, parse_cu_files_to_string

def perform_raynet_fp(
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
    does the complete RayNet forward pass, namely it does the ray-marching, the
    depth-to-voxels mapping and the mrf inference.

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
        "sampling_schemes.cu",
        "mrf_bp.cu"
    ]

    cu_source_code = parse_cu_files_to_string(file_paths)
    tpl = Template(cu_source_code + """
        inline __device__ void mvcnn_ray_marching_with_voxels_mapping(
            int ray_idx,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * voxel_grid,
            int * ray_voxel_indices,
            int * ray_voxel_count,
            float * S_voxel_space
        ) {
            // Estimate the ray_start and ray_end for the current pixel
            float ray_start[3], ray_end[3];
            $sampling_scheme(
                ray_idx,
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
                ray_voxel_indices,
                ray_voxel_count
            );

            // Map the depth planes to voxel centers
            planes_voxels_mapping(
                voxel_grid,
                ray_voxel_indices,
                ray_voxel_count,
                ray_start,
                ray_end,
                S,
                S_voxel_space
            );
        }

        __global__ void batch_raynet_fp(
            int n_rays,
            int * ray_idxs,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * voxel_grid,
            int * ray_voxel_indices,
            int * ray_voxel_count,
            float * S_voxel_space,
            float * ray_to_occupancy_accumulated_pon,
            float * ray_to_occupancy_messages_pon,
            float * ray_to_occupancy_accumulated_out_pon,
            float * ray_to_occupancy_messages_out_pon
        ) {
            // Compute the thread
            int r = threadIdx.x + blockDim.x * blockIdx.x;
            if (r >= n_rays)
                return;

            mvcnn_ray_marching_with_voxels_mapping(
                ray_idxs[r],
                features,
                P,
                P_inv,
                camera_center,
                voxel_grid,
                ray_voxel_indices + r*3*$max_voxels,
                ray_voxel_count + r,
                S_voxel_space + r*$max_voxels
            );

            // Do the belief propagation for the current batch
            belief_propagation(
                S_voxel_space + r*$max_voxels,
                ray_voxel_indices + 3*r*$max_voxels,
                ray_voxel_count + r,
                ray_to_occupancy_accumulated_pon,
                ray_to_occupancy_messages_pon + r*$max_voxels,
                ray_to_occupancy_accumulated_out_pon,
                ray_to_occupancy_messages_out_pon + r*$max_voxels
            );
        }

        __global__ void batch_complete_depth_estimation(
            int n_rays,
            int * ray_idxs,
            float * features,
            float * P,
            float * P_inv,
            float * camera_center,
            float * voxel_grid,
            int * ray_voxel_indices,
            int * ray_voxel_count,
            float * S_voxel_space,
            float * ray_to_occupancy_accumulated_pon,
            float * ray_to_occupancy_messages_pon,
            float * depth_map
        ) {
            // Compute the thread
            int r = threadIdx.x + blockDim.x * blockIdx.x;
            if (r >= n_rays)
                return;

            mvcnn_ray_marching_with_voxels_mapping(
                ray_idxs[r],
                features,
                P,
                P_inv,
                camera_center,
                voxel_grid,
                ray_voxel_indices + r*3*$max_voxels,
                ray_voxel_count + r,
                S_voxel_space + r*$max_voxels
            );

            // Do the depth estimation
            depth_estimation(
                S_voxel_space + r*$max_voxels,
                ray_voxel_indices + 3*r*$max_voxels,
                ray_voxel_count + r,
                ray_to_occupancy_accumulated_pon,
                ray_to_occupancy_messages_pon + r*$max_voxels,
                S_voxel_space + r*$max_voxels
            );

            // We need to find the voxel center with the highest probability
            // based on the S_new
            float * Sr = S_voxel_space + r*$max_voxels;
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
    cuda_fp = mod.get_function("batch_raynet_fp")
    cuda_fp.prepare("i" + "P"*13)

    mod_de = SourceModule(tpl.substitute(
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
    cuda_de = mod_de.get_function("batch_complete_depth_estimation")
    cuda_de.prepare("i" + "P"*12)

    @all_arrays_to_gpu
    def raynet_fp(
        ray_idxs,
        features,
        P,
        P_inv,
        camera_center,
        voxel_grid,
        ray_voxel_indices,
        ray_voxel_count,
        S_voxel_space,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon,
        ray_to_occupancy_accumulated_out_pon,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S_voxel_space.shape[1] == M
        assert ray_voxel_indices.shape[1:] == (M, 3)
        assert len(ray_voxel_count.shape) == 1
        assert len(ray_voxel_count) == len(S_voxel_space) == len(ray_voxel_indices)
        assert S_voxel_space.shape[1] == ray_to_occupancy_messages_pon.shape[1]
        assert ray_to_occupancy_accumulated_pon.shape == tuple(grid_shape)
        assert ray_to_occupancy_accumulated_out_pon.shape == tuple(grid_shape)
        assert np.float32 == S_voxel_space.dtype
        assert np.float32 == ray_to_occupancy_messages_pon.dtype
        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(S_voxel_space)
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
            S_voxel_space.gpudata,
            ray_to_occupancy_accumulated_pon.gpudata,
            ray_to_occupancy_messages_pon.gpudata,
            ray_to_occupancy_accumulated_out_pon.gpudata,
            ray_to_occupancy_messages_pon.gpudata
        )

        return ray_to_occupancy_messages_pon

    @all_arrays_to_gpu
    def raynet_de(
        ray_idxs,
        features,
        P,
        P_inv,
        camera_center,
        voxel_grid,
        ray_voxel_indices,
        ray_voxel_count,
        S_voxel_space,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon,
        depth_map,
        threads=2048
    ):
        # Assert everything is the right size, shape and dtype
        assert S_voxel_space.shape[1] == M
        assert ray_voxel_indices.shape[1:] == (M, 3)
        assert len(ray_voxel_count.shape) == 1
        assert len(ray_voxel_count) == len(S_voxel_space) == len(ray_voxel_indices)
        assert S_voxel_space.shape[1] == ray_to_occupancy_messages_pon.shape[1]
        assert ray_to_occupancy_accumulated_pon.shape == tuple(grid_shape)
        assert np.float32 == S_voxel_space.dtype
        assert np.float32 == ray_to_occupancy_messages_pon.dtype
        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype

        # Determine the grid and block arguments
        n_rays = len(S_voxel_space)
        blocks = n_rays / threads + int(n_rays % threads != 0)

        cuda_de.prepared_call(
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
            S_voxel_space.gpudata,
            ray_to_occupancy_accumulated_pon.gpudata,
            ray_to_occupancy_messages_pon.gpudata,
            depth_map.gpudata
        )

    return raynet_fp, raynet_de
