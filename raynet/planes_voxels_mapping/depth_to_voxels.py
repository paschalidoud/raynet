from planes_voxels_mapping import depth_to_voxels
from planes_voxels_mapping_cuda import depth_to_voxels as depth_to_voxels_cuda

def get_depth_to_voxels_backend(
    name,
    ray_voxel_count,
    ray_voxel_indices,
    rays_idxs,
    voxel_grid,
    points,
    S,
    S_new=None,
    single_ray_depth_to_voxels=None,
    gamma=None
):
    if name == "cuda":
        return depth_to_voxels_cuda(
            ray_voxel_count,
            ray_voxel_indices,
            rays_idxs,
            voxel_grid,
            points,
            S,
            S_new
        )
    elif name == "numpy":
        return depth_to_voxels(
            ray_voxel_count,
            ray_voxel_indices,
            rays_idxs,
            voxel_grid,
            points,
            S,
            S_new,
            single_ray_depth_to_voxels,
            gamma
        )
    else:
        raise NotImplementedError()
