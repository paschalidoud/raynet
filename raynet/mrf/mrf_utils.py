import numpy as np


def export_depth_map_from_voxel_indices(
    scene,
    img_idx,
    S,
    ray_voxel_indices,
    ray_idxs,
    grid_shape
):
    """Perform depth estimation for an image given with index i

    Arguments:
    ----------
    scene: Scene object
    img_idx: int, imge index
    S: array(shape=(N, M), dtype=float32)
       The per voxel depth distribution for each ray
    ray_voxel_indices: array(shape=(N, M, 3), dtype=int32)
                       The voxel indices through which the current ray
                       passes
    ray_idxs: array(shape=(N, ), dtype=int32) The ray index
    grid_shape: array(shape=(D1, D2, D3), dtype=int32), The dimensions
                of the voxel grid
    """
    # Extract the dimensions of the camera center
    H, W = scene.image_shape

    # Get the camera center of the reference image
    camera_center = scene.get_image(img_idx).camera.center

    # Get the indices of the voxel centers with the maximum probability
    # based on S
    N, M = S.shape
    idxs = ray_voxel_indices[np.arange(N), S.argmax(axis=-1)]
    # Make sure that the dimensions are correct
    assert idxs.shape[0] == N
    assert idxs.shape[1] == 3
    assert len(ray_idxs) == len(idxs)

    grid = scene.voxel_grid(grid_shape)
    points = grid[:, idxs[:, 0], idxs[:, 1], idxs[:, 2]]
    dist = np.sqrt(
        np.sum((camera_center[:-1] - points)**2, axis=0)
    )

    D = np.zeros((H*W), dtype=np.float32)
    D[ray_idxs] = dist
    D = D.reshape(W, H).T

    return D
