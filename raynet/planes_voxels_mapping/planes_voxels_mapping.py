import numpy as np
from scipy.interpolate import interp1d

from ..utils.generic_utils import get_voxel_grid


def depth_to_voxels(
    ray_voxel_count,
    ray_voxel_indices,
    rays_idxs,
    voxel_grid,
    points,
    S,
    S_new,
    single_ray_depth_to_voxels,
    gamma=None
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
        voxel_gird: array(shape=(3, D1, D2, D3)), The coordinates of the
                    centers of all voxels in the voxel grid of size
                    (D1, D2, D3)
        points: array(shape=(4, N, D), float32), D points sampled on each of
                the N rays
        S: array(shape=(N, D), float32), A depth probability distribution for
           each of the N rays
        single_ray_depth_to_voxels: callable that sets the depth to voxels
                                    sampling method
        gamma: int, gamma prior for the gaussian distribution, when we do the
               mapping using the kernel density estimation
    """
    # Extract the numbers N, M, D
    N, M, _ = ray_voxel_indices.shape
    _, _, D = points.shape

    # Compute the voxel probabilities
    S_new.fill(0)
    for r, ray_idx in enumerate(rays_idxs):
        C = ray_voxel_count[r]
        idxs = ray_voxel_indices[r, :C]
        S_new[r, :C] = single_ray_depth_to_voxels(
            voxel_grid[:, idxs[:, 0], idxs[:, 1], idxs[:, 2]],
            points[:3, ray_idx],
            S[ray_idx],
            gamma=gamma
        )

    return S_new


def project_voxels_to_rays_vectorized(
    ray_voxels,
    points,
    eps=1e-4,
    clipping=False
):
    """Projects the voxel centers to rays reconstructed by the 3D points

    Arguments
    ---------
        ray_voxel: array(shape=(N, M, 3), int), The indices in the
                           voxel grid per ray
        points: array(shape=(4, N, D), float32), D points sampled on each of
                the N rays
    Returns
    -------
        array(shape=(N, M), float32), the projections of the voxels
    """
    # Compute the directions of all rays
    rays = points[:3, :, -1] - points[:3, :, 0]

    # Taken the closest point as origin, compute the directions of all voxels
    # centers
    voxels_directions = ray_voxels - points[:3, :, 0].T[:, np.newaxis, :]

    # Compute the dot product of the rays with the corresponding voxels
    # directions in order to project the voxel center to the ray
    ray_norms = (rays * rays).sum(axis=0)[:, np.newaxis]
    t = (voxels_directions * rays.T[:, np.newaxis, :]).sum(axis=-1) / ray_norms

    # Make sure that t is between 0 and 1
    if clipping:
        t = np.clip(t, eps, 1-eps)

    return t


def project_voxels_to_ray(ray_voxels, points, eps=1e-4, clipping=False):
    """Projects the voxel centers to a ray constructed by the 3D points

    Arguments
    ---------
        ray_voxels: array(shape=(3, C), float32), The voxels intersecting the
                    ray
        points: array(shape=(3, D), float32), The sampled points on the ray

    Returns
    -------
        array(shape=(1, C), float32), the projections of the voxels
    """
    # Compute the direction of the ray
    ray = (points[:, -1] - points[:, 0]).reshape(3, 1)

    # Taken the point near as origin, compute the directions of all voxels
    # centers
    voxels_directions = ray_voxels - points[:, 0].reshape(3, 1)

    # Compute the dot product of the ray with the voxels directions in order to
    # project the voxel centers on the ray
    t = ray.T.dot(voxels_directions) / ray.T.dot(ray)
    # Make sure that t is between 0 and 1
    if clipping:
        t = np.clip(t, eps, 1-eps)

    return t


def single_ray_depth_to_voxels_li(
    ray_voxels,
    points,
    s,
    gamma=None
):
    """Map the depth probabilities s to voxel probabilities. In this function,
    we firstly project the voxels on the ray and then we compute the distances
    from the 3D points to the projected voxels

    Arguments
    ---------
        ray_voxels: array(shape=(3, C), float32), The voxels intersecting the
                    ray
        points: array(shape=(3, D), float32), The sampled points on the ray
        s: array(shape=(D,)), The depth probability for each of the sampled
           points

    Returns
    -------
        array(shape=(C,), float32), a probability value for each of the voxels
    """
    # Extract the shapes
    _, C = ray_voxels.shape
    D, = s.shape

    # ALlocate memory for the result
    s_new = np.zeros((C,), dtype=np.float32)

    # Project the voxels on the ray and compute the distance
    t = project_voxels_to_ray(ray_voxels, points, clipping=True)
    t_points = np.linspace(0, 1, D)

    s_new = np.interp(t.reshape(-1,), t_points, s)
    s_new /= s_new.sum()

    return s_new


def single_ray_depth_to_voxels_li_2(
    ray_voxels,
    points,
    s,
    gamma=None
):
    """Map the depth probabilities s to voxel probabilities. In this function,
    we firstly project the voxels on the ray and then we compute the distances
    from the 3D points to the projected voxels

    Arguments
    ---------
        ray_voxels: array(shape=(3, C), float32), The voxels intersecting the
                    ray
        points: array(shape=(3, D), float32), The sampled points on the ray
        s: array(shape=(D,)), The depth probability for each of the sampled
           points

    Returns
    -------
        array(shape=(C,), float32), a probability value for each of the voxels
    """
    # Extract the shapes
    _, C = ray_voxels.shape
    D, = s.shape

    # ALlocate memory for the result
    s_new = np.zeros((C,), dtype=np.float32)

    # Project the voxels on the ray and compute the distance
    t = project_voxels_to_ray(ray_voxels, points, clipping=True)
    t_points = np.linspace(0, 1, D)

    distances = np.abs(t_points.reshape(-1, 1) - t.reshape(1, -1))
    # Sort the distances in order to get the two closest neighbors of each
    # voxel
    neighbors = distances.argsort(axis=0)
    # Compute the interpolation coeeficient
    coeff = distances[neighbors[:2, :], np.arange(C)]
    coeff /= coeff.sum(axis=0, keepdims=True)
    coeff = 1.0 - coeff

    # Now we can compute the final probabilities
    s_new[:] = (s[neighbors[:2, :]]*coeff).sum(axis=0)
    s_new /= s_new.sum()

    return s_new


def single_ray_depth_to_voxels_quadratic(
    ray_voxels,
    points,
    s,
    gamma=None
):
    # Extract the shapes
    _, C = ray_voxels.shape
    D, = s.shape

    # ALlocate memory for the result
    s_new = np.zeros((C,), dtype=np.float32)

    # Project the voxels on the ray and compute the distance
    t = project_voxels_to_ray(ray_voxels, points, clipping=True)
    t_points = np.linspace(0, 1, D)

    f = interp1d(t_points, s, kind="quadratic")
    s_new = f(t.reshape(-1,))
    s_new /= s_new.sum()

    return s_new


def single_ray_depth_to_voxels_kde(ray_voxels, points, s, gamma=None):
    """Map the depth probabilities s to voxel probabilities. In this function,
    we firstly project the voxels on the ray and then we compute the distances
    from the 3D points to the projected voxels and the final per voxel
    probability is computed using a (Gaussian) kernel density estimation.

    Arguments
    ---------
        ray_voxels: array(shape=(3, C), float32), The voxels intersecting the
                    ray
        points: array(shape=(3, D), float32), The sampled points on the ray
        s: array(shape=(D,)), The depth probability for each of the sampled
           points
        gamma: int, the gamma value for the gaussian distribution

    Returns
    -------
        array(shape=(C,), float32), a probability value for each of the voxels
    """
    if gamma is None:
        gamma = 10.

    # Extract the shapes
    _, C = ray_voxels.shape
    D, = s.shape

    # ALlocate memory for the result
    s_new = np.zeros((C,), dtype=np.float32)

    # Compute the direction of the ray
    ray = (points[:, -1] - points[:, 0]).reshape(3, 1)
    # ray_norm = np.linalg.norm(ray)
    ray_norm = ray.T.dot(ray)

    # Project the voxels on the ray and compute the distance
    t = project_voxels_to_ray(ray_voxels, points)
    t_points = np.linspace(0, 1, D)

    distances = ((t_points.reshape(-1, 1) - t.reshape(1, -1))**2)*ray_norm
    kernel = np.exp(-distances*gamma)
    s_new[:] = (kernel * s.reshape(-1, 1)).sum(axis=0)
    s_new /= s_new.sum()

    return s_new


def get_single_depth_voxel_mapping(name):
    return {
        "kde": single_ray_depth_to_voxels_kde,
        "li": single_ray_depth_to_voxels_li_2,
        "quadratic": single_ray_depth_to_voxels_quadratic
    }[name]
