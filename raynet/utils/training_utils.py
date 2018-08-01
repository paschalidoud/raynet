"""Contains general purpose functions that are used throughout the codebase
"""
import numpy as np

from checks import assert_vector_with_wrong_size
from geometry import distance


def get_adjacent_frames_idxs(ref_idx, n_frames, n_adjacent, skip):
    """ Returns the indices of adjacent images of the reference image
    specified with index img_idx. This function is based on the assumption that
    images with neighboring indices are also neighboring_images in 3D space. In
    other words, consecutive frames are neighboring in the 3D space.

    Arguments
    ---------
        ref_idx: int, Index of the image used as reference
        n_frames: int, The total number of images in the dataset
        n_adjacent: int, The number of images used as neighbors
        skip: int, Parameter used to control whether, we will skip images while
              computing the neighbors
    """
    if ref_idx > n_frames:
        raise ValueError("Ref index needs to be smaller than n_frames")
    # Ideally we want to take images before and after the reference image
    median = np.floor(n_adjacent / 2.0)
    # Specify the index of the leftmost neighboring image
    if n_adjacent % 2 == 0:
        min_idx = max(0, ref_idx - median * (skip + 1))
    else:
        min_idx = max(0, ref_idx - median * (skip + 1) - 1)
    # Specify the index of the rightmost neighboring_image
    max_idx = min(n_frames, ref_idx + median * (skip + 1) + 1)

    idxs = np.append(
        np.arange(min_idx, ref_idx, step=skip + 1, dtype=np.uint32),
        np.arange(ref_idx + 1, max_idx, step=skip + 1, dtype=np.uint32)
    )

    if len(idxs) != n_adjacent:
        step = skip + 1

        # Check if it is around the borders
        if ref_idx == 0:
            stop = (n_adjacent + 1) * step
            idxs = np.arange(step, stop, step=step)
        elif ref_idx == n_frames - 1:
            start = ref_idx - n_adjacent * step
            idxs = np.arange(start, ref_idx, step=step)
        elif max(idxs) == n_frames - 1:
            missing = n_adjacent - len(idxs)
            # Check if we have reached the maximum number of frames
            # Insert missing frames
            cnt = 0
            while cnt < missing:
                idxs = np.insert(idxs, 0, min(idxs) - step)
                cnt = cnt + 1
        elif min(idxs) == 0:
            missing = n_adjacent - len(idxs)
            # Check if we have reached the maximum number of frames
            # Insert missing frames
            cnt = 0
            while cnt < missing:
                idxs = np.append(idxs, max(idxs) + step)
                cnt = cnt + 1

    # Return all the indices between min_idx and max_idx except for img_idx
    return idxs


def dirac_distribution(target, points):
    """ Compute a dirac like distribution around the point with the
        smallest distance from the target.

        Arguments
        ---------
           target: array(shape=(4,1), float32), the true depth value
           points: array(shape=(D, 4), float32), D points sampled on each of
                   the N rays

        Returns
        -------
            D: array(shape=(len(points),), float32) the points distribution
    """
    # Make sure that vector has the appropriate size
    assert_vector_with_wrong_size(target, 4)
    D = np.zeros(len(points), dtype=np.float32)

    dists = ((target[:-1].T - points[:, :-1])**2).sum(axis=1)
    D[dists.argmin()] = 1.

    return D


def get_std(stddev_factor, points, std_is_distance):
    p_near = points[0, :-1].reshape(-1, 1)
    p_far = points[-1, :-1].reshape(-1, 1)
    if std_is_distance:
        dists = distance(p_near, p_far)
        std = stddev_factor * dists / len(points)
    else:
        std = stddev_factor * ((p_near - p_far)**2).sum()
        std /= len(points)

    return std


def gaussian_distribution(stddev_factor, std_is_distance):
    def inner(target, points):
        """
        Arguments
        ---------
            target: array(shape=(4,1), float32), the true depth value
            points: array(shape=(D, 4), float32), D points sampled on each of
                    the N rays

        Returns
        -------
            D: array(shape=(len(points),), float32) the points distribution
        """
        # Make sure that vector has the appropriate size
        assert_vector_with_wrong_size(target, 4)

        # Calculate the value of the standard deviation
        std = get_std(stddev_factor, points, std_is_distance)

        dists = ((target[:-1].T - points[:, :-1])**2).sum(axis=-1)
        D = np.exp(-dists/(2*std**2))

        if D.sum() == 0:
            print "Something went wrong"
            print "Distances from target:", dists
            print "target:", target
            print "points:", points
            print std
            import sys
            sys.exit(0)

        return D / D.sum()

    return inner


def get_per_voxel_gaussian_depth_distribution(
    stddev_factor,
    std_is_distance,
    M
):
    def inner(target, points, M):
        """
        Arguments
        ---------
            target: array(shape=(4,1), float32), the true depth value
            points: array(shape=(D, 4), float32), D points sampled on each of
                    the N rays
            M: int, the maximum number of marched voxels
        """
        # Make sure that vector has the appropriate size
        assert_vector_with_wrong_size(target, 4)

        # Make space to save the final depth distribution.
        S = np.zeros((M,), dtype=np.float32)
        D, _ = points.shape

        # Calculate the value of the standard deviation
        std = get_std(stddev_factor, points, std_is_distance)

        dists = ((target[:3].T - points[:, :3])**2).sum(axis=1)
        gd = np.exp(-dists/(2*std**2))
        gd /= gd.sum()
        S[:D] = gd
        idx = S.argsort()

        return S

    return inner


def get_triangles(points, faces):
    triangles = []
    for i in faces:
        t = np.hstack((
            points[i[0]],
            np.hstack((
                points[i[1]],
                points[i[2]]
            ))
        ))
        triangles.append(t)

    return np.array(triangles)


def get_ray_meshes_first_intersection(origin, destination, meshes):
    """ Given a set of meshes that are organized as an octree find the first
    intersection point between the set of meshes and the ray defined with the
    origin and the destination.

    Arguments:
    ---------
        origin: 4x1 array, dtype=np.float32
                The starting 3D point of the ray in homogenous coordinates
        destination: 4x1 array, dtype=np.float32
                     The ending 3D point of the ray in homogenous coordinates
        meshes: struct that models the triangular meshes, normally it is an
                octree object
    Return:
    ------
        target_point: 4x1 array, dtype=np.float32
                      The intersection point

    """
    intersections = meshes.ray_intersections(origin, destination)
    if len(intersections) == 0:
        return None

    dists = ((intersections - origin[:3].T)**2).sum(axis=1)
    target_point = intersections[dists.argmin()].reshape(-1, 1)
    target_point = np.vstack((target_point, [1]))
    # Make sure that the target point will have the correct shape
    assert target_point.shape == (4, 1)

    return target_point
