"""Contains various functions that perform geometrical operations.
"""
import numpy as np

from fast_utils import fast_cross_mm, fast_cross_vm
from checks import assert_col_vectors


def project(P, point, augment_dims=False):
    """Perform an affine transformation on homogenous coordinates.

    Arguments
    ---------
        P: array like shape (D1, D2)
           The projection matrix
        point: array like shape (N, D2)
               One point, duh
    """
    if augment_dims:
        if point.shape[0] < point.shape[1]:
            point = np.hstack((point, np.ones(1).reshape(-1, 1))).T
        else:
            point = np.vstack((point, np.ones(1).reshape(-1, 1)))

    points_hat = np.dot(P, point).T

    # Normalize homogenous coordinataes
    points_hat /= points_hat[:, -1:]

    # If we only projected one point return it as a column vector
    if len(points_hat) == 1:
        points_hat = points_hat.T

    return points_hat


def ray_triangles_intersection_mt(origin, destination, p0, p1, p2):
    """Vectorized implementation of the Moeller and Trumbore method from their
    1997 paper."""

    # Get a normalized ray direction
    ray = destination - origin
    ray /= np.sqrt((ray**2).sum())

    # Calculate the edges sharing the first point (p0)
    e1 = p1 - p0
    e2 = p2 - p0
    pvec = np.empty_like(e2)
    qvec = np.empty_like(e1)

    # Start the calculation of the determinant (see Moelller & Trumbore 1997
    # for what this means)
    # pvec = np.cross(ray, e2)
    fast_cross_vm(ray.astype(np.float32), e2, pvec)
    det = (e1*pvec).sum(axis=1)
    inv_det = 1.0 / det

    # Calculate U
    tvec = origin - p0
    u = (tvec*pvec).sum(axis=1) * inv_det

    # Calculate V
    # qvec = np.cross(tvec, e1)
    fast_cross_mm(tvec.astype(np.float32), e1, qvec)
    v = (ray*qvec).sum(axis=1) * inv_det

    # Get the intersections and compute the points
    idxs = np.logical_and.reduce([u > 0, v > 0, u + v < 1])
    if not np.any(idxs):
        return []

    t = (e2[idxs]*qvec[idxs]).sum(axis=1) * inv_det[idxs]

    return origin + t[:, np.newaxis] * ray


def ray_aabbox_intersection(origin, destination, bbox_min, bbox_max):
    """Computes the intersection of a ray with an axis aligned bounding box
    using the slab method.

    This method looks at the intersection of each pair of slabs by the ray,
    where slab is the space between two parallel planes. It finds t_far and
    t_near for each pair of slabs. If the overall largest t_near value i.e,.
    intersection with the near slab, is greater than the smallest t_far value
    (intersection with far slab) then the ray misses the box, else it hits the
    box.

    Arguments
    ---------
        origin: 4x1, numpy array
                The origin of the ray in homogenous coordinates
        destination: 4x1 numpy array 
                     The destination of the ray in homogenous coordinates
        bbox_min: 1x3 numpy array
                  The minimum vertex of the bounding box
        bbox_max: 1x3 numpy array
                  The maximum vertex of the bounding box
    """
    # Make sure that the origin and the destination are vectors
    assert_col_vectors(origin, destination)
    assert origin.shape == (4, 1)
    assert destination.shape == (4, 1)

    t_near = float("-inf")
    t_far = float("inf")
    direction = destination - origin

    # For each pair of planes associated with each direction
    for i in xrange(len(bbox_min)):
        # If the ray is parallel to the query plane
        if direction[i, 0] == 0:
            # Check if the origin is inside the slabs for the current plane and
            # if it doens't return None because there is no intersection
            if origin[i, 0] < bbox_min[i] or origin[i, 0] > bbox_max[i]:
                return None, None
        else:
            # If ray is not parallel to the planes in the query direction
            # compute the intersection of the query planes with the ray. A ray
            # is defined as
            # R = R_0 + R_d*t =>
            # (X, Y, Z)^T = (X_0, Y_0, Z_0)^T + (X_d, Y_d, Z_d)^T*t

            # If we assume that bbox_min=(x_min, y_min, z_min) and
            # bbox_max=(x_max, y_max, z_max) then the intersections in the i.e.
            # X direction can be given by
            #     x_min = X_0 + X_d * t1 => t1 = (x_min - X_0) / X_d
            #     x_max = X_0 + X_d * t2 => t2 = (x_max - X_0) / X_d

            t1 = (bbox_min[i] - origin[i, 0]) / direction[i, 0]
            t2 = (bbox_max[i] - origin[i, 0]) / direction[i, 0]

            # If t1 > t2 swap them because we want t1 to hold values for the
            # intersection with the near plane
            if t1 > t2:
                tmp = t2
                t2 = t1
                t1 = tmp
            t_near = max(t1, t_near)
            t_far = min(t2, t_far)
            # If the bbox is missed or it is behind the ray there is no
            # intersection
            if t_near > t_far or t_far < 0:
                return None, None

    return t_near, t_far


def is_collinear(p1, p2, p3):
    """Given three points check whether they are collinear

    Arguments:
    ----------
        p1: array(shape=(D, 1))
        p2: array(shape=(D, 1))
        p3: array(shape=(D, 1))
    """
    # Make sure that the inputs are vectors
    assert_col_vectors(p1, p2)
    assert_col_vectors(p2, p3)

    v0 = (p2 - p1).astype(np.float32)
    v1 = (p1 - p3).astype(np.float32)

    return np.allclose(np.cross(v0, v1, axis=0), 0.0, atol=2e-5)


def distance(p1, p2):
    """Given two points compute and return their distance

    Arguments:
    ----------
        p1: array(shape=(D, 1))
        p2: array(shape=(D, 1))
    """

    # Make sure that the inputs are vectors
    assert_col_vectors(p1, p2)

    return np.sqrt(np.sum((p1 - p2)**2))


def is_between(start, stop, p):
    """Given three point check if the query point p is between the other two
    points

    Arguments:
    ----------
        start: array(shape=(D, 1))
        stop: array(shape=(D, 1))
        p: array(shape=(D, 1))
    """
    # Make sure that the inputs are vectors
    assert_col_vectors(start, stop)
    assert_col_vectors(stop, p)

    # First make sure that the three points are collinear
    # if not is_collinear(start, p, stop):
    #     return False
    v0 = p - start
    v1 = stop - p

    # Check that p is between start and stop
    v2 = stop - start

    dot = np.dot(v2.reshape(1, -1), v0)

    # Check that the total distance is equal to the distance from start-point
    # and from point-stop
    d = distance(stop, start)
    d1 = distance(start, p)
    d2 = distance(stop, p)
    if dot < 0 or not np.allclose(d1 + d2, d):
        return False

    return True


def is_between_simple(start, stop, p):
    """Given three point check if the query point p is between the other two
    points

    Arguments:
    ----------
        start: array(shape=(3, 1))
        stop: array(shape=(3, 1))
        p: array(shape=(3, 1))

    """
    # Make sure that the inputs are vectors
    assert_col_vectors(start, stop)
    assert_col_vectors(stop, p)

    return np.logical_and(start[0] <= p[0], stop[0] >= p[0]) and\
        np.logical_and(start[1] <= p[1], stop[1] >= p[1]) and\
        np.logical_and(start[2] <= p[2], stop[2] >= p[2])


def point_in_aabbox(point, bbox_min, bbox_max):
    """Checks wether a point is inside an axis aligned bounding box"""
    return np.all(point >= bbox_min) and np.all(point <= bbox_max)


def ray_ray_intersection(p1, a1, p2, a2):
    """ Given two rays in the form of an origin and a direction find the
    intersection of two rays.

    Each ray can be expressed as follows
        r1 = p1 + a1 * t1
        r2 = p2 + a2 * t2
    In order to compute their intersection, we use the Least squares, namely
        d = (r1 - r2)^2 = 0
    In order to compute the values of t1 and t2, we use sympy

        In [1]: t1, t2 = symbols("t1 t2")
        In [2]: p1x, p1y, p1z = symbols("p1x p1y p1z")
        In [3]: p2x, p2y, p2z = symbols("p2x p2y p2z")
        In [4]: a2x, a2y, a2z = symbols("a2x a2y a2z")
        In [5]: a1x, a1y, a1z = symbols("a1x a1y a1z")
        In [6]: p1 = Matrix([[p1x, p1y, p1z]]).T
        In [7]: p2 = Matrix([[p2x, p2y, p2z]]).T
        In [8]: a1 = Matrix([[a1x, a1y, a1z]]).T
        In [9]: a2 = Matrix([[a2x, a2y, a2z]]).T
        In [10]: d = p1 + t1*a1 - p2 - t2*a2
        In [11]: dt1 = diff(d.T*d, t1)
        In [12]: dt1
        In [13]: dt2 = diff(d.T*d, t2)
        In [14]: solve([dt1, dt2], t1, t2)

    We re-write t1 and t2 produced from [15] in a vectorized manner
    t1 = - ((a2.T * a2)(a1.T * p1 - a1.T * p2) +
            (a1.T * a2)(a2.T * p1 - a2.T * p2))/
           (a1.T * a1) (a2.T * a2) - (a1.T * a.2)^2

    t2 = ((a1.T * a1)(a2.T * p1 - a2.T * p2) -
          (a1.T * a2)(a1.T * p1 - a1.T * p2))/
          (a1.T * a1) (a2.T * a2) - (a1.T * a.2)^2

    Arguments:
    ----------
        p1: array of shape (3, 1)
            Origin of the first ray
        a1: array of shape (3, 1)
            Direction of the first ray
        p2: array of shape (3, 1)
            Origin of the second ray
        a2: array of shape (3, 1)
            Direction of the second ray
    """
    # Make sure that the arguments are correct
    assert_col_vectors(p1, a1)
    assert_col_vectors(a1, a2)
    assert_col_vectors(p1, p2)

    a1_pow2 = np.dot(a1.T, a1)
    a2_pow2 = np.dot(a2.T, a2)
    a1a2 = np.dot(a1.T, a2)

    # divisor = (a1.T * a1) (a2.T * a2) - (a1.T * a.2)^2
    divisor = a1_pow2 * a2_pow2 - a1a2.T*a1a2

    a1p1 = np.dot(a1.T, p1)
    a1p2 = np.dot(a1.T, p2)
    a2p1 = np.dot(a2.T, p1)
    a2p2 = np.dot(a2.T, p2)

    t1 = - a2_pow2 * (a1p1 - a1p2) + a1a2 * (a2p1 - a2p2)
    t1 /= divisor

    # t2 = - a1_pow2 * (a2p1 - a2p2) + a1a2 * (a1p1 - a1p2)
    # t2 /= divisor

    return (p1 + a1 * t1).T


def keep_points_in_aabbox(points, bbox_min, bbox_max):
    """Given a point cloud and the minimum and maximum of an axis-aligned
    bounding box, compute and return only points that are inside the bounding
    box.

    Arguments:
    ----------
        points: array of shape (3, N)
                The 3D points to be decimated
        bbox_min: array of shape (3, 1)
                  The minimum vertex of the bounding box
        bbox_max: array of shape (3, 1)
                  The maximum vertex of the bounding box

    Returns:
    --------
        new_points: array of shape (3, M). where M <= N
    """
    # Make a bunch of sanity checks to make sure that the dimensions of the
    # inputs are correct
    assert points.shape[0] == 3
    assert bbox_min.shape[0] == 3
    assert bbox_max.shape[0] == 3
    assert np.all(bbox_min < bbox_max)

    cnt = 0
    new_points = []

    for i in range(points.shape[1]):
        if point_in_aabbox(points[:, i].reshape(-1, 1), bbox_min, bbox_max):
            new_points.append(points[:, i].reshape(-1, 1))
            cnt += 1

    return np.hstack(new_points)
