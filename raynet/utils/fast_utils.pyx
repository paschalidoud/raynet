
cimport cython
from libc.math cimport sqrt
cimport numpy as cnp

import numpy as np


@cython.boundscheck(False)
cpdef void fast_cross_mm(float[:, :] a, float [:, :] b, float[:, :] r) nogil:
    cdef int N = r.shape[0]
    for i in range(N):
        r[i, 0] = a[i, 1]*b[i, 2] - a[i, 2]*b[i, 1]
        r[i, 1] = a[i, 2]*b[i, 0] - a[i, 0]*b[i, 2]
        r[i, 2] = a[i, 0]*b[i, 1] - a[i, 1]*b[i, 0]


@cython.boundscheck(False)
cpdef void fast_cross_vm(float[:] a, float [:, :] b, float[:, :] r) nogil:
    cdef int N = r.shape[0]
    for i in range(N):
        r[i, 0] = a[1]*b[i, 2] - a[2]*b[i, 1]
        r[i, 1] = a[2]*b[i, 0] - a[0]*b[i, 2]
        r[i, 2] = a[0]*b[i, 1] - a[1]*b[i, 0]


@cython.boundscheck(False)
cpdef inline void fast_cross_vv(float[:] a, float[:] b, float[:] r) nogil:
    r[0] = a[1]*b[2] - a[2]*b[1]
    r[1] = a[2]*b[0] - a[0]*b[2]
    r[2] = a[0]*b[1] - a[1]*b[0]


@cython.boundscheck(False)
cdef inline void sub(float[:] a, float[:] b, float[:] r) nogil:
    r[0] = a[0] - b[0]
    r[1] = a[1] - b[1]
    r[2] = a[2] - b[2]


@cython.boundscheck(False)
cdef inline float dot(float[:] a, float[:] b) nogil:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@cython.boundscheck(False)
cpdef cnp.ndarray[cnp.float32_t, ndim=2] fast_ray_triangles_intersection(
    float[:] origin,
    float[:] destination,
    float[:, :] p0,
    float[:, :] p1,
    float[:, :] p2
):
    # Compute the normalized direction of the ray
    cdef float[3] ray_d
    cdef float[:] ray = ray_d
    cdef float norm = 0
    for i in range(3):
        ray[i] = destination[i] - origin[i]
    for i in range(3):
        norm += ray[i]*ray[i]
    norm = sqrt(norm)
    for i in range(3):
        ray[i] /= norm

    # Start the main ray intersection algorithm (see Moeller & Trumbore 1997)
    cdef int N = p0.shape[0]
    cdef float[3] e1_d
    cdef float[3] e2_d
    cdef float[3] tvec_d
    cdef float[3] pvec_d
    cdef float[3] qvec_d
    cdef float[:] e1 = e1_d
    cdef float[:] e2 = e2_d
    cdef float[:] tvec = tvec_d
    cdef float[:] pvec = pvec_d
    cdef float[:] qvec = qvec_d
    cdef float det, inv_det, eps=1e-6, u, v, t
    cdef cnp.ndarray[cnp.float32_t, ndim=2] results = \
        np.empty(shape=(100, 3), dtype=np.float32)
    cdef int rs_cnt = 0
    with nogil:
        for i in range(N):
            # Calculate the edges sharing the first point
            sub(p1[i], p0[i], e1)
            sub(p2[i], p0[i], e2)

            # Calculate the determinant saving pvec for later use
            fast_cross_vv(ray, e2, pvec)
            det = dot(e1, pvec)

            # If the determinant is 0 then the ray is parallel to the plane
            if -eps < det < eps:
                continue
            inv_det = 1.0 / det

            # Calculate U saving tvec for later
            sub(origin, p0[i], tvec)
            u = dot(tvec, pvec) * inv_det
            if u < 0 or u > 1:
                continue

            # Calculate V
            fast_cross_vv(tvec, e1, qvec)
            v = dot(ray, qvec) * inv_det
            if v < 0 or u + v > 1:
                continue

            # Calculate T to find the intersection point
            t = dot(e2, qvec) * inv_det
            for i in range(3):
                results[rs_cnt, i] = origin[i] + t * ray[i]
            rs_cnt += 1
            if rs_cnt >= 100:
                break

    return results[:rs_cnt]
