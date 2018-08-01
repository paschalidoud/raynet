"""Implement the algorithm for ray tracing developed by John Amanatides and
Andrew Woo in their paper 'A Fast Voxel Traversal Algorithm for Ray Tracing'.

http://www.cse.yorku.ca/~amana/research/grid.pdf
"""

cimport cython
from libc.math cimport floor
from libc.float cimport FLT_EPSILON, FLT_MAX


cdef float _EPS = 1e-2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void sub3(float[:] a, float[:] b, float[:] c) nogil:
    c[0] = a[0] - b[0]
    c[1] = a[1] - b[1]
    c[2] = a[2] - b[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void div3_fi(float[:] a, int[:] b, float[:] c) nogil:
    c[0] = a[0] / b[0]
    c[1] = a[1] / b[1]
    c[2] = a[2] / b[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void to_voxel(float[:] point, float[:] bin_size,
                          int[:] voxel) nogil:
    voxel[0] = <int>floor(point[0] / bin_size[0])
    voxel[1] = <int>floor(point[1] / bin_size[1])
    voxel[2] = <int>floor(point[2] / bin_size[2])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int voxel_equal(int[:] a, int[:] b) nogil:
    return (
        a[0] == b[0] and
        a[1] == b[1] and
        a[2] == b[2]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int is_inside(int[:] a, int[:] b) nogil:
    return (
        (a[0] >= 0 and a[0] < b[0]) and
        (a[1] >= 0 and a[1] < b[1]) and
        (a[2] >= 0 and a[2] < b[2])
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int voxel_traversal(float[:] bbox, int[:] grid_shape, int[:, :] voxels,
                           float[:] ray_start, float[:] ray_end):
    """Traverse the voxel grid and write the voxel indices in the voxels array.

    Arguments
    ---------
        bbox: The two points defining a bounding box
        grid_shape: The number of voxels per axis
        voxels: An array to write the output
        ray_start: The position where the ray intersects the voxel grid for the
                   first time
        ray_end: The position where the ray intersects the voxel grid for the
                 last time
    """
    # We can intersect at the most N voxels
    N = voxels.shape[0]

    # Allocate all memory here and create all the necessary python objects so
    # that the rest of the code is GIL free.
    #
    # NOTICE: Below we allocate memory on the stack as C arrays and then create
    #         python memory views on this memory
    cdef float[3] new_ray_start_d, new_ray_end_d, bin_size_d, ray_d
    cdef int[3] current_voxel_d, last_voxel_d
    cdef float[:] new_ray_start = new_ray_start_d
    cdef float[:] new_ray_end = new_ray_end_d
    cdef float[:] bin_size = bin_size_d
    cdef int[:] current_voxel = current_voxel_d
    cdef int[:] last_voxel = last_voxel_d
    cdef float[:] ray = ray_d
    cdef int stepX, stepY, stepZ
    cdef float tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ
    cdef int ii
    cdef float current_coordinate

    with nogil:
        # Make the rays use bbox[:3] as origin and compute voxel (bin) sizes
        sub3(ray_start, bbox[:3], new_ray_start)
        sub3(ray_end, bbox[:3], new_ray_end)
        sub3(bbox[3:], bbox[:3], bin_size)
        div3_fi(bin_size, grid_shape, bin_size)

        # Compute the ray and the steps for X, Y, Z
        sub3(new_ray_end, new_ray_start, ray)
        stepX = 1 if ray[0] >= 0 else -1
        stepY = 1 if ray[1] >= 0 else -1
        stepZ = 1 if ray[2] >= 0 else -1

        # Move the start and end points just a bit so that they are never on
        # the boundary
        new_ray_start[0] += stepX*bin_size[0]*_EPS
        new_ray_start[1] += stepY*bin_size[1]*_EPS
        new_ray_start[2] += stepZ*bin_size[2]*_EPS
        new_ray_end[0] -= stepX*bin_size[0]*_EPS
        new_ray_end[1] -= stepY*bin_size[1]*_EPS
        new_ray_end[2] -= stepZ*bin_size[2]*_EPS

        # Compute the first and last voxels for our traversal
        to_voxel(new_ray_start, bin_size, current_voxel)
        to_voxel(new_ray_end, bin_size, last_voxel)

        if not is_inside(current_voxel, grid_shape):
            return 0

        # Compute the values of t (u + t*v) where the ray crosses the next
        # boundaries
        tMaxX = tMaxY = tMaxZ = FLT_MAX
        if ray[0] != 0:
            # tMaxX now contains the next voxels boundary in X
            current_coordinate = current_voxel[0]*bin_size[0]
            if  stepX < 0 and current_coordinate < new_ray_start[0]:
                tMaxX = current_coordinate
            else:
                tMaxX = current_coordinate + stepX*bin_size[0]
            # Now it contains the boundary in t
            tMaxX = (tMaxX - new_ray_start[0]) / ray[0]
        if ray[1] != 0:
            # tMaxY now contains the next voxels boundary in Y
            current_coordinate = current_voxel[1]*bin_size[1]
            if stepY < 0 and current_coordinate < new_ray_start[1]:
                tMaxY = current_coordinate
            else:
                tMaxY = current_coordinate + stepY*bin_size[1]
            # Now it contains the boundary in t
            tMaxY = (tMaxY - new_ray_start[1]) / ray[1]
        if ray[2] != 0:
            # tMaxZ now contains the next voxels boundary in Z
            current_coordinate = current_voxel[2]*bin_size[2]
            if stepZ < 0 and current_coordinate < new_ray_start[2]:
                tMaxZ = current_coordinate
            else:
                tMaxZ = current_coordinate + stepZ*bin_size[2]
            # Now it contains the boundary in t
            tMaxZ = (tMaxZ - new_ray_start[2]) / ray[2]

        # Compute how much we need to move in t for the ray to move bin_size in
        # the world coordinates
        tDeltaX = stepX * bin_size[0] / ray[0] if ray[0] != 0 else FLT_MAX
        tDeltaY = stepY * bin_size[1] / ray[1] if ray[1] != 0 else FLT_MAX
        tDeltaZ = stepZ * bin_size[2] / ray[2] if ray[2] != 0 else FLT_MAX

        ii = 0
        voxels[ii, :] = current_voxel
        ii += 1
        while not voxel_equal(current_voxel, last_voxel) and ii < N:
            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    # We move on the X axis
                    current_voxel[0] += stepX
                    if current_voxel[0] < 0 or current_voxel[0] >= grid_shape[0]:
                        return ii
                    tMaxX += tDeltaX
                else:
                    # We move on the Z axis
                    current_voxel[2] += stepZ
                    if current_voxel[2] < 0 or current_voxel[2] >= grid_shape[2]:
                        return ii
                    tMaxZ += tDeltaZ
            else:
                if tMaxY < tMaxZ:
                    # We move on the Y axis
                    current_voxel[1] += stepY
                    if current_voxel[1] < 0 or current_voxel[1] >= grid_shape[1]:
                        return ii
                    tMaxY += tDeltaY
                else:
                    # We move on the Z axis
                    current_voxel[2] += stepZ
                    if current_voxel[2] < 0 or current_voxel[2] >= grid_shape[2]:
                        return ii
                    tMaxZ += tDeltaZ

            voxels[ii, :] = current_voxel
            ii += 1

    return ii  # Return how many voxels were intersected
