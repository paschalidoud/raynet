#include <float.h>
#include <stdio.h>


__inline__ __device__ void unravel(int ray_idx, float *pixel) {
    pixel[0] = ray_idx / $height;
    pixel[1] = ray_idx % $height;
}

/**
 * Compute the dot product between a matrix of size 4x3 and a vector 3x1
 * assuming that the vector is in homogenous coordinates meaning its size is
 * 3x1. Store the result as a 3d point (assuming homogenous again).
 */
__inline__ __device__ void dot_m43v3(float *m, float *v, double *out) {
    // Used to normalize the 2d point into homogenous coordinates
    double normalizer;
    normalizer = out[0] = out[1] = out[2] = 0;

    out[0] += m[0*3 + 0] * v[0];
    out[0] += m[0*3 + 1] * v[1];
    out[0] += m[0*3 + 2] * 1.0;

    out[1] += m[1*3 + 0] * v[0];
    out[1] += m[1*3 + 1] * v[1];
    out[1] += m[1*3 + 2] * 1.0;

    out[2] += m[2*3 + 0] * v[0];
    out[2] += m[2*3 + 1] * v[1];
    out[2] += m[2*3 + 2] * 1.0;

    normalizer += m[3*3 + 0] * v[0];
    normalizer += m[3*3 + 1] * v[1];
    normalizer += m[3*3 + 2] * 1.0;

    out[0] /= normalizer;
    out[1] /= normalizer;
    out[2] /= normalizer;
}

/**
 * Sample uniform points in a bounding box
 */
__inline__ __device__  void sample_in_bbox(
    int ray_idx,
    float * P_inv,
    float * camera_center,
    float * ray_start,
    float * ray_end
) {
    float pixel[2], dir[3];
    // Get the pixel based on the ray index
    unravel(ray_idx, pixel);

    // Project the 2d pixel and get the corresponding ray
    double ray[3];
    dot_m43v3(P_inv, pixel, ray);
    for (int i=0; i<3; i++) {
        dir[i] = ray[i] - camera_center[i];
    }

    float t_near = -INFINITY;
    float t_far = INFINITY;
    float t1, t2, t_near_actual, t_far_actual;
    t1 = ($bbox_min_x - camera_center[0]) / dir[0];
    t2 = ($bbox_max_x - camera_center[0]) / dir[0];
    t_near = max(min(t1, t2), t_near);
    t_far = min(max(t1, t2), t_far);

    t1 = ($bbox_min_y - camera_center[1]) / dir[1];
    t2 = ($bbox_max_y - camera_center[1]) / dir[1];
    t_near = max(min(t1, t2), t_near);
    t_far = min(max(t1, t2), t_far);

    t1 = ($bbox_min_z - camera_center[2]) / dir[2];
    t2 = ($bbox_max_z - camera_center[2]) / dir[2];
    t_near = max(min(t1, t2), t_near);
    t_far = min(max(t1, t2), t_far);

    // Swap t_near and t_far in case of negative values
    float near_mask = abs(t_near) < abs(t_far);
    t_near_actual = t_near * near_mask + t_far * (1 - near_mask);
    t_far_actual = (1 - near_mask) * t_near + near_mask * t_far;

    // Compute the ray_start and ray_end
    for (int i=0; i<3; i++) {
        ray_start[i] = camera_center[i] + t_near_actual*dir[i];
        ray_end[i] = camera_center[i] + t_far_actual*dir[i];
    }
}

__global__ void batch_sample_points_in_bbox(
    int n_rays,
    int * ray_idxs,
    float * P_inv,
    float * camera_center,
    float * points
) {
    // Compute the ray that this thread is going to be computing stuff for
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= n_rays)
        return;

    // Get the ray start and ray_end for the current ray
    float ray_start[3], ray_end[3];
    $sampling_scheme(
        ray_idxs[r],
        P_inv,
        camera_center,
        ray_start,
        ray_end
    );

    int offset = r * $depth_planes * 4;
    // Get the rest of the uniformly sampled points
    for (int k=0; k<$depth_planes; k++) {
        points[offset + 4*k] = ray_start[0] + k*(ray_end[0] - ray_start[0])/($depth_planes - 1);
        points[offset + 4*k + 1] = ray_start[1] + k*(ray_end[1] - ray_start[1])/($depth_planes - 1);
        points[offset + 4*k + 2] = ray_start[2] + k*(ray_end[2] - ray_start[2])/($depth_planes - 1);
        points[offset + 4*k + 3] = 1.0;
    }
}
