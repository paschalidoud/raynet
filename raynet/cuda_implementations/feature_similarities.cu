#include <float.h>
#include <math.h>
#include <stdio.h>

/**
 * Compute the dot product between a matrix of size 3x4 and a vector 4x1
 * assuming that the vector is in homogenous coordinates meaning its size is
 * 4x1. Store the result as a 2d point (assuming homogenous again).
 */
__inline__ __device__ void dot_m34v3(float *m, float *v, float *out) {
    // Used to normalize the 2d point into homogenous coordinates
    float normalizer;
    normalizer = out[0] = out[1] = 0;

    out[0] += m[0*4 + 0] * v[0];
    out[0] += m[0*4 + 1] * v[1];
    out[0] += m[0*4 + 2] * v[2];
    out[0] += m[0*4 + 3] * 1;

    out[1] += m[1*4 + 0] * v[0];
    out[1] += m[1*4 + 1] * v[1];
    out[1] += m[1*4 + 2] * v[2];
    out[1] += m[1*4 + 3] * 1;

    normalizer += m[2*4 + 0] * v[0];
    normalizer += m[2*4 + 1] * v[1];
    normalizer += m[2*4 + 2] * v[2];
    normalizer += m[2*4 + 3] * 1;

    out[0] /= normalizer;
    out[1] /= normalizer;
}

__inline__ __device__ float dot(float *m, float *v) {
    float sum = 0.0;
    for (int i=0; i<$features_dimensions; i++) {
        sum += m[i] * v[i];
    }
    return sum;
}

__inline__ __device__ void pixel_to_features(
    float *x,
    int *f_idx,
    int padding,
    int h,
    int w
) {
    f_idx[0] = round(x[0]) + padding - (padding - 1) /2;
    f_idx[1] = round(x[1]) + padding - (padding - 1) /2;

    // Make sure the index is inside the image boundaries
    f_idx[0] = max(f_idx[0], 0);
    f_idx[0] = min(f_idx[0], w);
    f_idx[1] = max(f_idx[1], 0);
    f_idx[1] = min(f_idx[1], h);

    if (f_idx[0] == 0 || f_idx[1] == 0) {
        f_idx[0] = f_idx[1] = 0;
    }
}

/**
 * ray_start, ray_end and S refer to a specific ray
 */
inline __device__ void compute_similarities_per_ray(
    float * features,
    float * P,
    float * ray_start,
    float * ray_end,
    float * S
) {
    // Declare some useful composite constants
    const int features_height = $height + $padding + 1;
    const int features_width = $width + $padding + 1;
    const int dim_x = features_height*features_width*$features_dimensions;
    const int dim_y = features_width*$features_dimensions;
    const int dim_z = $features_dimensions;

    float pixel_i[2], pixel_j[2], point[3];
    int f_idx[2];
    for (int i=0; i<$n_views; i++) {
        for (int j=i+1; j<$n_views; j++) {
            for (int k=0; k<$depth_planes; k++) {
                // Reproject the 3d point to the views i and j
                point[0] = ray_start[0] + k*(ray_end[0] - ray_start[0])/($depth_planes - 1);
                point[1] = ray_start[1] + k*(ray_end[1] - ray_start[1])/($depth_planes - 1);
                point[2] = ray_start[2] + k*(ray_end[2] - ray_start[2])/($depth_planes - 1);
                dot_m34v3(P + i*4*3, point, pixel_i);
                dot_m34v3(P + j*4*3, point, pixel_j);

                // Figure out the features of view i and pixel pixel_i and j, pixel_j
                pixel_to_features(pixel_i, f_idx, $padding, $height, $width);
                float * features_i = features + dim_x*i + dim_y*f_idx[1] + dim_z*f_idx[0];
                pixel_to_features(pixel_j, f_idx, $padding, $height, $width);
                float * features_j = features + dim_x*j + dim_y*f_idx[1] + dim_z*f_idx[0];

                // Do the dot product
                S[k] += dot(features_i, features_j);
            }
        }
    }

    // Normalize based on the views
    for (int i=0; i<$depth_planes; i++) {
        S[i] /= ($n_views * ($n_views - 1)) / 2;
    }

    // Find the element with the maximum value
    float maximum = -INFINITY;
    for (int i=0; i<$depth_planes; i++) {
        maximum = max(maximum, S[i]);
    }

    // Compute the numerically stable softmax
    float sum = 0.0;
    for (int i=0; i<$depth_planes; i++) {
        S[i] = expf(S[i] - maximum);
        sum += S[i];
    }
    for (int i=0; i<$depth_planes; i++) {
        S[i] /= sum;
    }
}

__global__ void batch_compute_similarities(
    int n_rays,
    float * features,
    float * P,
    float * ray_start,
    float * ray_end,
    float * S
) {
    // Compute the ray that this thread is going to be computing stuff for
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= n_rays)
        return;

    compute_similarities_per_ray(
        features,
        P,
        ray_start + r*3,
        ray_end + r*3,
        S + r*$depth_planes
    );
}
