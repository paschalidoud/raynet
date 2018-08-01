#include <float.h>
#include <stdio.h>
#include <stdlib.h>


inline __device__ void planes_voxels_mapping(
    float * voxel_grid,
    int * ray_voxel_indices,
    int * ray_voxel_count,
    float * ray_start,
    float * ray_end,
    float * S,
    float * S_new
) {
    // Declare some variables
    int M = ray_voxel_count[0]; // The number of voxels for this ray
    float sum = 0.0;
    float eps = 1e-4;

    // Compute the ray
    float ray[3];
    for (int i=0; i<3; i++) {
        ray[i] = ray_end[i] - ray_start[i];
        // printf("%f %f\\n", ray[i]);
    }
    float ray_norm = 0.0;
    for (int i=0; i<3; i++) {
        ray_norm += ray[i] * ray[i];
    }

    // Declare some variables
    float vd, t, left_d, right_d, coeff_1, coeff_2;
    float start = 0.0;
    float end = 1.0;
    float step = (end - start) / ($depth_planes - 1);
    int left=0, right=1;
    int idx_x, idx_y, idx_z;
    int dim_x = 3*$grid_y*$grid_z;
    int dim_y = 3*$grid_z;
    int dim_z = 3;

    float srsum = 0.0;
    for (int i=0; i<M; i++) {
        // Compute the dot product of the ray with the voxels directions in
        // order to project the voxel centers on the ray
        sum = 0.0;
        idx_x = ray_voxel_indices[3*i];
        idx_y = ray_voxel_indices[3*i + 1];
        idx_z = ray_voxel_indices[3*i + 2];

        //printf("%f-%f-%f \\n", voxel_grid[0], voxel_grid[1], voxel_grid[2]);
        for (int j=0; j<3; j++) {
            // Compute the directions of the voxels centers in this axis
            vd = voxel_grid[idx_x*dim_x + idx_y*dim_y + idx_z*dim_z + j];
            vd -= ray_start[j];
            sum += ray[j] * vd;
        }


        // Update the value and make sure that t is between 0 and 1
        t = clamp(
            sum / ray_norm,
            eps,
            1-eps
        );

        // For every voxel center find the two closest depth planes
        left_d = t - (start + left*step);
        right_d = t - (start + right*step);
        while (left_d > 0 && right_d > 0) {
            left++;
            right++;

            left_d = t - (start + left*step);
            right_d = t - (start + right*step);
        }
        left_d = abs(left_d);
        right_d = abs(right_d);

        // Compute the interpolation coeeficients
        coeff_1 = 1.0 - (left_d / (left_d + right_d));
        coeff_2 = 1.0 - (right_d / (left_d + right_d));

        S_new[i] = coeff_1 * S[left] + coeff_2 * S[right];
        srsum += S_new[i];
    }

    // Normalize the output depth distribution before exiting
    for (int i=0; i<M; i++) {
        S_new[i] = S_new[i] / srsum;
    }
}

__global__ void batch_planes_voxels_mapping(
    int n_rays,
    float * voxel_grid,
    int * ray_voxel_indices,
    int * ray_voxel_count,
    float * ray_start,
    float * ray_end,
    float * S,
    float * S_new
) {
    // Compute the ray that this thread is going to be computing stuff for
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= n_rays)
        return;

    planes_voxels_mapping(
        voxel_grid,
        ray_voxel_indices + 3*$max_voxels*r,
        ray_voxel_count + r,
        ray_start + 3*r,
        ray_end + 3*r,
        S + r*$depth_planes,
        S_new + $max_voxels*r
    );
}

