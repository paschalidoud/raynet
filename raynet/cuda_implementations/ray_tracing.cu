#include <stdio.h>
#include <float.h>
#include <math.h>

inline __device__ int voxel_equal(int * a, int * b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

inline __device__ void voxel_traversal(
    float * ray_start,
    float * ray_end,
    int * ray_voxel_indices,
    int * ray_voxel_count
) {
    // Estimate the dimensionality in every axis of the voxel based on the
    // grid_shape and the bounding box that encloses the scene
    float bin_size[3];
    bin_size[0] = ($bbox_max_x - $bbox_min_x) / $grid_x;
    bin_size[1] = ($bbox_max_y - $bbox_min_y) / $grid_y;
    bin_size[2] = ($bbox_max_z - $bbox_min_z) / $grid_z;

    // Bring the ray_start and ray_end in voxel coordinates
    float new_ray_start[3];
    float new_ray_end[3];
    new_ray_start[0] = ray_start[0] - $bbox_min_x;
    new_ray_start[1] = ray_start[1] - $bbox_min_y;
    new_ray_start[2] = ray_start[2] - $bbox_min_z;
    new_ray_end[0] = ray_end[0] - $bbox_min_x;
    new_ray_end[1] = ray_end[1] - $bbox_min_y;
    new_ray_end[2] = ray_end[2] - $bbox_min_z;

    // Declare some variables that we will need
    float ray[3]; // keeep the ray
    int step[3];
    float tDelta[3];
    int current_voxel[3];
    int last_voxel[3];
    float _EPS = 1e-2;
    for (int i=0; i<3; i++) {
        // Compute the ray
        ray[i] = new_ray_end[i] - new_ray_start[i];

        // Get the step along each axis based on whether we want to move
        // left or right
        step[i] = (ray[i] >=0) ? 1:-1;

        // Compute how much we need to move in t for the ray to move bin_size
        // in the world coordinates
        tDelta[i] = (ray[i] !=0) ? (step[i] * bin_size[i]) / ray[i]: FLT_MAX;

        // Move the start and end points just a bit so that they are never
        // on the boundary
        new_ray_start[i] = new_ray_start[i] + step[i]*bin_size[i]*_EPS;
        new_ray_end[i] = new_ray_end[i] - step[i]*bin_size[i]*_EPS;

        // Compute the first and the last voxels for the voxel traversal
        current_voxel[i] = (int) floor(new_ray_start[i] / bin_size[i]);
        last_voxel[i] = (int) floor(new_ray_end[i] / bin_size[i]);
    }

    // Make sure that the starting voxel is inside the voxel grid
    if (
        ((current_voxel[0] >= 0 && current_voxel[0] < $grid_x) &&
        (current_voxel[1] >= 0 && current_voxel[1] < $grid_y) &&
        (current_voxel[2] >= 0 && current_voxel[2] < $grid_z)) == 0
    ) {
        return;
    }

    // Compute the values of t (u + t*v) where the ray crosses the next
    // boundaries
    float tMax[3];
    float current_coordinate;
    for (int i=0; i<3; i++) {
        if (ray[i] !=0 ) {
            // tMax contains the next voxels boundary in every axis
            current_coordinate = current_voxel[i]*bin_size[i];
            if (step[i] < 0 && current_coordinate < new_ray_start[i]) {
                tMax[i] = current_coordinate;
            }
            else {
                tMax[i] = current_coordinate + step[i]*bin_size[i];
            }
            // Now it contains the boundaries in t units
            tMax[i] = (tMax[i] - new_ray_start[i]) / ray[i];
        }
        else {
            tMax[i] = FLT_MAX;
        }
    }

    // Start the traversal
    int ii = 0;
    // Add the starting voxel
    ray_voxel_indices[3*ii] = current_voxel[0];
    ray_voxel_indices[3*ii + 1] = current_voxel[1];
    ray_voxel_indices[3*ii + 2] = current_voxel[2];
    // Add the current voxel
    ii += 1;
    while (voxel_equal(current_voxel, last_voxel) == 0 && ii < $max_voxels) {
        // if tMaxX < tMaxY
        if (tMax[0] < tMax[1]) {
            if (tMax[0] < tMax[2]) {
                // We move on the X axis
                current_voxel[0] = current_voxel[0] + step[0];
                if (current_voxel[0] < 0 || current_voxel[0] >= $grid_x)
                    break;
                tMax[0] = tMax[0] + tDelta[0];
            }
            else {
                // We move on the Z axis
                current_voxel[2] = current_voxel[2] + step[2];
                if (current_voxel[2] < 0 || current_voxel[2] >= $grid_z)
                    break;
                tMax[2] = tMax[2] + tDelta[2];
            }
        }
        else {
            // if tMaxY < tMaxZ
            if (tMax[1] < tMax[2]) {
                // We move of the Y axis
                current_voxel[1] = current_voxel[1] + step[1];
                if (current_voxel[1] < 0 || current_voxel[1] >= $grid_y)
                    break;
                tMax[1] = tMax[1] + tDelta[1];
            }
            else {
                // We move on the Z axis
                current_voxel[2] = current_voxel[2] + step[2];
                if (current_voxel[2] < 0 || current_voxel[2] >= $grid_z)
                    break;
                tMax[2] = tMax[2] + tDelta[2];
            }
        }

        // Increment the counter of the traversed voxels
        ray_voxel_indices[3*ii] = current_voxel[0];
        ray_voxel_indices[3*ii + 1] = current_voxel[1];
        ray_voxel_indices[3*ii + 2] = current_voxel[2];
        ii += 1;
    }
    ray_voxel_count[0] = ii; // Update the number of the traversed voxels
}

__global__ void batch_voxel_traversal(
    int n_rays,
    float * ray_start,
    float * ray_end,
    int * ray_voxel_indices,
    int * ray_voxel_count
) {
    // Compute the ray that this thread is going to be computing stuff for
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= n_rays)
        return;

    voxel_traversal(
        ray_start + 3*r,
        ray_end + 3*r,
        ray_voxel_indices + r*$max_voxels*3,
        ray_voxel_count + r
    );
}

