#include <stdio.h>

__device__ int grid_index(int *ray_voxel_indices, int i) {
    int x, y, z;
    x = ray_voxel_indices[i];
    y = ray_voxel_indices[i + 1];
    z = ray_voxel_indices[i + 2];

    return $grid_y * $grid_z * x + $grid_z * y + z;
}

__device__ float extract_occupancy_to_ray_pos(
    float * ray_to_occupancy_accumulated_pon,
    float * ray_to_occupancy_messages_pon,
    int * ray_voxel_indices,
    int i
) {
    int grid_idx = grid_index(ray_voxel_indices, 3*i);

    // Compute the occupancy to ray positive/negative (in logspace)
    float occupancy_to_ray_pon =
        ray_to_occupancy_accumulated_pon[grid_idx] -
        ray_to_occupancy_messages_pon[i];

    // Compute the occupancy to ray positive (in normal space :P)
    float max_occupancy_to_ray = max(0.0f, occupancy_to_ray_pon);
    float t1 = exp(0 - max_occupancy_to_ray);
    float t2 = exp(occupancy_to_ray_pon - max_occupancy_to_ray);

    return clamp(
        t2 / (t1 + t2),
        1e-4,
        1-1e-4
    );
}

inline __device__ void depth_estimation(
    float * S,
    int * ray_voxel_indices,
    int * ray_voxel_count,
    float * ray_to_occupancy_accumulated_pon,
    float * ray_to_occupancy_messages_pon,
    float * S_new
) {
    // Declare some variables
    int M = ray_voxel_count[0];  // the number of voxels for this ray
    float occupancy_to_ray_pos;
    float sum, cumprod1, cumprod1_prev;

    // Clip and renorm into Sr
    float * Sr = S;
    sum = 0.0;
    for (int i=0; i<M; i++) {
        Sr[i] = clamp(Sr[i], 1e-5, 1-1e-5);
        sum += Sr[i];
    }
    for (int i=0; i<M; i++) {
        Sr[i] = Sr[i] / sum;
    }

    // Walk the voxels and accumulate the quantities to predict the final
    // depth estimate
    cumprod1 = cumprod1_prev = 1.0f;
    sum = 0.0;
    for (int i=0; i<M; i++) {
        occupancy_to_ray_pos = extract_occupancy_to_ray_pos(
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_messages_pon,
            ray_voxel_indices,
            i
        );

        // Compute the cumulative product of the negative occupancy to ray
        // messages
        cumprod1_prev = cumprod1;
        cumprod1 *= (1.0f - occupancy_to_ray_pos);

        S_new[i] = occupancy_to_ray_pos * cumprod1_prev * Sr[i];
        sum += S_new[i];
    }

    // Normalize the output depth distribution before exiting
    for (int i=0; i<M; i++) {
        S_new[i] = S_new[i] / sum;
    }
}

inline __device__ void belief_propagation(
    float * S,
    int * ray_voxel_indices,
    int * ray_voxel_count,
    float * ray_to_occupancy_accumulated_pon,
    float * ray_to_occupancy_messages_pon,
    float * ray_to_occupancy_accumulated_out_pon,
    float * ray_to_occupancy_messages_out_pon
) {
    // Declare some variables
    int M = ray_voxel_count[0];  // the number of voxels for this ray
    float occupancy_to_ray_pos, ray_to_occupancy_pos, ray_to_occupancy_neg;
    float cumsum1, cumprod1, cumprod1_prev, cumsum2, cumsum2_prev;

    // Clip and renorm into Sr
    float * Sr = S;
    float sum = 0.0;
    for (int i=0; i<M; i++) {
        Sr[i] = clamp(Sr[i], 1e-5, 1-1e-5);
        sum += Sr[i];
    }
    for (int i=0; i<M; i++) {
        Sr[i] = Sr[i] / sum;
    }

    // Walk the voxels and accumulate quantities that will help with the
    // computation of the second part of equation 48.
    cumsum1 = 0.0f;
    cumprod1 = cumprod1_prev = 1.0f;
    for (int i=0; i<M; i++) {
        occupancy_to_ray_pos = extract_occupancy_to_ray_pos(
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_messages_pon,
            ray_voxel_indices,
            i
        );

        // Compute the cumulative product of the negative occupancy to ray
        // messages
        cumprod1_prev = cumprod1;
        cumprod1 *= (1.0f - occupancy_to_ray_pos);

        // Compute the following cumulative sum
        // \sum_{j=1}^i \mu(o=1) \prod_{k=1}^{j-1} \mu(o=0) s_j
        cumsum1 += occupancy_to_ray_pos * cumprod1_prev * Sr[i];
    }

    // Walk the voxels again computing the pon messages
    cumsum2 = cumsum2_prev = 0.0f;
    cumprod1 = cumprod1_prev = 1.0f;
    for (int i=0; i<M; i++) {
        occupancy_to_ray_pos = extract_occupancy_to_ray_pos(
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_messages_pon,
            ray_voxel_indices,
            i
        );

        // Compute the cumulative product of the negative occupancy to ray
        // messages
        cumprod1_prev = cumprod1;
        cumprod1 *= (1.0f - occupancy_to_ray_pos);

        // Compute the common part cumulative sum (see eq. 44, 48)
        cumsum2_prev = cumsum2;
        cumsum2 += occupancy_to_ray_pos * cumprod1_prev * Sr[i];

        // Now we can compute the ray to occupancy messages according to
        // equations 44 and 48
        ray_to_occupancy_pos = cumsum2_prev + cumprod1_prev * Sr[i];
        ray_to_occupancy_neg = cumsum2_prev + (cumsum1 - cumsum2) / (1.0f - occupancy_to_ray_pos);
        // printf("%d-%d-%f\\n", r, i, ray_to_occupancy_pos);

        // Normalize them and save them in the output array in log space
        ray_to_occupancy_pos = ray_to_occupancy_pos / (ray_to_occupancy_pos + ray_to_occupancy_neg);
        ray_to_occupancy_messages_out_pon[i] =
            log(ray_to_occupancy_pos) -
            log(1.0f - ray_to_occupancy_pos);
        // printf("%d-%f\\n", i, ray_to_occupancy_messages_out_pon[i]);
    }

    // Aggregate the ray to occupancy messages
    for (int i=0; i<M; i++) {
        int grid_idx = grid_index(ray_voxel_indices, 3*i);
        atomicAdd(
            ray_to_occupancy_accumulated_out_pon + grid_idx,
            ray_to_occupancy_messages_out_pon[i]
        );
    }
}


__global__ void batch_belief_propagation(
    int n_rays,
    float * S,
    int * ray_voxel_indices,
    int * ray_voxel_count,
    float * ray_to_occupancy_accumulated_pon,
    float * ray_to_occupancy_messages_pon,
    float * ray_to_occupancy_accumulated_out_pon,
    float * ray_to_occupancy_messages_out_pon
) {
    // Compute the ray that this thread is going to be computing stuff for
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= n_rays)
        return;

    belief_propagation(
        S + r*$max_voxels,
        ray_voxel_indices + r*$max_voxels*3,
        ray_voxel_count + r,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon + r*$max_voxels,
        ray_to_occupancy_accumulated_out_pon,
        ray_to_occupancy_messages_out_pon + r*$max_voxels
    );
}

__global__ void batch_depth_estimation(
    int n_rays,
    float * S,
    int * ray_voxel_indices,
    int * ray_voxel_count,
    float * ray_to_occupancy_accumulated_pon,
    float * ray_to_occupancy_messages_pon,
    float * S_new
) {
    // Compute the ray that this thread is going to be computing stuff for
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= n_rays)
        return;
    
    depth_estimation(
        S + r*$max_voxels,
        ray_voxel_indices + r*$max_voxels*3,
        ray_voxel_count + r,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon + r*$max_voxels,
        S_new + r*$max_voxels
    );
}
