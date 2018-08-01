
from keras import backend as K
import numpy as np


def clip_and_renorm(S, ray_voxel_count, eps):
    # Hack to avoid NaN values
    # NOTE: When reweighting account for the different voxel counts per ray
    N, M = K.int_shape(S)
    clipped_S = K.clip(S, eps, 1-eps)
    S_sum = K.sum(clipped_S, axis=1, keepdims=True) - \
        (M - K.reshape(K.cast(ray_voxel_count, K.floatx()), (-1, 1)))*eps
    S_norm = clipped_S / S_sum

    return S_norm


def extract_occupancy_to_ray_pos(
    ray_voxel_indices,
    ray_to_occupancy_accumulated_pon,
    ray_to_occupancy_messages_pon
):
    """Extract the occupancy to ray positive messages for a single ray

    Arguments
    ---------
    ray_voxel_indices: tensor (M, 3), dtype=int32
                       The voxel indices in the voxel grid per ray
    ray_to_occupancy_accumulated_pon: tensor (D1, D2, D3), dtype=float32
        Accumulator used to hold the quotient of the positive ray to occupancy
        message with the negative ray to occupancy message (in logspace)
    ray_to_occupancy_messages_pon: tensor (M, 1), dtype=float32
        Ray to occupancy messages (in logspace)
    """
    # Retain only the voxels through which each ray intersects
    ray_to_occupancy_accumulated_pon = K.tf.gather_nd(
        ray_to_occupancy_accumulated_pon,
        ray_voxel_indices
    )

    # Compute the occupancy to ray messages for all rays
    occupancy_to_ray_pon = (
        ray_to_occupancy_accumulated_pon -
        ray_to_occupancy_messages_pon
    )

    # We need to compute the log-sum-exp expression with large numbers, however
    # this might cause numeric instabilities. Therefore we do the following trick
    max_occupancy_to_ray = K.maximum(0.0, occupancy_to_ray_pon)
    t1 = K.exp(0.0 - max_occupancy_to_ray)
    t2 = K.exp(occupancy_to_ray_pon - max_occupancy_to_ray)

    # Normalize the occupancy to ray message for the positive case
    t = t2 / (t1 + t2)
    occupancy_to_ray = K.clip(t, 1e-4, 1-1e-4)

    return occupancy_to_ray


def single_ray_belief_propagation(
    S,
    ray_voxel_indices,
    ray_to_occupancy_accumulated_pon,
    ray_to_occupancy_messages_pon,
    output_size
):
    """Run the sum product belief propagation for a single ray

    Arguments
    ---------
    S: tensor (M,) dtype=float32
       The depth probability distribution for that ray
    ray_voxel_indices: tensor (M, 3), dtype=int32
                       The voxel indices in the voxel grid per ray
    ray_to_occupancy_accumulated_pon: tensor (D1, D2, D3), dtype=float32
        Accumulator used to hold the quotient of the positive ray to occupancy
        message with the negative ray to occupancy message (in logspace)
    ray_to_occupancy_messages_pon: tensor (M, 1), dtype=float32
        Ray to occupancy messages (in logspace)
    output_size: int
        Pad the output with 0 until its size is output_size
    """
    occupancy_to_ray = extract_occupancy_to_ray_pos(
        ray_voxel_indices,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_messages_pon
    )

    # Compute the cumulative products in linear time (see eq. 13, 14 Ulusoy
    # 3DV)
    # For the computation of the cumulative product we need
    # the occupancy-to-ray messages for the negative case.
    # We append 1 at the top because for the o_1 voxel this term is equal to 1
    occupancy_to_ray_neg_cumprod = K.tf.cumprod(
        1.0 - occupancy_to_ray,
        exclusive=True
    )

    # Compute the part of the messages that is the same for positive and
    # negative messages
    common_part = occupancy_to_ray_neg_cumprod * S
    ray_to_occupancy_new_common = K.tf.cumsum(
        occupancy_to_ray * common_part,
        exclusive=True,
    )

    # Finalize the positive messages
    ray_to_occupancy_new_positive = common_part + ray_to_occupancy_new_common

    # Finalize the negative messages (adding 2nd part of eq. 14 Ulusoy 3DV)
    # The summations we want to calculate are as follows:
    #       i=1, \sum_{i=2}^N(\cdot)
    #       i=2, \sum_{i=3}^N(\cdot)
    #           ...
    #       i=N-2, \sum_{i=N-1}^N(\cdot)
    # lets assume that we have [a, b, c, d, e]. We first inverse the array,
    # thus resulting in [e, d, c, b, a] and then we compute the cumulative sum
    # on this array. The output is [e, e+d, e+d+c, e+d+c+b, e+d+c+b+a]. However
    # we want them in the inverse order, thus we inverse the output once again
    # and we have [e+d+c+b+a, e+d+c+b, e+d+c, e+d, e]
    # Finally we also divide with the incoming message for the negative case
    t1 = K.tf.cumsum(
        occupancy_to_ray * common_part,
        reverse=True,
        exclusive=True
    )
    t2 = t1 / (1.0 - occupancy_to_ray)
    ray_to_occupancy_new_negative = ray_to_occupancy_new_common + t2

    # Normalize the positive ray_to_occupancy message
    ray_to_occupancy_new_pos =\
        ray_to_occupancy_new_positive / (ray_to_occupancy_new_positive + ray_to_occupancy_new_negative)

    ray_to_occupancy_pon = K.log(ray_to_occupancy_new_pos) - K.log(1.0 - ray_to_occupancy_new_pos)

    # Make the size equal to the output_size by appending 0s
    M = K.shape(ray_to_occupancy_pon)[0]
    ray_to_occupancy_pon = K.concatenate([
        ray_to_occupancy_pon,
        K.tf.zeros((output_size-M,))
    ])

    return ray_to_occupancy_pon


def single_ray_depth_estimate(
    S,
    ray_voxel_indices,
    ray_to_occupancy_accumulated_pon,
    ray_to_occupancy_pon,
    output_size
):
    occupancy_to_ray = extract_occupancy_to_ray_pos(
        ray_voxel_indices,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon
    )

    occupancy_to_ray_neg_cumprod = K.tf.cumprod(
        1.0 - occupancy_to_ray,
        exclusive=True
    )

    P = occupancy_to_ray * occupancy_to_ray_neg_cumprod * S
    P = P / K.sum(P)

    M = K.shape(P)[0]
    P = K.concatenate([
        P,
        K.tf.zeros((output_size - M,))
    ])

    return P


def belief_propagation(
    S,
    ray_voxel_indices,
    ray_voxel_count,
    grid_shape,
    gamma=0.031,
    bp_iterations=3
):
    """Run the sum product belief propagation on a batch of ray potentials

    Arguments
    ---------
    S: tensor (N, M) dtype=float32
       The depth probability distribution for each one of the N rays
    ray_voxel_indices: tensor (N, M, 3), dtype=int32
                       The voxel indices in the voxel grid per ray

    ray_voxel_count: array(shape=(N,1), int) The number of voxels
                     intersected by each ray
    grid_shape: tensor (3,) int32, The number of voxels for each axis
    gamma: float32, Prior probabilitiy that the ith voxel is occupied
    bp_iterations: int, Number of belief-propagation iterations
    """
    # Extract the number of the rays as well as the maximum number of marched
    # voxels
    N, M, _ = K.int_shape(ray_voxel_indices)

    # Initialize the ray to occupancy messages to a uniform distribution (and
    # assume that they were accumulated)
    ray_to_occupancy_pon = K.zeros((N, M), dtype="float32")
    prior_pon = K.log(gamma) - K.log(np.float32(1) - gamma)
    #ray_to_occupancy_accumulated_pon_init = K.constant(
    #    prior_pon,
    #    shape=grid_shape
    #)
    ray_to_occupancy_accumulated_pon_init = K.tf.fill(grid_shape, prior_pon)
    ray_to_occupancy_accumulated_pon = ray_to_occupancy_accumulated_pon_init
    #ray_to_occupancy_accumulated_pon = K.variable(
    #    ray_to_occupancy_accumulated_pon_init,
    #    name="ray_to_occupancy_accumulated_pon"
    #)

    # Iterate over the rays
    for it in range(bp_iterations):
        # Compute the messages given the previous messages
        ray_to_occupancy_pon = K.map_fn(
            lambda i: single_ray_belief_propagation(
                S[i, :ray_voxel_count[i]],
                ray_voxel_indices[i, :ray_voxel_count[i]],
                ray_to_occupancy_accumulated_pon,
                ray_to_occupancy_pon[i, :ray_voxel_count[i]],
                M
            ),
            K.tf.range(0, N, dtype="int32"),
            dtype="float32"
        )

        # Accumulate the messages to make the computation of the new messages
        # faster
        ray_to_occupancy_accumulated_pon = K.foldl(
            lambda acc, i: K.tf.sparse_add(
                acc,
                K.tf.SparseTensor(
                    K.cast(ray_voxel_indices[i], dtype="int64"),
                    ray_to_occupancy_pon[i],
                    grid_shape
                )
            ),
            K.tf.range(0, N, dtype="int32"),
            initializer=K.tf.zeros_like(ray_to_occupancy_accumulated_pon)
        )
        ray_to_occupancy_accumulated_pon = ray_to_occupancy_accumulated_pon + ray_to_occupancy_accumulated_pon_init

    return ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon


def depth_estimate(
    S,
    ray_voxel_indices,
    ray_voxel_count,
    ray_to_occupancy_accumulated_pon,
    ray_to_occupancy_pon
):
    N, M = K.int_shape(S)

    return K.map_fn(
        lambda i: single_ray_depth_estimate(
            S[i, :ray_voxel_count[i]],
            ray_voxel_indices[i, :ray_voxel_count[i]],
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_pon[i, :ray_voxel_count[i]],
            M
        ),
        K.tf.range(0, N, dtype="int32"),
        dtype="float32"
    )


def build_belief_propagation(N, M, grid_shape, gamma=0.031, bp_iterations=3):
    """Create a Keras function that implements the sum-product belief
    propagation algorithm for N rays in a voxel grid of shape grid_shape.

    Arguments
    ---------
        N: int, The number of rays
        M: int, The maximum number of voxels a ray can intersect with
        grid_shape: array(3), The shape of the voxel grid
        gamma: float32, The occupancy prior probability per voxel
        bp_iterations: int, The sum-product belief propagation iterations
    """
    # For every ray the per voxel depth probability
    S = K.placeholder(shape=(N, M), dtype="float32", name="S")

    # The voxels that each ray intersects with (in order)
    ray_voxel_indices = K.placeholder(
        shape=(N, M, 3),
        dtype="int32",
        name="ray_voxel_indices"
    )
    ray_voxel_count = K.placeholder(
        shape=(N,),
        dtype="int32",
        name="ray_voxel_count"
    )

    S_norm = clip_and_renorm(S, ray_voxel_count, 1e-5)
    ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon = \
        belief_propagation(
            S_norm,
            ray_voxel_indices,
            ray_voxel_count,
            grid_shape,
            np.float32(gamma),
            bp_iterations
        )

    return K.function(
        [S, ray_voxel_indices, ray_voxel_count],
        [ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon]
    )


def build_depth_estimate(N, M, grid_shape):
    """Create a Keras function that implements the depth estimation based on
    belief's computed using the sum-product belief propagation algorithm

    Arguments
    ---------
        N: int, The number of rays
        M: int, The maximum number of voxels a ray can intersect with
        grid_shape: array(3), The shape of the voxel grid
    """
    # For every ray the per voxel depth probability
    S = K.placeholder(shape=(N, M), dtype="float32", name="S")

    # The voxels that each ray intersects with (in order)
    ray_voxel_indices = K.placeholder(
        shape=(N, M, 3),
        dtype="int32",
        name="ray_voxel_indices"
    )
    ray_voxel_count = K.placeholder(
        shape=(N,),
        dtype="int32",
        name="ray_voxel_count"
    )

    # The ray to occupancy messages 
    ray_to_occupancy_pon = K.placeholder(
        shape=(N, M),
        dtype="float32",
        name="ray_to_occupancy_pon"
    )
    ray_to_occupancy_accumulated_pon = K.placeholder(
        shape=(grid_shape[0], grid_shape[1], grid_shape[2]),
        dtype="float32",
        name="ray_to_occupancy_accumulated_pon"
    )

    S_norm = clip_and_renorm(S, ray_voxel_count, 1e-5)
    S_new = depth_estimate(
        S_norm,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon
    )
    return K.function(
        [
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_pon
        ],
        [S_new]
    )
