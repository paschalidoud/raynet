import numpy as np


def clip_and_renorm(x, eps=1e-5):
    """Clip the values of x from eps to 1-eps and renormalize them so that they
    sum to 1."""
    x = np.clip(x, eps, 1-eps)
    return x / x.sum()


def single_ray_belief_propagation(ray_voxel_indices,
                                  ray_to_occupancy_accumulated_pon,
                                  ray_to_occupancy_pon, s):
    """Run the sum-product belief propagation for a single ray accumulating the
    occupancy to ray messages in log space and producing the new ray to
    occupancy messages.

    Arguments
    ---------
    ray_voxel_indices: array(shape=(M, 3), dtype=int32)
                       The voxel indices through which the current ray passes
    ray_to_occupancy_accumulated_pon: array(shape=(D, D, D), dtype=float32)
                                  Accumulated in log space the
                                  positive ray to occupancy messages over the
                                  negative ray to occupancy messages from the
                                  previous iteration of the belief propagation
                                  algorithm
    ray_to_occupancy_pon: array(shape=(M, 1), dtype=float32)
                      The positive ray to occupancy message over the negative
                      ray to occupancy message from the previous belief
                      propagation iteration
    s: array(shape=(M,), dtype=float32)
    """
    # Create an index that when passed to a numpy array will return the voxels
    # that this ray passes through
    # TODO: Remove this check. This is just to make the code run for the
    # 2D tests.
    if ray_voxel_indices.shape[-1] == 3:
        indices = (
            ray_voxel_indices[:, 0],
            ray_voxel_indices[:, 1],
            ray_voxel_indices[:, 2]
        )
    else:
        indices = (
            ray_voxel_indices[:, 0],
            ray_voxel_indices[:, 1]
        )

    # Compute the the occupancy_to_ray message
    # NOTE: The ray_to_occupancy_accumulated is in log space
    occupancy_to_ray_pon = (
        ray_to_occupancy_accumulated_pon[indices] -
        ray_to_occupancy_pon
    )
    # We assume that incoming messages are normalized to 1, thus we need to
    # normalize the occupancy-to-ray message
    # Make sure that the occupancy-to-ray message for every voxel is greater or
    # equal to 0
    max_occupancy_to_ray = np.maximum(0.0, occupancy_to_ray_pon)
    t1 = np.exp(0.0 - max_occupancy_to_ray)
    t2 = np.exp(occupancy_to_ray_pon - max_occupancy_to_ray)

    # Now we normalize the occupancy to ray message for the positive case.
    # The occupancy_to_ray holds the positive occupancy-to-ray messages for the
    # current ray (not in logspace) from Equation (44) in my report
    occupancy_to_ray = np.clip(
        t2 / (t2 + t1),
        1e-4,
        1-1e-4
    )

    # Compute the cumulative products in linear time (see eq. 13, 14 Ulusoy
    # 3DV)
    # For the computation of the cumulative product we need
    # the occupancy-to-ray messages for the negative case.
    # We append 1 at the top because for the o_1 voxel this term is equal to 1
    occupancy_to_ray_neg_cumprod = np.hstack([
        [1.], (1 - occupancy_to_ray).cumprod()
    ])

    # Get the number of voxels that intersect with the ray
    M = ray_to_occupancy_pon.shape[0]
    # Make space to compute the ray to occupancy messages for both the positive
    # and the negative case according to eq 44, 48 in my report
    ray_to_occupancy_new = np.zeros((2, M), dtype=np.float32)

    # Compute the part of the messages that is the same for positive and
    # negative messages
    ray_to_occupancy_new[:] += np.hstack([
        [0.], occupancy_to_ray * occupancy_to_ray_neg_cumprod[:-1] * s
    ])[:-1].cumsum()

    # Finalize the positive messages
    ray_to_occupancy_new[1] += occupancy_to_ray_neg_cumprod[:-1] * s

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
    ray_to_occupancy_new[0] += np.hstack([
            occupancy_to_ray * occupancy_to_ray_neg_cumprod[:-1] * s,
            [0.0]
    ])[::-1].cumsum()[::-1][1:] / (1 - occupancy_to_ray)

    # Normalize the positive ray_to_occupancy message
    ray_to_occupancy_new_pos =\
        ray_to_occupancy_new[1] / (ray_to_occupancy_new[1] + ray_to_occupancy_new[0])

    # Return the quotient of the positive ray to occupancy message with the
    # negative ray to occupancy message in logspace
    t = np.log(ray_to_occupancy_new_pos) - np.log(1 - ray_to_occupancy_new_pos)

    if np.isnan(t).any() or np.isinf(t).any():
        print "ray_to_occupancy_pon contains weird values %r" % (t)
        print "ray_to_occupancy_new_pos", ray_to_occupancy_new_pos

    return t


def single_ray_depth_estimate(
    ray_voxel_indices,
    ray_to_occupancy_accumulated_pon,
    ray_to_occupancy_pon,
    s
):
    """Compute the depth distribution for each ray according to eq 55 in my
    report.

    Arguments:
    ----------
    ray_voxel_indices: array(shape=(M, 3), dtype=int32)
                       The voxel indices through which the current ray passes
    ray_to_occupancy_accumulated_pon: array(shape=(D, D, D), dtype=float32)
                                  Accumulated in log space the
                                  positive ray to occupancy messages over the
                                  negative ray to occupancy messages from the
                                  previous iteration of the belief propagation
                                  algorithm
    ray_to_occupancy_pon: array(shape=(M, 1), dtype=float32)
                      The positive ray to occupancy message over the negative
                      ray to occupancy message from the previous belief
                      propagation iteration
    s: array(shape=(M,), dtype=float32)
    """
    # Create an index that when passed to a numpy array will return the voxels
    # that this ray passes through
    if ray_voxel_indices.shape[-1] == 3:
        indices = (
            ray_voxel_indices[:, 0],
            ray_voxel_indices[:, 1],
            ray_voxel_indices[:, 2]
        )
    else:
        indices = (
            ray_voxel_indices[:, 0],
            ray_voxel_indices[:, 1]
        )

    # Compute the log of the occupancy_to_ray message for the positive case
    # NOTE: The ray_to_occupancy_accumulated is in log space
    occupancy_to_ray_pon = (
        ray_to_occupancy_accumulated_pon[indices] -
        ray_to_occupancy_pon
    )

    # We assume that incoming messages are normalized to 1, thus we need to
    # normalize the occupancy-to-ray message
    max_occupancy_to_ray = np.maximum(0, occupancy_to_ray_pon)
    t1 = np.exp(0.0 - max_occupancy_to_ray)
    t2 = np.exp(occupancy_to_ray_pon - max_occupancy_to_ray)

    # Now we normalize the occupancy to ray message for the positive case.
    # NOTE: We only normalize and store the occupancy-to-ray message for the
    # positive case
    # The occupancy_to_ray holds the positive occupancy-to-ray messages for the
    # current ray (not in logspace) from Equation (44) in my report
    occupancy_to_ray = np.clip(
        t2 / (t2 + t1),
        1e-4,
        1-1e-4
    )

    # Compute the cumulative products in linear time (see eq. 13, 14 Ulusoy
    # 3DV)
    # For the computation of the cumulative product we need
    # the occupancy-to-ray messages for the negative case.
    # We append 1 at the top because for the o_1 voxel this term is equal to 1
    occupancy_to_ray_neg_cumprod = np.hstack([
        [1.], (1 - occupancy_to_ray).cumprod()
    ])

    P = occupancy_to_ray * occupancy_to_ray_neg_cumprod[:-1] * s

    return P / P.sum()


def compute_occupancy_probabilities(
    ray_to_occupancy_accumulated_pon,
    gamma=0.031
):
    """Compute the approximate marginal distributions of each occupancy
    variable

    Arguments:
    ----------
        ray_to_occupancy_accumulated_pon: array(shape=(D, D, D), dtype=float32)
                                      Accumulated in log space the
                                      positive ray to occupancy messages over
                                      the negative ray to occupancy messages
                                      from the previous iteration of the belief
                                      propagation algorithm
        gamma: float32, Prior probabilitiy that the ith voxel is occupied
    """
    # The probability of the i^th voxel is given from
    # p(o_i) =\
    # \mu_{\phi \to o_i}(o_i) \prod_{\psi_k \in L} \mu_{\psi_k \to o_i}(o_i)
    #
    # We need to compute the \prod_{\psi_k \in L} \mu_{\psi_k \to o_i}(o_i)
    # either for the positive or for the negative case. However, we only have
    # their quotient and we know that they sum to 1
    # A + B = 1 && A/B = C => CB + B = 1 => B = 1/(C+1)

    # We need to exponentiate the ray_to_occupancy_accumulated_pon
    max_ray_to_occupancy_accumulated_pon =\
        np.maximum(0.0, ray_to_occupancy_accumulated_pon)
    t1 = np.exp(0.0 - max_ray_to_occupancy_accumulated_pon)
    t2 = np.exp(
        ray_to_occupancy_accumulated_pon - max_ray_to_occupancy_accumulated_pon
    )

    return t2 / (t2 + t1)


def belief_propagation(
    S,
    ray_voxel_indices,
    ray_voxel_count,
    ray_to_occupancy_messages_pon,
    grid_shape,
    gamma=0.05,
    bp_iterations=3,
    progress_callback=lambda *args: None
):
    """Run the belief propagation for a set of rays

    Arguments:
    ---------
        S: array(shape=(N, M), float32), A depth probability distribution for
           each of the N rays
        ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                           voxel grid per ray. M denotes the maximum number of
                           marched voxels.
        ray_voxel_count: array(shape=(N,1), int) The number of voxels
                         intersected by each ray
        ray_to_occupancy_messages_pon: array(shape=(N, M), float32), Holds the
                                       ray_to_occupancy messages between bp
                                       iterations
        grid_shape: array(shape=(3,), int), The number of voxels for each axis
        gamma: float32, Prior probabilitiy that the ith voxel is occupied
        bp_iterations: Number of belief-propagation iterations
    """
    # Extract the number of rays
    N, M = S.shape

    # Initialize the ray to occupancy messages to uniform
    ray_to_occupancy_messages_pon.fill(0)

    # Initialize the ray-to-occupancy accumulated to $\phi(o_i)$ The
    # ray_to_occupancy_accumulated_prev_pon and the
    # ray_to_occupancy_accumulated_new_pon holds the accumulation of the
    # quotient of the positive ray to occupancy message with the negative ray
    # to occupancy message in log space for the current and for the next belief
    # propagation iteration.
    # Both messages are initialized to
    # \log(\frac{\phi_(o_i=1)}{\phi_(o_i=0)}
    ray_to_occupancy_accumulated_prev_pon = np.ones(
        tuple(grid_shape),
        dtype=np.float32
    ) * (np.log(gamma) - np.log(1 - gamma))
    ray_to_occupancy_accumulated_new_pon = np.ones(
        tuple(grid_shape),
        dtype=np.float32
    ) * (np.log(gamma) - np.log(1 - gamma))

    # Iterate over the rays multiple times
    for it in xrange(bp_iterations):
        print "Iteration %d " % (it,)
        for r in xrange(N):
            # Get the actual number of voxels which this ray passes through
            c = ray_voxel_count[r]
            if c <= 1:
                continue
            ray_to_occupancy_pon = single_ray_belief_propagation(
                ray_voxel_indices[r, :c, :],
                ray_to_occupancy_accumulated_prev_pon,
                ray_to_occupancy_messages_pon[r, :c],
                clip_and_renorm(S[r, :c])
            )

            idxs = ray_voxel_indices[r, :c]
            idxs = (idxs[:, 0], idxs[:, 1], idxs[:, 2])
            ray_to_occupancy_accumulated_new_pon[idxs] += ray_to_occupancy_pon

            # Update the array of the ray-to-occupancy messages with the
            # current message that will be used for the next iteration
            ray_to_occupancy_messages_pon[r, :c] = ray_to_occupancy_pon

        # Swap the accumulators for the next bp iteration
        ray_to_occupancy_accumulated_prev_pon[:] = ray_to_occupancy_accumulated_new_pon
        ray_to_occupancy_accumulated_new_pon.fill(np.log(gamma) - np.log(1 - gamma))

        progress_callback(
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_messages_pon,
            ray_to_occupancy_accumulated_prev_pon,
            it
        )

    return ray_to_occupancy_accumulated_prev_pon, ray_to_occupancy_messages_pon


def compute_depth_distribution(
    S,
    ray_voxel_indices,
    ray_voxel_count,
    ray_to_occupancy_messages_pon,
    ray_to_occupancy_accumulated_pon,
    S_new
):
    """Perform the depth estimation according to Equation 55 in my report

       p(D_r = d_i) = 1\Z \mu_{o_i \to \psi_r}(o_i=1)
            \prod_{j=1}^{i-1} \mu_{o_j \to \psi_r}(o_j = 0) s_i

    Arguments:
    ----------
        S: array(shape=(N, M), float32), A depth probability distribution for
           each of the N rays
        ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                           voxel grid per ray
        ray_voxel_count: array(shape=(N,1), int) The number of voxels
                         intersected by each ray
        ray_to_occupancy_messages_pon: array(shape=(N, M), float32), The ray to
                                       occupancy messages for the positve
                                       case over the ray to occupancy messages
                                       for the negative case in logspace from
                                       the last bp iteration
        ray_to_occupancy_accumulated_pon: array(shape=(D, D, D), float32), The
                                          accumulation of the quotient of the
                                          ray to occupancy messages in the
                                          positive and in the negative case
        S_new: array(shape=(N, M), float32), The final depth probability
           distribution for each of the N rays
    """
    # Extract the number of rays
    N, M = S.shape

    # Fill S_new with zeros
    S_new.fill(0)

    # Iterate over the rays
    for r in range(N):
        # Get the actual number of voxels which this ray passes through
        c = ray_voxel_count[r]
        if c <= 1:
            continue
        S_new[r, :c] = single_ray_depth_estimate(
            ray_voxel_indices[r, :c, :],
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_messages_pon[r, :c],
            clip_and_renorm(S[r, :c])
        )

    return S_new
