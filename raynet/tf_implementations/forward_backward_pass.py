from keras import backend as K
K.set_learning_phase(0)

import numpy as np

from .loss_functions import emd, squared_emd, expected_squared_error
from ..mrf.mrf_tf import belief_propagation, clip_and_renorm, depth_estimate


def compute_similarities(n1, n2, features):
    """
    Arguments
    ---------
    n1: int
        Index of the first view in the pair of image patches
    n2: int
        Index of the second view in the pair of image patches
    features: tensor 5xF
        Features for all images
    """
    # Compute the features from the images with index n1 and n2
    F1 = features[n1]
    F2 = features[n2]

    N, D, f = K.int_shape(F1)

    sim = K.reshape(
        K.batch_dot(
            K.reshape(F1, (N*D, 1, f)),
            K.reshape(F2, (N*D, f, 1)),
        ),
        (N, D)
    )
    return sim


def single_ray_depth_to_voxels_map(ray_voxels, points, s, M):
    """ Map the depth probability s to voxel probabilities using kernel density
    estimation

    Arguments
    ---------
        ray_voxels: shape=(C, 3), float32, The voxels intersecting with the ray
        points: shape=(4, D), float32, The 3D sampled points
        s: shape=(D), The depth probability for each one of the sampled points
        M: maximum number of marched voxels
    """
    D, = K.int_shape(s)
    # Compute the direction of the ray
    ray = K.reshape(points[:-1, -1] - points[:-1, 0], (-1, 1))
    ray_norm = K.dot(K.transpose(ray), ray)
    # Compute the direction of all voxel centers
    voxels_direction = ray_voxels - points[:-1, 0]

    # Compute the dot product of the ray with the voxel directions
    t = K.dot(voxels_direction, ray)
    t = t / ray_norm
    t_points = K.tf.lin_space(0.0, 1.0, D)

    dist = K.pow(t_points - t, 2) * ray_norm
    gamma = 10.
    kernel = K.exp(-dist * gamma)

    s_new = K.sum(kernel * s, axis=1)
    s_new = s_new / K.sum(s_new)

    C = K.shape(s_new)[0]
    s_new = K.concatenate([
        s_new,
        K.tf.zeros((M-C,))
    ])

    return s_new


def single_ray_depth_to_voxels_map_li(ray_voxels, points, s, M):
    """ Map the depth probability s to voxel probabilities using linear
    interpolation.

    Arguments
    ---------
        ray_voxels: shape=(C, 3), float32, The voxels intersecting with the ray
        points: shape=(4, D), float32, The 3D sampled points
        s: shape=(D), The depth probability for each one of the sampled points
        M: maximum number of marched voxels
    """
    # Extract dimensions
    D, = K.int_shape(s)
    C = K.shape(ray_voxels)[0]

    # Compute the direction of the ray
    ray = K.reshape(points[:-1, -1] - points[:-1, 0], (-1, 1))
    ray_norm = K.dot(K.transpose(ray), ray)
    # Compute the direction of all voxel centers
    voxels_direction = ray_voxels - points[:-1, 0]

    # Compute the dot product of the ray with the voxel directions
    t = K.dot(voxels_direction, ray)
    t = t / ray_norm
    t_points = K.tf.lin_space(0.0, 1.0, D)

    dist = K.abs(t_points - t)
    values, neighbors_idxs = K.tf.nn.top_k(-dist, k=2)

    # Compute the interpolation coeeficient
    coeff = - values
    coeff = coeff / K.sum(coeff, axis=1, keepdims=True)
    coeff = 1.0 - coeff
    s_new = K.reshape(
        K.tf.gather_nd(
            s,
            K.reshape(neighbors_idxs, (-1, 1))
        ),
        (-1, 2)
    )*coeff
    s_new = K.sum(s_new, axis=1)
    s_new = s_new / K.sum(s_new)

    N = K.shape(s_new)[0]
    s_new = K.concatenate([
        s_new,
        K.tf.zeros((M-N,))
    ])

    return s_new


def forward_backward_pass(
    model,
    images,
    voxel_grid,
    ray_voxel_indices,
    ray_voxel_count,
    S_target,
    points,
    camera_centers,
    views=5,
    gamma=0.031,
    bp_iterations=3,
    loss="squared_emd"
):
    """
    Arguments
    ---------
    model: Keras CNN model
    images: list[tensors, shape(n_rays, D, H, W, 3), dtype=float32]
            A list of tensors containing the images (patches) used for the
            forward pass
    voxel_grid: tensor 3xD1xD2xD3, float32
        The tensor containing the centers of the voxels
    ray_voxel_indices: tensor shape=(N, M, 3), int32
        The number of voxels intersected by each ray
    ray_voxel_count: tensor shape=(N,), int32
        The indices in the voxel grid per ray
    S_target: tensor shape=(N, M), float32
              The target depth distribution
    points: tensor shape=(N, D, 4), float32
            3D points randomly sampled for every ray
    camera_centers: tensor shape(N, 4), float32
                   The camera centers
    views: int
        Number of views used for the forward pass
    gamma: float or variable
           The occupancy prior
    bp_iterations: int
        The number of BP updates
    loss: str
          Identifier for the loss to be used
    """
    # Extract the number of the rays as well as the patch size from the images
    # N holds the total number of rays and D holds the different views
    N, D, H, W, C = K.int_shape(images[0])
    _, M, _ = K.int_shape(ray_voxel_indices)
    _, D1, D2, D3 = K.int_shape(voxel_grid)

    # Compute the features with the CNN
    features = [
        K.reshape(
            model(K.reshape(img, (N*D, H, W, C))),
            (N, D, -1))
        for img in images
    ]

    # The depth distribution (softmax output of the network) for each ray
    S = K.tf.zeros(shape=(N, D), dtype="float32")
    # Compute the similarities between CNN's features
    for n1 in range(views):
        for n2 in range(n1+1, views):
            S = S + compute_similarities(n1, n2, features)

    S = S / ((views * (views - 1))/2.0)
    S = K.softmax(S)

    # Map the depth distribution in corresponing depth distributions in voxel
    # space
    voxel_grid = K.permute_dimensions(voxel_grid, (1, 2, 3, 0))
    S_voxel_space = K.map_fn(
        lambda i: single_ray_depth_to_voxels_map_li(
            K.tf.gather_nd(
                voxel_grid,
                ray_voxel_indices[i, :ray_voxel_count[i]]
            ),
            K.transpose(points[i]),
            S[i],
            M
        ),
        K.tf.range(0, N, dtype="int32"),
        dtype="float32"
    )
    S_voxel_space = K.reshape(S_voxel_space, (N, M))

    # Do the MRF and the depth estimation based on the sum-product belief
    # propagation
    S_norm = clip_and_renorm(S_voxel_space, ray_voxel_count, 1e-5)
    ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon = \
        belief_propagation(
            S_norm,
            ray_voxel_indices,
            ray_voxel_count,
            np.array([D1, D2, D3]),
            gamma,
            bp_iterations
        )
    S_mrf = depth_estimate(
        S_norm,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon
    )

    # Start training
    # Specify the loss function to be used
    if loss == "emd":
        loss = K.mean(emd(S_target, S_mrf))
    elif loss == "squared_emd":
        loss = K.mean(squared_emd(S_target, S_mrf))
    elif loss == "expected_squared_error":
        loss = K.mean(expected_squared_error(S_target, S_mrf, voxel_grid,
                                             ray_voxel_indices, camera_centers))

    weights = model.trainable_weights
    if isinstance(gamma, K.tf.Variable):
        weights += [gamma]

    updates = model.optimizer.get_updates(loss, weights)

    return updates, loss


def build_end_to_end_training(
    model,
    n_rays,
    n_views,
    maximum_number_of_marched_voxels,
    depth_planes,
    grid_shape,
    patch_shape=(11, 11, 3),
    initial_gamma=0.031,
    gamma_range=(1e-3, 0.99),
    train_with_gamma=False,
    loss="squared_emd"
):
    """Build the function that will do the weights update of the given model

    Arguments:
    ---------
        model: Keras model
        n_rays: int, Number of rays used for training
        n_views: int, Number of different views
        maximum_number_of_marched_voxels: int, Maximum number of voxels that
                                          can intersect with a ray
        depth_planes: int, Number of discrete depth points along the viewing
                      ray
        grid_shape: tuple, Containing the number of voxels in each dimension
        patch_shape: tuple, Containing the shape of each patch
        initial_gamma: float, The initial value for the gamma prior
        gamma_range: tuple, The range between which gamma should move during
                     training
        train_with_gamma: bool, Variable indicating whether we will also learn
                          the gamma
    """
    # Make sure that the gamma_range has the proper ordering, namely
    # (min_gamma, max_gamma)
    assert gamma_range[0] < gamma_range[1]

    inputs = []

    # Specify the tensors that will be used as input
    # Tensor containing the image patches for $n_rays$ in $n_views$ in
    # $depth_planes$ depth_planes
    t_images = [
        K.placeholder(
            shape=(n_rays, depth_planes) + patch_shape,
            dtype="float32",
            name="images"
        )
        for i in range(n_views)
    ]
    inputs.extend(t_images)

    # Tensor containing the dimensions of the voxel grid
    t_voxel_grid = K.placeholder(
        shape=(3,) + grid_shape,
        dtype="float32",
        name="voxel_grid"
    )
    inputs.extend([t_voxel_grid])
    # Tensor containing the ray_marching indices for every ray
    t_ray_voxel_indices = K.placeholder(
        shape=(n_rays, maximum_number_of_marched_voxels, 3),
        dtype="int32",
        name="ray_voxel_indices"
    )
    inputs.extend([t_ray_voxel_indices])
    # Tensor containing the number of voxels that intersect with that ray
    t_ray_voxel_count = K.placeholder(
        shape=(n_rays, ),
        dtype="int32",
        name="ray_voxel_count"
    )
    inputs.extend([t_ray_voxel_count])
    # Tensor containing the ground-truth target distribution in voxel space
    t_S_target = K.placeholder(
        shape=(n_rays, maximum_number_of_marched_voxels),
        dtype="float32",
        name="s_target"
    )
    inputs.extend([t_S_target])

    # Tensor containing the 3D points for every ray
    t_points = K.placeholder(
        shape=(n_rays, depth_planes, 4),
        dtype="float32",
        name="points"
    )
    inputs.extend([t_points])

    t_camera_centers = K.placeholder(
        shape=(n_rays, 4),
        dtype="float32",
        name="camera_center"
    )
    inputs.extend([t_camera_centers])

    if train_with_gamma:
        gamma = K.variable(
            np.float32(initial_gamma),
            name="gamma",
            constraint=lambda x: K.clip(x, gamma_range[0], gamma_range[1])
        )
    else:
        gamma = K.constant(initial_gamma, dtype="float32")

    weights_updates, loss = forward_backward_pass(
        model,
        t_images,
        t_voxel_grid,
        t_ray_voxel_indices,
        t_ray_voxel_count,
        t_S_target,
        t_points,
        t_camera_centers,
        gamma=gamma,
        loss=loss
    )

    _forward_backward_f = K.function(inputs, [loss, gamma], updates=weights_updates)
    _forward_f = K.function(inputs, [loss])
    return _forward_backward_f, _forward_f
