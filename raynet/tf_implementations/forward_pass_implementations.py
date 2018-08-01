from itertools import product

import numpy as np
from keras import backend as K
K.set_learning_phase(0)

from sampling_schemes import sample_points_in_range, sample_points_in_bbox,\
    compute_rays, compute_rays_from_pixels


def swap_xy(tensor, axis=-1):
    """Swap the values in the axis `axis`. It can be used to to swap rgb with
    bgr or xy to yx etc.

    This method swaps the first and last values of axis.

    Arguments
    ---------
    tensor: tensor of variable shape
    axis: int, the axis whose values to swap
    """
    # Make sure that the axis is positive in [0, rank] (negative could probably
    # work but just to be sure)
    shape = K.int_shape(tensor)
    rank = len(shape)
    axis = (rank + axis) % rank

    if shape[axis] < 2:
        raise ValueError(("The tensor has only one value at the axis "
                          "we are supposed to swap"))

    # Create indices for indexing in the tensor
    idxs_axis = range(shape[axis])
    idxs_axis[0], idxs_axis[-1] = idxs_axis[-1], idxs_axis[0]
    idxs_all = [
        [slice(None) for r in range(rank)]
        for a in range(shape[axis])
    ]
    for idxs, idx_axis in zip(idxs_all, idxs_axis):
        idxs[axis] = idx_axis

    # Extract and restack
    values = [tensor[tuple(idx)] for idx in idxs_all]
    return K.stack(values, axis=axis)


def pixel_to_feature(x, padding, h, w):
    """Associate a pixel in a non-zeropadded image with each corresponding
    pixel in the zeropadded image
    """
    x = x + padding - (padding - 1) / 2

    # Make sure that the pixels are inside the image boundaries
    mask_h = K.tf.logical_and(x[0] >= 0, x[0] < h)
    mask_w = K.tf.logical_and(x[1] >= 0, x[1] < w)
    mask_hw = K.tf.logical_and(mask_h, mask_w)

    # Combine the masks
    mask = K.stack([mask_hw, mask_hw])
    mask = K.cast(mask, "int32")

    return x * mask


def compute_similarities(
    n1,
    n2,
    features,
    neighbor_pixels,
    N,
    padding,
    depth_planes
):
    """
    Arguments
    ---------
    n1: int
        Index of the first image in the pair of images
    n2: int
        Index of the second image in the pair of images
    features: tensor 5xF
        Features for all images
    neighbor_pixels: list of tensors of size 3xNxdepth_planes
        The pixels of the projected 3D points in all image views
    N: int
       Number of pixels in each image
    """
    # Compute the features from the images with index n1 and n2
    F1 = features[n1]
    F2 = features[n2]

    h = K.int_shape(F1)[0]
    w = K.int_shape(F1)[1]
    f = K.shape(F1)[-1]

    # Get the pixels that correspond to the each image from the pair
    pixels1 = pixel_to_feature(neighbor_pixels[n1], padding, h, w)
    pixels2 = pixel_to_feature(neighbor_pixels[n2], padding, h, w)

    # Create list to keep all the similarities per depth plane
    sims = []
    for d in range(depth_planes):
        # Get the features that correspond to the pixels from the d-th depth
        # plane from the first image in the image pair
        X1 = K.tf.gather_nd(F1, K.transpose(pixels1[:, :, d]))

        # Get the features that correspond to the pixels from the d-th depth
        # plane from the second image in the image pair
        X2 = K.tf.gather_nd(F2, K.transpose(pixels2[:, :, d]))

        # Compute the dot product as the similarity measure between the
        # features from the two pairs for that corresponding depth plane
        sim_per_depth = K.reshape(
            K.batch_dot(
                K.reshape(X1, (N, 1, f)),
                K.reshape(X2, (N, f, 1))
            ),
            (-1,)
        )
        sims.append(sim_per_depth)

    sim = K.transpose(K.stack(sims))
    return sim


def multi_view_cnn_forward_pass(
    model,
    images,
    P,
    points,
    padding=11,
    views=5
):
    """
    Arguments
    ---------
    model: Keras CNN model
    images: list[tensors, shape(H+2*padding, W+2*padding), dtype=float32]
            A list of tensors containing the images used for the forward pass
    P: list[tensors, shape=(3, 4), dtype=float32]
        A list of tensors containing the projection matrices of all images
        (including the reference image)
    points: tensor 4xNx32, float32
            The uniformly sampled points using a sampling scheme
    padding: int
        Size of padding used to zero-padd the images before feeding them to the
        network
    views: int
        Number of views used for the forward pass
    """
    # Extract the width and height of the images
    H, W, C = K.int_shape(images[0])
    H -= 2*padding
    W -= 2*padding

    # Extract number of rays and depth planes
    _, _, D = K.int_shape(points)

    # For simplicity, the P also contains the projection matrix of the
    # reference image, thus we do an extra useless projection computation also
    # in the reference image
    neighbor_pixels = [K.dot(p, K.reshape(points, (4, -1))) for p in P]
    neighbor_pixels = [npixel / npixel[-1:] for npixel in neighbor_pixels]
    neighbor_pixels = [
        K.reshape(npixel, (3, -1, D)) for npixel in neighbor_pixels
    ]
    neighbor_pixels = [
        K.cast(K.round(npixel), "int32") for npixel in neighbor_pixels
    ]
    neighbor_pixels = [
        swap_xy(npixel[:2], axis=0) for npixel in neighbor_pixels
    ]

    # Compute the features with the CNN
    features = [
        K.reshape(
            model(K.reshape(img, (1, H+2*padding, W+2*padding, C))),
            (H+padding+1, W+padding+1, -1))
        for img in images
    ]

    # The depth distribution (softmax output of the network) for each pixel in
    # the reference image
    s = K.zeros(shape=(H*W, D), dtype="float32")

    # Compute the similarity scores for each image pair
    for n1 in range(views):
        for n2 in range(n1+1, views):
            s = s + compute_similarities(n1, n2, features, neighbor_pixels, H*W, padding, D)

    s /= (views * (views - 1))/2.0
    s = K.softmax(s)
    return s


def multi_view_cnn_forward_pass_with_features(
    model,
    features,
    P,
    points,
    padding=11,
    views=5
):
    """
    Arguments
    ---------
    model: Keras CNN model
    features: list[tensors, shape(H+padding+1, W+padding+1), dtype=float32]
            A list of tensors containing the features from the forward pass
    P: list[tensors, shape=(4, 3), dtype=float32]
        A list of tensors containing the projection matrices of all images
        (including the reference image)
    points: tensor 4xNx32, float32
            The uniformly sampled points using a sampling scheme
    padding: int
        Size of padding used to zero-padd the images before feeding them to the
        network
    views: int
        Number of views used for the forward pass
    """
    # Extract number of rays and depth planes
    N = K.shape(points)[1]
    _, _, D = K.int_shape(points)

    # For simplicity, the P also contains the projection matrix of the
    # reference image, thus we do an extra useless projection computation also
    # in the reference image
    neighbor_pixels = [K.dot(p, K.reshape(points, (4, -1))) for p in P]
    neighbor_pixels = [npixel / npixel[-1:] for npixel in neighbor_pixels]
    neighbor_pixels = [
        K.reshape(npixel, (3, -1, D)) for npixel in neighbor_pixels
    ]
    neighbor_pixels = [
        K.cast(K.round(npixel), "int32") for npixel in neighbor_pixels
    ]
    neighbor_pixels = [
        swap_xy(npixel[:2], axis=0) for npixel in neighbor_pixels
    ]

    # The depth distribution (softmax output of the network) for each pixel in
    # the reference image
    s = K.tf.zeros(shape=(N, D), dtype="float32")

    # Compute the similarity scores for each image pair
    for n1 in range(views):
        for n2 in range(n1+1, views):
            s = s + compute_similarities(n1, n2, features, neighbor_pixels, N, padding, D)

    s /= (views * (views - 1))/2.0
    s = K.softmax(s)

    return s


def full_multi_view_cnn_forward_pass(
    model,
    images,
    P,
    P_inv,
    camera_center,
    bbox,
    padding=11,
    depth_planes=32,
    views=5,
    depth_range_min=450,
    depth_range_max=1000,
    sampling_scheme="sample_points_in_bbox"
):
    # Extract the width and height of the images
    H, W, C = K.int_shape(images[0])
    H -= 2*padding
    W -= 2*padding

    # Backproject the pixels into rays
    pixels, rays, directions = compute_rays(H, W, P_inv, camera_center)

    # Based on a sampling scheme sample the points
    if sampling_scheme == "sample_points_in_range":
        points = sample_points_in_range(
            depth_range_min,
            depth_range_max,
            camera_center,
            directions,
            depth_planes
        )
    elif sampling_scheme == "sample_points_in_bbox":
        points = sample_points_in_bbox(
            bbox,
            camera_center,
            directions,
            depth_planes
        )

    # Estimate the per-pixel depth distributions
    s = multi_view_cnn_forward_pass(
        model,
        images,
        P,
        P_inv,
        points,
        padding=padding,
        depth_planes=depth_planes,
        views=views
    )
    return s, points


def full_multi_view_cnn_forward_pass_with_features(
    model,
    features,
    P,
    P_inv,
    camera_center,
    bbox,
    pixels,
    padding=11,
    depth_planes=32,
    views=5,
    depth_range_min=450,
    depth_range_max=1000,
    sampling_scheme="sample_points_in_bbox"
):
    # Backproject the pixels into rays
    rays, directions = compute_rays_from_pixels(pixels, P_inv, camera_center)

    # Based on a sampling scheme sample the points
    if sampling_scheme == "sample_points_in_range":
        points = sample_points_in_range(
            depth_range_min,
            depth_range_max,
            camera_center,
            directions,
            depth_planes
        )
    elif sampling_scheme == "sample_points_in_bbox":
        points = sample_points_in_bbox(
            bbox,
            camera_center,
            directions,
            depth_planes
        )

    # Estimate the per-pixel depth distributions
    s = multi_view_cnn_forward_pass_with_features(
        model,
        features,
        P,
        points,
        padding=padding,
        views=views
    )
    return s, points

def build_multi_view_cnn_forward_pass(
    model,
    H,
    W,
    views,
    depth_planes,
    padding
):
    """
    Arguments
    ---------
        model: the Keras model to be used for feature extraction
        H: int, the height of the image
        W: int, the width of the image
        views: int, the number of views (neighbors+1)
        depth_planes: int, the discretization steps along the viewing ray
        padding:int, the dimension of the zero-padding around the image
    """
    inputs = []

    # Specify all the tensors we will need
    t_images = [
        K.placeholder(
            shape=(H+2*padding, W+2*padding, 3),
            dtype="float32",
            name="images"
        )
        for i in range(views)
    ]
    inputs.extend(t_images)
    t_P = [
        K.placeholder(shape=(3, 4), dtype="float32", name="proj_matrix")
        for i in range(views)
    ]
    inputs.extend(t_P)
    t_points = K.placeholder(
        shape=(4, H*W, depth_planes),
        dtype="float32",
        name="points"
    )
    inputs.extend([t_points])

    S = multi_view_cnn_forward_pass(
        model,
        t_images,
        t_P,
        t_points,
        padding,
        views
    )
    f = K.function(inputs, [S])
    return f


def build_multi_view_cnn_forward_pass_with_features(
    model,
    H,
    W,
    views,
    depth_planes,
    padding
):
    """
    Arguments
    ---------
        model: the Keras model to be used for feature extraction
        H: int, the height of the image
        W: int, the width of the image
        views: int, the number of views (neighbors+1)
        depth_planes: int, the discretization steps along the viewing ray
        padding: int, the dimension of the zero-padding around the image
    """
    inputs = []

    # Specify all the tensors we will need
    t_features = [
        K.placeholder(
            shape=(H+padding+1, W+padding+1, None),
            dtype="float32",
            name="features"
        )
        for i in range(views)
    ]
    inputs.extend(t_features)
    t_P = [
        K.placeholder(shape=(3, 4), dtype="float32", name="proj_matrix")
        for i in range(views)
    ]
    inputs.extend(t_P)
    t_points = K.placeholder(
        shape=(4, None, depth_planes),
        dtype="float32",
        name="points"
    )
    inputs.extend([t_points])

    S = multi_view_cnn_forward_pass_with_features(
        model,
        t_features,
        t_P,
        t_points,
        padding,
        views
    )
    f = K.function(inputs, [S])
    return f

def build_full_multi_view_cnn_forward_pass(
    model,
    H,
    W,
    views,
    depth_planes,
    padding,
    depth_range_min,
    depth_range_max,
    sampling_scheme
):
    """
    Arguments
    ---------
        model: the Keras model to be used for feature extraction
        H: int, the height of the image
        W: int, the width of the image
        views: int, the number of views (neighbors+1)
        depth_planes: int, the discretization steps along the viewing ray
        padding: int, the dimension of the zero-padding around the image
        depth_range_min: float, the minimum depth value for the sampling in range scheme
        depth_range_max: float, the maximum depth value for the sampling in range scheme
        sampling_scheme: string, Identifier for the sampling scheme to be used
    """
    inputs = []

    # Specify all the tensors we will need
    t_images = [
        K.placeholder(
            shape=(H+2*padding, W+2*padding, 3),
            dtype="float32",
            name="images"
        )
        for i in range(views)
    ]
    inputs.extend(t_images)
    t_P = [
        K.placeholder(shape=(3, 4), dtype="float32", name="proj_matrix")
        for i in range(views)
    ]
    inputs.extend(t_P)
    t_P_pinv = K.placeholder(
        shape=(4, 3),
        dtype="float32",
        name="inv_proj_matrix"
    )
    inputs.extend([t_P_pinv])

    t_camera_center = K.placeholder(
        shape=(4, 1),
        dtype="float32",
        name="camera_center"
    )
    inputs.extend([t_camera_center])
    t_bbox = K.placeholder(shape=(6, 1), dtype="float32", name="bbox")
    inputs.extend([t_bbox])

    S, points = full_multi_view_cnn_forward_pass(
        model,
        t_images,
        t_P,
        t_P_pinv,
        t_camera_center,
        t_bbox,
        padding,
        depth_planes,
        views,
        depth_range_min,
        depth_range_max,
        sampling_scheme
    )

    f = K.function(inputs, [S, points])
    return f


def build_full_multi_view_cnn_forward_pass_with_features(
    model,
    H,
    W,
    views,
    depth_planes,
    padding,
    depth_range_min,
    depth_range_max,
    sampling_scheme
):
    """
    Arguments
    ---------
        model: the Keras model to be used for feature extraction
        H: int, the height of the image
        W: int, the width of the image
        views: int, the number of views (neighbors+1)
        depth_planes: int, the discretization steps along the viewing ray
        padding: int, the dimension of the zero-padding around the image
        depth_range_min: float, the minimum depth value for the sampling in range scheme
        depth_range_max: float, the maximum depth value for the sampling in range scheme
        sampling_scheme: string, Identifier for the sampling scheme to be used
    """
    inputs = []

    # Specify all the tensors we will need
    t_features = [
        K.placeholder(
            shape=(H+padding+1, W+padding+1, None),
            dtype="float32",
            name="features"
        )
        for i in range(views)
    ]
    inputs.extend(t_features)

    t_P = [
        K.placeholder(shape=(3, 4), dtype="float32", name="proj_matrix")
        for i in range(views)
    ]
    inputs.extend(t_P)
    t_P_pinv = K.placeholder(
        shape=(4, 3),
        dtype="float32",
        name="inv_proj_matrix"
    )
    inputs.extend([t_P_pinv])

    t_camera_center = K.placeholder(
        shape=(4, 1),
        dtype="float32",
        name="camera_center"
    )
    inputs.extend([t_camera_center])
    t_bbox = K.placeholder(shape=(6, 1), dtype="float32", name="bbox")
    inputs.extend([t_bbox])

    t_pixels = K.placeholder(
        shape=(3, None),
        dtype="float32",
        name="pixels"
    )
    inputs.extend([t_pixels])

    S, points = full_multi_view_cnn_forward_pass_with_features(
        model,
        t_features,
        t_P,
        t_P_pinv,
        t_camera_center,
        t_bbox,
        t_pixels,
        padding,
        depth_planes,
        views,
        depth_range_min,
        depth_range_max,
        sampling_scheme
    )
    f = K.function(inputs, [S, points])
    return f
