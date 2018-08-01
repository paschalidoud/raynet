from keras import backend as K


def emd(y_true, y_pred):
    """Earth movers distance loss"""
    return K.mean(K.abs(K.cumsum(y_true - y_pred, axis=-1)), axis=-1)


def squared_emd(y_true, y_pred):
    """Squared Earth movers distance loss"""
    return K.sum(K.pow(K.cumsum(y_true - y_pred, axis=-1), 2), axis=-1)


def expected_squared_error(
    y_true,
    y_pred,
    voxel_grid,
    ray_voxel_indices,
    camera_center
):
    # Calculate the voxel centers from based on the ray_voxel_indices
    voxel_centers = K.tf.gather_nd(
        voxel_grid,
        ray_voxel_indices
    )
    camera_center = K.reshape(camera_center[:, :3], (-1, 1, 3))
    dists = K.sqrt(
        K.sum(K.square(voxel_centers - camera_center), axis=-1)
    )

    depths_true = K.batch_dot(y_true, dists, axes=1)
    depths_pred = K.batch_dot(y_pred, dists, axes=1)

    #return K.square(depths_true - depths_pred)
    return K.sum(K.abs(depths_true - depths_pred), axis=-1)


def loss_factory(loss):
    if loss == "categorical_crossentropy":
        return "categorical_crossentropy"
    elif loss == "emd":
        return emd
    elif loss == "squared_emd":
        return squared_emd
    elif loss == "mse":
        return "mse"
    else:
        return emd
