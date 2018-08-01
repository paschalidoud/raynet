import numpy as np


def pixel_to_ray(y, x, axis_length, axis_order="columns"):
    """Given a pixel we want to associate it to a ray index.

    Arguments:
    ---------
        y: int
           The pixel value along the y-axis (namely the height)
        x: int
           The pixel value along the x-axis (namely the width)
        axis_length: int
                     The length of the major axis
        axis_order: string
                    String that indicates whether the produced index will be
                    used to access a row- or a column-major array

    Return:
    -------
        ray_index: int
                   The index of the ray
    """
    if axis_order is "columns":
        return x*axis_length + y
    elif axis_order is "rows":
        return y*axis_length + x
    else:
        raise ValueError("axis_order argument can be either columns or rows")


def point_from_depth(camera_center, direction, depth):
    """Given the camera_center (origin), the direction of the ray and the depth
    value find the 3D point on the ray that corresponds to this depth value.

    Arguments:
    ----------
        camera_center: array(shape=(3,1), dtype=np.float32)
                       Ray's origin
        direction: array(shape=(3,1), dtype=np.float32)
                   Ray's direction
        depth: int, The depth value
    Returns:
    -------
        point: array(shap=(3,1), dtype=np.float32)
               3D point
    """
    # || camera_center + a * t - camera_center || = depth =>
    # || a * t || = depth => t = depth / ||a||
    #
    # The equation of the ray is given from
    # p = camera_center + t * a =>
    # p = camera_center + a*depth / ||a||
    assert camera_center.shape == (3, 1)
    assert direction.shape == (3, 1)
    a_norm = direction / np.sqrt(np.sum(direction**2))
    p = a_norm * depth + camera_center

    return p


def voxel_to_world_coordinates(voxel_index, bbox, grid_shape):
    """Given a voxel, the bbox that encloses the scene and the shape of the
    grid, it computes the center of the voxel in world coordinates.

    Arguments:
    ----------
    voxel_index: array(shape=(3, ), dtype=int32)
                 The voxel indices in the voxel grid
    bbox: array(shape=(1, 6), dtype=np.float32)
          The min and max of the corners of the bbox that encloses the scene
    grid_shape: array(shape(3,), dtype=int32)
                The dimensions of the voxel grid used to discretize the scene
    """
    # Make sure that we have the appropriate inputs
    assert bbox.shape[0] == 1
    assert bbox.shape[1] == 6
    # Computhe the size of each voxel in the voxel grid in each dimension
    bin_size = (bbox[0, 3:] - bbox[0, :3]) / grid_shape
    # Transform the voxel index to actual size
    t = voxel_index*bin_size
    # Transform it to world coordinates
    t += bbox[0, :3]
    # Shift it to the center of the corresponding voxel
    t += bin_size / 2

    return t


def get_voxel_grid(bbox, grid_shape):
    """Given a bounding box and the dimensionality of a grid generate a grid of
    voxels and return their centers.

    Arguments:
    ----------
    bbox: array(shape=(1, 6), dtype=np.float32)
          The min and max of the corners of the bbox that encloses the scene
    grid_shape: array(shape(3,), dtype=int32)
                The dimensions of the voxel grid used to discretize the scene
    """
    # Make sure that we have the appropriate inputs
    assert bbox.shape[0] == 1
    assert bbox.shape[1] == 6
    xyz = [
        np.linspace(s, e, c, endpoint=False, dtype=np.float32)
        for s, e, c in
        zip(bbox[0, :3], bbox[0, 3:], grid_shape)
    ]
    bin_size = np.array([xyzi[1]-xyzi[0] for xyzi in xyz]).reshape(3, 1, 1, 1)
    return np.stack(np.meshgrid(*xyz, indexing="ij")) + bin_size/2


def point_to_voxel(p, bbox_origin, bin_size):
    """Given a 3D point, the bbox that encloses the scene and the bin_size in
    each dimension find the voxel that corresponds to this point

    Arguments:
    ----------
        p: array(shape=(3,1), dtype=np.float32), The 3D point
        bbox_origin: array(shape=(3,1), dtype=np.float32),
                     The origin of the bbox
        bin_size: array(shape=(3,1), dtype=np.float32)
                  The voxel dimensions in each axis
    """
    assert p.shape == (3, 1)
    assert bbox_origin.shape == (3, 1)
    assert bin_size.shape == (3, 1)

    v = (p - bbox_origin) / bin_size
    return np.floor(v).astype(np.int32)
