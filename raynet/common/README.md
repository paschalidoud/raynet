# Common module

The *common* module contains various general purpose structures as well as
functions that are used through out the entire codebase.

Dependency
----------
1. NumPy
2. SciPy
3. ElementTree XML API

## Create camera instance `camera.py`
----------------------------------

The *Camera* class wraps a simple finite pinhole camera defined by the
intrinsic camera matrix K, the rotation matrix R and the translation matrix t.

## Create image instance `image.py`
--------------------------------
The *Image* class contains the image buffer and the camera that produced it.
