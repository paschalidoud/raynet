import numpy as np


class Camera(object):
    """Camera is a simple finite pinhole camera defined by the matrices K, R
    and t.

    see "Multiple View Geometry in Computer Vision" by R. Hartley and A.
    Zisserman for notation.

    Parameters
    ----------
        K: The 3x3 intrinsic camera parameters
        R: The 3x3 rotation matrix from world to camera coordinates
        t: The 3x1 translation vector for the camera center in camera coordinates
           (so that the camera center is the origin in the camera coordinates)
    """
    def __init__(self, K, R, t):
        # Make sure the input data have the right shape
        assert K.shape == (3, 3)
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)

        self._K = K
        self._R = R
        self._t = t
        self._P = None
        self._P_pinv = None
        self._center = None

    @property
    def K(self):
        return self._K

    @property
    def R(self):
        return self._R

    @property
    def t(self):
        return self._t

    @property
    def center(self):
        # Compute the center of the camera in homogenous coordinates and return
        # it as a 4x1 vector
        if self._center is None:
            self._center = np.vstack(
                [(-np.linalg.inv(self.R)).dot(self.t), [1]]
            ).astype(np.float32)
            assert self._center.shape == (4, 1)
        return self._center

    @property
    def P(self):
        # Compute and return a 3x4 projection matrix
        if self._P is None:
            self._P = self._K.dot(np.hstack([self._R, self._t]))
        return self._P

    @property
    def P_pinv(self):
        if self._P_pinv is None:
            self._P_pinv = np.linalg.pinv(self.P)
        return self._P_pinv
