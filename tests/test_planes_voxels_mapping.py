from itertools import combinations
import os
import unittest

from keras import backend as K
import numpy as np

from raynet.tf_implementations.forward_backward_pass import\
    single_ray_depth_to_voxels_map_li
from raynet.planes_voxels_mapping.planes_voxels_mapping import\
    single_ray_depth_to_voxels_li, single_ray_depth_to_voxels_li_2


class PlanesVoxelsMappingTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PlanesVoxelsMappingTest, self).__init__(*args, **kwargs)

        self.functions = [
            self.tf_interp,
            self._np_interp_wrapper(single_ray_depth_to_voxels_li),
            self._np_interp_wrapper(single_ray_depth_to_voxels_li_2)
        ]
        self._tf_interp_sizes = tuple()

    def tf_interp(self, ray_voxels, points, s, M):
        C = ray_voxels.shape[0]
        D = points.shape[1]
        if self._tf_interp_sizes != (C, D, M):
            self._tf_interp_sizes = (C, D, M)

            t_ray_voxels = K.placeholder(
                shape=(C, 3),
                dtype="float32",
                name="ray_voxels"
            )
            t_points = K.placeholder(
                shape=(4, D),
                dtype="float32",
                name="points"
            )
            t_s = K.placeholder(
                shape=(D,),
                dtype="float32",
                name="Sr"
            )

            self._tf_interp_inner = K.function(
                [t_ray_voxels, t_points, t_s],
                [single_ray_depth_to_voxels_map_li(
                    t_ray_voxels, t_points, t_s, M
                )]
            )
        return self._tf_interp_inner([ray_voxels, points, s])[0]

    def _np_interp_wrapper(self, f):
        def inner(ray_voxels, points, s, M):
            s_new = f(ray_voxels.T, points[:-1], s)
            return np.hstack([s_new, np.zeros(M-s_new.size)])
        return inner

    def test_random_inputs(self):
        C, D, M = 10, 5, 20
        for i in range(10):
            voxels = np.random.rand(C, 3)
            points_start = np.random.rand(4, 1) - 1
            points_start[-1:] = 1
            points_end = np.random.rand(4, 1) + 1
            points_end[-1:] = 1
            ray = points_end - points_start
            points = points_start + np.linspace(0, 1, D)*ray
            s = np.random.rand(D)
            s /= s.sum()

            outputs = [f(voxels, points, s, M) for f in self.functions]
            for o in outputs:
                self.assertEqual(M, o.size)
            for o1, o2 in combinations(outputs, 2):
                self.assertTrue(np.allclose(o1, o2))


if __name__ == "__main__":
    unittest.main()
