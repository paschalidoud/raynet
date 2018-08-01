
import time
import unittest

import numpy as np

from raynet.ray_marching.ray_tracing import voxel_traversal
from raynet.ray_marching.ray_tracing_cuda import voxel_traversal as voxel_traversal_cuda


def backends():
    BACKENDS = []  # Holds a list of all available backends
    BACKENDS.append(voxel_traversal)
    BACKENDS.append(voxel_traversal_cuda)

    return BACKENDS


class TestRayTracing(unittest.TestCase):
    def test_2d(self):
        for voxel_traversal in backends():
            bbox = np.array([3, 3, 0, 6, 6, 1], dtype=np.float32)
            grid_shape = np.array([3, 3, 1], dtype=np.int32)
            voxels = np.empty((10, 3), dtype=np.int32)

            # Check an almost straight line
            voxels.fill(0)
            ray_start = np.array([3., 4.1, 0.5], dtype=np.float32)
            ray_end = np.array([6., 4.9, 0.5], dtype=np.float32)
            N = voxel_traversal(bbox, grid_shape, voxels, ray_start, ray_end)
            self.assertEqual(3, N)
            self.assertTrue(np.all(voxels[:3, 1] == 1))
            self.assertTrue(np.all(voxels[:3, 0] == np.arange(3)))

            # Check a downwards line
            voxels.fill(0)
            ray_start = np.array([4., 6., 0.5], dtype=np.float32)
            ray_end = np.array([6., 5., 0.5], dtype=np.float32)
            N = voxel_traversal(bbox, grid_shape, voxels, ray_start, ray_end)
            self.assertEqual(2, N)

            # Check a diagonal line
            voxels.fill(0)
            ray_start = np.array([3., 3., 0.5], dtype=np.float32)
            ray_end = np.array([6., 6., 0.5], dtype=np.float32)
            N = voxel_traversal(bbox, grid_shape, voxels, ray_start, ray_end)
            self.assertEqual(5, N)

            # Check a reversed diagonal line
            voxels.fill(0)
            ray_start = np.array([6., 6., 0.5], dtype=np.float32)
            ray_end = np.array([3., 3., 0.5], dtype=np.float32)
            N = voxel_traversal(bbox, grid_shape, voxels, ray_start, ray_end)
            self.assertEqual(5, N)

    def test_2d_2(self):
        for voxel_traversal in backends():
            bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
            grid_shape = np.array([6, 6, 1], dtype=np.int32)
            ray_voxels_indices = np.empty((10, 3), dtype=np.int32)
            ray_voxels_indices.fill(0)
            ray_start = np.array([0., 3.5, 0.5], dtype=np.float32)
            ray_end = np.array([6., 0.5, 0.5], dtype=np.float32)
            Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices, ray_start, ray_end)
            self.assertEqual(9, Nr)
            self.assertTrue(np.all(ray_voxels_indices == np.array([
                [0, 3, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
                [2, 1, 0],
                [3, 1, 0],
                [4, 1, 0],
                [4, 0, 0],
                [5, 0, 0],
                [0, 0, 0]
            ])))

    def test_speed(self):
        bbox = np.array([3, 3, 1, 6, 6, 2], dtype=np.float32)
        grid_shape = np.array([64, 64, 15], dtype=np.int32)
        voxels = np.zeros((256, 3), dtype=np.int32)
        ray_start = np.array([3., 3., 1.], dtype=np.float32)
        ray_end = np.array([6., 6., 2.], dtype=np.float32)

        start = time.time()
        for i in range(1000):
            N = voxel_traversal(bbox, grid_shape, voxels, ray_start, ray_end)
        end = time.time()
        self.assertGreater(1, end-start)

    def test_3d(self):
        for voxel_traversal in backends():
            bbox = np.array([-3., -3., -0.5, 3., 3., 2.], dtype=np.float32)
            grid_shape = np.array([32, 32, 10], dtype=np.int32)
            voxels = np.empty((100, 3), dtype=np.int32)

            voxels.fill(0)
            ray_start = np.array([-1.40056884, -1.34645462, 2.,], dtype=np.float32)
            ray_end = np.array([-2.30040455, 3., -0.37297964], dtype=np.float32)
            N = voxel_traversal(bbox, grid_shape, voxels, ray_start, ray_end)
            self.assertLess(N, 50)


if __name__ == "__main__":
    unittest.main()
