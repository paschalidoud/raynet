import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import unittest

from raynet.common.generation_parameters import GenerationParameters

from raynet.mrf.bp_inference import get_bp_backend
from raynet.mrf.mrf_np import compute_occupancy_probabilities
from raynet.ray_marching.ray_tracing import voxel_traversal


def get_generation_params(grid_shape, max_number_of_marched_voxels):
    return GenerationParameters(
        grid_shape=grid_shape,
        max_number_of_marched_voxels=max_number_of_marched_voxels
    )


def append_to_backends(grid_shape, N, M, batch_size=1):
    BACKENDS = []  # Holds a list of all available backends
    common_params = dict(
        generation_params=get_generation_params(grid_shape, M),
        bp_iterations=3
    )
    BACKENDS.append(get_bp_backend("numpy", **common_params))
    BACKENDS.append(get_bp_backend("tf", N=N, **common_params))
    BACKENDS.append(get_bp_backend("cuda", batch_size=batch_size, **common_params))

    return BACKENDS


class TestMRF(unittest.TestCase):
    def test_2d_single_ray(self):
        # Define a 2d grid of size 6x6
        bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
        grid_shape = np.array([6, 6, 1], dtype=np.int32)
        ray_voxels_indices = np.empty((10, 3), dtype=np.int32)
        ray_voxels_indices.fill(0)

        ray_start = np.array([0., 3.5, 0.5], dtype=np.float32)
        ray_end = np.array([6., 0.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(
            bbox,
            grid_shape,
            ray_voxels_indices,
            ray_start,
            ray_end
        )

        # We assume that the (2, 2) voxel is occupied, thus we will give a
        # higher probability
        S = np.array(
            [[0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.0]],
            dtype=np.float32
        )

        # Test for the different backends
        BACKENDS = append_to_backends(grid_shape, 1, 10)
        for i, bp in enumerate(BACKENDS):
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_messages_pon = bp.update_bp_messages(
                S,
                ray_voxels_indices.reshape(1, 10, 3),
                np.ones(1, dtype=np.int32) * Nr,
                np.random.random((1, 10)).astype(np.float32)
            )
            occupancy_probabilities = compute_occupancy_probabilities(
                ray_to_occupancy_accumulated_pon
            )

            # The (2, 2) voxel should have the higher probability
            max_idx = np.where(occupancy_probabilities == occupancy_probabilities.max())
            self.assertEqual(max_idx[0][0], 2)
            self.assertEqual(max_idx[1][0], 2)

            fig = plt.figure()
            plt.imshow(occupancy_probabilities[:, :, 0].T)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig("/tmp/bp%d_occupancy_probability_1ray.png" % (i,))
            plt.close()

    def test_2d_single_2rays(self):
        # Define a 2d grid of size 6x6
        bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
        grid_shape = np.array([6, 6, 1], dtype=np.int32)
        ray_voxels_indices = np.empty((2, 10, 3), dtype=np.int32)
        ray_voxels_indices.fill(0)
        S_total = np.zeros((2, 10), dtype=np.float32)

        ray_start = np.array([0., 3.5, 0.5], dtype=np.float32)
        ray_end = np.array([6., 0.5, 0.5], dtype=np.float32)
        ray_voxel_count = []
        Nr = voxel_traversal(
            bbox,
            grid_shape,
            ray_voxels_indices[0, :, :],
            ray_start,
            ray_end
        )
        ray_voxel_count.append(Nr)

        # We assume that the (2, 2) voxel is occupied, thus we will give a
        # higher probability
        S1 = np.array(
            [[0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.0]],
            dtype=np.float32
        )
        S_total[0, :] = S1

        ray_start = np.array([6., 5.5, 0.5], dtype=np.float32)
        ray_end = np.array([0.0, 2.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices[1, :, :], ray_start, ray_end)
        ray_voxel_count.append(Nr)

        # We assume that the (4, 3) voxel is occupied, thus we will give a
        # higher probability
        S2 = np.array(
            [[0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.0]],
            dtype=np.float32
        )
        S_total[1, :] = S2

        # Test for the different backends
        BACKENDS = append_to_backends(grid_shape, 2, 10, batch_size=2)
        for i, bp in enumerate(BACKENDS):
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_messages_pon = bp.update_bp_messages(
                S_total,
                ray_voxels_indices,
                np.stack(ray_voxel_count).astype(np.int32),
                np.random.random((2, 10)).astype(np.float32)
            )
            occupancy_probabilities = compute_occupancy_probabilities(
                ray_to_occupancy_accumulated_pon
            ).T

            # Find which one of the two occupied voxels has the larger probability
            max_prob = [occupancy_probabilities[0, 4, 3]
                if occupancy_probabilities[0, 4, 3] > occupancy_probabilities[0, 2, 2] else occupancy_probabilities[0, 2, 2]][0]
            for i in range(6):
                for j in range(6):
                    self.assertGreaterEqual(max_prob, occupancy_probabilities[0, i, j])

            fig = plt.figure()
            plt.imshow(occupancy_probabilities[:, :, 0].T)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig("/tmp/bp%d_occupancy_probability_2rays.png" % (i,))
            plt.close()

    def test_2d_single_2rays_2example(self):
        # Define a 2d grid of size 6x6
        bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
        grid_shape = np.array([6, 6, 1], dtype=np.int32)
        ray_voxels_indices = np.empty((2, 11, 3), dtype=np.int32)
        ray_voxels_indices.fill(0)
        S_total = np.zeros((2, 11), dtype=np.float32)

        ray_start = np.array([0., 3.5, 0.5], dtype=np.float32)
        ray_end = np.array([6., 0.5, 0.5], dtype=np.float32)
        ray_voxel_count = []
        Nr = voxel_traversal(
            bbox,
            grid_shape,
            ray_voxels_indices[0, :, :],
            ray_start,
            ray_end
        )
        ray_voxel_count.append(Nr)

        # We assume that the (2, 2) voxel is occupied, thus we will give a
        # higher probability
        S1 = np.array(
            [[0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.0, 0.0]],
            dtype=np.float32
        )
        S_total[0, :] = S1

        ray_start = np.array([6., 5.5, 0.5], dtype=np.float32)
        ray_end = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(
            bbox,
            grid_shape,
            ray_voxels_indices[1, :, :],
            ray_start,
            ray_end
        )
        ray_voxel_count.append(Nr)

        # We assume that the (2, 2) voxel is occupied, thus we will give a
        # higher probability
        S2 = np.array(
            [[0.07, 0.07, 0.185, 0.07, 0.07, 0.07, 0.185, 0.07, 0.07, 0.07, 0.07]],
            dtype=np.float32
        )
        S_total[1, :] = S2

        # Test for the different backends
        BACKENDS = append_to_backends(grid_shape, 2, 11, batch_size=2)
        for i, bp in enumerate(BACKENDS):
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_messages_pon = bp.update_bp_messages(
                S_total,
                ray_voxels_indices,
                np.stack(ray_voxel_count).astype(np.int32),
                np.random.random((2, 11)).astype(np.float32)
            )
            occupancy_probabilities = compute_occupancy_probabilities(
                ray_to_occupancy_accumulated_pon
            ).T

            for i in range(6):
                for j in range(6):
                    self.assertGreaterEqual(occupancy_probabilities[0, 2, 2], occupancy_probabilities[0, i, j])

            fig = plt.figure()
            plt.imshow(occupancy_probabilities[:, :, 0].T)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig("/tmp/bp%d_occupancy_probability_2rays_2.png" % (i,))
            plt.close()

    def test_2d_single_3rays(self):
        # Define a 2d grid of size 6x6
        bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
        grid_shape = np.array([6, 6, 1], dtype=np.int32)
        ray_voxels_indices = np.empty((3, 11, 3), dtype=np.int32)
        ray_voxels_indices.fill(0)
        S_total = np.zeros((3, 11), dtype=np.float32)
        ray_voxel_count = []

        # Ray 1
        ray_start = np.array([0., 3.5, 0.5], dtype=np.float32)
        ray_end = np.array([6., 0.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices[0, :, :], ray_start, ray_end)
        ray_voxel_count.append(Nr)
        S1 = np.array(
            [[0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.0, 0.0]],
            dtype=np.float32
        )
        S_total[0, :] = S1

        # Ray 2
        ray_start = np.array([0.0, 2.5, 0.5], dtype=np.float32)
        ray_end = np.array([6.0, 2.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices[1, :, :], ray_start, ray_end)
        ray_voxel_count.append(Nr)
        S2 = np.array(
            [[0.45, 0.0875, 0.2, 0.0875, 0.0875, 0.0875, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32
        )
        S_total[1, :] = S2

        # Ray 3
        ray_start = np.array([6., 5.5, 0.5], dtype=np.float32)
        ray_end = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices[2, :, :], ray_start, ray_end)
        ray_voxel_count.append(Nr)
        S3 = np.array(
            [[0.07, 0.07, 0.185, 0.07, 0.07, 0.07, 0.185, 0.07, 0.07, 0.07, 0.07]],
            dtype=np.float32
        )
        S_total[2, :] = S3

        # Test for the different backends
        BACKENDS = append_to_backends(grid_shape, 3, 11, batch_size=3)
        for i, bp in enumerate(BACKENDS):
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_messages_pon = bp.update_bp_messages(
                S_total,
                ray_voxels_indices,
                np.stack(ray_voxel_count).astype(np.int32),
                np.random.random((3, 11)).astype(np.float32)
            )
            occupancy_probabilities = compute_occupancy_probabilities(
                ray_to_occupancy_accumulated_pon
            ).T

            # Make sure that the voxel, for which all rays vote there is a higher
            # probability
            for i in range(6):
                for j in range(6):
                    self.assertGreaterEqual(
                        occupancy_probabilities[0, 2, 2],
                        occupancy_probabilities[0, i, j]
                    )

            for i in range(6):
                for j in range(6):
                    if i == 2 and j == 2:
                        continue
                    self.assertGreaterEqual(
                        occupancy_probabilities[0, 2, 0],
                        occupancy_probabilities[0, i, j]
                    )

            for i in range(6):
                for j in range(6):
                    if i == 2 and (j == 2 or j==0):
                        continue
                    self.assertGreaterEqual(
                        occupancy_probabilities[0, 4, 4],
                        occupancy_probabilities[0, i, j]
                    )

            fig = plt.figure()
            plt.imshow(occupancy_probabilities[:, :, 0].T)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig("/tmp/bp%d_occupancy_probability_3rays.png" % (i,))
            plt.close()

    def test_2d_conflict(self):
        # Define a 2d grid of size 6x6
        bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
        grid_shape = np.array([6, 6, 1], dtype=np.int32)
        ray_voxels_indices = np.empty((2, 11, 3), dtype=np.int32)
        ray_voxels_indices.fill(0)
        S_total = np.zeros((2, 11), dtype=np.float32)
        ray_voxel_count = []

        # Ray 1
        ray_start = np.array([0.0, 3.5, 0.5], dtype=np.float32)
        ray_end = np.array([6.0, 0.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices[0, :, :], ray_start, ray_end)
        ray_voxel_count.append(Nr)
        S_total[0, 2] = 0.5
        S_total[0, 6] = 0.5

        # Ray 2
        ray_start = np.array([0.0, 1.5, 0.5], dtype=np.float32)
        ray_end = np.array([4.5, 6.0, 0.5], dtype=np.float32)
        Nr = voxel_traversal(bbox, grid_shape, ray_voxels_indices[1, :, :], ray_start, ray_end)
        ray_voxel_count.append(Nr)
        S_total[1, 4] = 1.0

        # Test for the different backends
        BACKENDS = append_to_backends(grid_shape, 2, 11, batch_size=2)
        for i, bp in enumerate(BACKENDS):
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_messages_pon = bp.update_bp_messages(
                S_total,
                ray_voxels_indices,
                np.stack(ray_voxel_count).astype(np.int32),
                np.random.random((2, 11)).astype(np.float32)
            )
            occupancy_probabilities = compute_occupancy_probabilities(
                ray_to_occupancy_accumulated_pon
            ).T
            self.assertTrue(occupancy_probabilities[0, 0, 2] < 0.1)

            fig = plt.figure()
            plt.imshow(occupancy_probabilities[:, :, 0].T)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig("/tmp/bp%d_occupancy_probability_conflict.png" % (i,))
            plt.close()

    def test_depth_distribution(self):
        # Define a 2d grid of size 6x6
        bbox = np.array([0, 0, 0, 6, 6, 1], dtype=np.float32)
        grid_shape = np.array([6, 6, 1], dtype=np.int32)
        ray_voxels_indices = np.empty((2, 11, 3), dtype=np.int32)
        ray_voxels_indices.fill(0)
        S_total = np.zeros((2, 11), dtype=np.float32)
        ray_voxel_count = []

        # Ray 1
        ray_start = np.array([0.0, 3.5, 0.5], dtype=np.float32)
        ray_end = np.array([6.0, 0.5, 0.5], dtype=np.float32)
        Nr = voxel_traversal(
            bbox,
            grid_shape,
            ray_voxels_indices[0, :, :],
            ray_start,
            ray_end
        )
        ray_voxel_count.append(Nr)
        S_total[0, 2] = 0.5
        S_total[0, 6] = 0.5

        # Ray 2
        ray_start = np.array([0.0, 1.5, 0.5], dtype=np.float32)
        ray_end = np.array([4.5, 6.0, 0.5], dtype=np.float32)
        Nr = voxel_traversal(
            bbox,
            grid_shape,
            ray_voxels_indices[1, :, :],
            ray_start, ray_end
        )
        ray_voxel_count.append(Nr)
        S_total[1, 4] = 1.0

        # Test for the different backends
        BACKENDS = append_to_backends(grid_shape, 2, 11)
        for i, bp in enumerate(BACKENDS):
            ray_to_occupancy_accumulated_pon, ray_to_occupancy_messages_pon =\
                bp.update_bp_messages(
                    S_total,
                    ray_voxels_indices,
                    np.stack(ray_voxel_count).astype(np.int32),
                    np.random.random((2, 11)).astype(np.float32)
                )
            S_new = bp.estimate_depth_probabilities_from_messages(
                S_total,
                ray_voxels_indices,
                np.stack(ray_voxel_count).astype(np.int32),
                ray_to_occupancy_accumulated_pon,
                ray_to_occupancy_messages_pon,
                np.zeros_like(S_total)
            )
            if isinstance(S_new, list):
                S_new = S_new[0]

            self.assertGreater(0.5, S_new[0, 2])
            self.assertLess(0.9, S_new[0, 6])
            self.assertLess(0.9, S_new[1, 4])

    
if __name__ == "__main__":
    unittest.main()
