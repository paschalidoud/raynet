import os
import numpy as np

from .mrf_cuda import belief_propagation as cuda_bp
from .mrf_cuda import compute_depth_distribution as cuda_compute_depth_distribution
from .mrf_np import belief_propagation as np_bp
from .mrf_np import compute_depth_distribution as np_compute_depth_distribution
from .mrf_tf import build_belief_propagation, build_depth_estimate
from .mrf_utils import export_depth_map_from_voxel_indices
from ..utils.generic_utils import voxel_to_world_coordinates
from ..utils.geometry import distance


class BPInference(object):
    def __init__(self, generation_params, bp_iterations=3, gamma_prior=0.05):
        self._generation_params = generation_params
        # Number of bp updates
        self.bp_iterations = bp_iterations
        # Initial probability for a voxel being occupied
        self.gamma_prior = gamma_prior

    @staticmethod
    def reconstruct_scene(
        scene,
        S,
        ray_voxel_indices,
        rays_idxs,
        grid_shape,
        start=0,
        end=10,
        step=1,
        output_directory="/tmp",
        filter_out=False
    ):
        """Reconstruct a 3D scene given a set of depth estimations in a voxel space
        """
        # Extract the image's shape
        H, W = scene.image_shape
        N = H*W
        image_cnt = 0
        s = 0
        for img_idx in range(start, end, step):
            if filter_out:
                # Get the gt depth map for the current scene
                G = scene.get_depth_map(img_idx)
                # Get the number of rays with non-zero gt for this image
                N = G[np.nonzero(G)].shape[0]

            D = export_depth_map_from_voxel_indices(
                scene,
                img_idx,
                S[s:s + N],
                ray_voxel_indices[s:s + N],
                rays_idxs[s:s + N],
                grid_shape
            )
            s += N
            np.save(os.path.join(output_directory, "depth_mrf_%03d" % (image_cnt,)), D)
            print "%d" % (image_cnt,)
            image_cnt += 1

    def update_bp_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_pon=None
    ):
        """Run the belief propagation for a set of rays

        Arguments:
        ---------
            S: array(shape=(N, M), float32), A depth probability distribution
               for each of the N rays
            ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                               voxel grid per ray. M denotes the maximum number
                               of marched voxels.
            ray_voxel_count: array(shape=(N,), int) The number of voxels
                             intersected by each ray
            ray_to_occupancy_messages_pon: array(shape=(N, M), float32), Holds
                                           the ray_to_occupancy messages
                                           between bp iterations
        Returns:
        --------
            ray_to_occupancy_accumulated_pon: array(shape=(D1, D2, D3), float32)
            ray_to_occupancy_pon: array(shape=(N, M), float32)
        """
        raise NotImplementedError

    def estimate_depth_probabilities_from_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon,
        S_new
    ):
        """Based on the message updates compute the new depth probabilities
            Arguments:
            ---------
                S: array(shape=(N, M), float32), A depth probability
                   distribution for each of the N rays
                ray_voxel_indices: array(shape=(N, M, 3), int), The indices in
                                   the voxel grid per ray. M denotes the
                                   maximum number of marched voxels.
                ray_voxel_count: array(shape=(N,), int) The number of voxels
                                 intersected by each ray
                ray_to_occupancy_accumulated_pon: array(shape=(D1, D2, D3), float32)
                ray_to_occupancy_pon: array(shape=(N, M), float32)
                S_new: array(shape=(N, M), float32), A depth probability
                   distribution for each of the N rays
            Returns:
            --------
                S: array(shape=(N, M), float32), A depth probability
                   distribution for each of the N rays
        """
        raise NotImplementedError

    def mrf_inference(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_pon=None,
        S_new=None
    ):
        print "Computing the messages ..."
        ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon =\
            self.update_bp_messages(
                S,
                ray_voxel_indices,
                ray_voxel_count,
                ray_to_occupancy_pon
            )

        print "Computing the new depth distribution ..."
        S_new = self.estimate_depth_probabilities_from_messages(
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_accumulated_pon,
            ray_to_occupancy_pon,
            S_new
        )

        return ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon, S_new


class NPBPInference(BPInference):

    def update_bp_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_pon=None
    ):
        """Run the belief propagation for a set of rays

        Arguments:
        ---------
            S: array(shape=(N, M), float32), A depth probability distribution
               for each of the N rays
            ray_voxel_indices: array(shape=(N, M, 3), int), The indices in the
                               voxel grid per ray. M denotes the maximum number
                               of marched voxels.
            ray_voxel_count: array(shape=(N,), int) The number of voxels
                             intersected by each ray
            ray_to_occupancy_pon: array(shape=(N, M), float32), Holds the
                                  ray_to_occupancy messages between bp
                                  iterations
        Returns:
        --------
            ray_to_occupancy_accumulated_pon: array(shape=(D1, D2, D3), float32)
            ray_to_occupancy_pon: array(shape=(N, M), float32)
        """
        # Make sure that the messages have the correct shape and dtype
        assert S.shape[0] == ray_voxel_indices.shape[0]
        assert S.shape[0] == ray_voxel_count.shape[0]
        assert S.shape[0] == ray_to_occupancy_pon.shape[0]
        assert S.shape[1] == ray_voxel_indices.shape[1]
        assert S.shape[1] == ray_to_occupancy_pon.shape[1]
        assert len(ray_voxel_count.shape) == 1

        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype
        assert np.float32 == S.dtype
        assert np.float32 == ray_to_occupancy_pon.dtype

        # Compute the message updates
        ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon = np_bp(
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_pon,
            self._generation_params.grid_shape,
            gamma=self.gamma_prior,
            bp_iterations=self.bp_iterations
        )

        return ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon

    def estimate_depth_probabilities_from_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon,
        S_new
    ):
        """Based on the message updates compute the new depth probabilities
            Arguments:
            ---------
                S: array(shape=(N, M), float32), A depth probability
                   distribution for each of the N rays
                ray_voxel_indices: array(shape=(N, M, 3), int), The indices in
                                   the voxel grid per ray. M denotes the
                                   maximum number of marched voxels.
                ray_voxel_count: array(shape=(N,), int) The number of voxels
                             intersected by each ray
                ray_to_occupancy_accumulated_pon: array(shape=(D1, D2, D3), float32)
                ray_to_occupancy_pon: array(shape=(N, M), float32)
                S_new: array(shape=(N, M), float32), A depth probability
                       distribution for each of the N rays
            Returns:
            --------
                S_new: array(shape=(N, M), float32), A depth probability
                       distribution for each of the N rays
        """
        assert len(ray_voxel_count.shape) == 1

        S_new = np_compute_depth_distribution(
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_pon,
            ray_to_occupancy_accumulated_pon,
            S_new
        )
        return S_new


class TFBPInference(BPInference):
    def __init__(
        self,
        generation_params,
        N,
        bp_iterations=3,
        gamma_prior=0.05
    ):
        # N holds the number of rays
        super(TFBPInference, self).__init__(
            generation_params,
            bp_iterations,
            gamma_prior
        )
        self.tf_bp = build_belief_propagation(
            N,
            self._generation_params.max_number_of_marched_voxels,
            self._generation_params.grid_shape,
            self.gamma_prior,
            self.bp_iterations
        )
        self.tf_depth_update = build_depth_estimate(
            N,
            self._generation_params.max_number_of_marched_voxels,
            self._generation_params.grid_shape,
        )

    def update_bp_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_pon=None
    ):
        # Make sure that the messages have the correct shape and dtype
        assert S.shape[0] == ray_voxel_indices.shape[0]
        assert S.shape[0] == ray_voxel_count.shape[0]
        assert S.shape[1] == ray_voxel_indices.shape[1]
        assert len(ray_voxel_count.shape) == 1

        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype
        assert np.float32 == S.dtype

        # Prepare the inputs
        inputs = []
        inputs.append(S)
        inputs.append(ray_voxel_indices)
        inputs.append(ray_voxel_count)

        # Compute the message updates
        ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon =\
            self.tf_bp(inputs)

        return ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon

    def estimate_depth_probabilities_from_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon,
        S_new=None
    ):
        """Based on the message updates compute the new depth probabilities
            Arguments:
            ---------
                S: array(shape=(N, M), float32), A depth probability
                   distribution for each of the N rays
                ray_voxel_indices: array(shape=(N, M, 3), int)
                                   The indices in the voxel grid per ray.
                                   M denotes the maximum number of
                                   marched voxels.
                ray_voxel_count: array(shape=(N,), int) The number of voxels
                             intersected by each ray
                ray_to_occupancy_accumulated_pon: array(shape=(D1, D2, D3), float32)
                ray_to_occupancy_pon: array(shape=(N, M), float32)
            Returns:
            --------
                S_new: array(shape=(N, M), float32), A depth probability
                       distribution for each of the N rays
        """
        assert len(ray_voxel_count.shape) == 1

        inputs = []
        inputs.append(S)
        inputs.append(ray_voxel_indices)
        inputs.append(ray_voxel_count)
        inputs.append(ray_to_occupancy_accumulated_pon)
        inputs.append(ray_to_occupancy_pon)

        return self.tf_depth_update(inputs)


class CUDABPInference(BPInference):
    def __init__(
        self,
        generation_params,
        batch_size=1,
        bp_iterations=3,
        gamma_prior=0.05
    ):
        # N holds the number of rays
        super(CUDABPInference, self).__init__(
            generation_params,
            bp_iterations,
            gamma_prior
        )
        self.batch_size = batch_size

    def update_bp_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_pon
    ):
        # Make sure that the messages have the correct shape and dtype
        assert S.shape[0] == ray_voxel_indices.shape[0]
        assert S.shape[0] == ray_voxel_count.shape[0]
        assert S.shape[0] == ray_to_occupancy_pon.shape[0]
        assert S.shape[1] == ray_voxel_indices.shape[1]
        assert S.shape[1] == ray_to_occupancy_pon.shape[1]
        assert len(ray_voxel_count.shape) == 1

        assert np.int32 == ray_voxel_indices.dtype
        assert np.int32 == ray_voxel_count.dtype
        assert np.float32 == S.dtype
        assert np.float32 == ray_to_occupancy_pon.dtype

        # Compute the message updates
        ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon = cuda_bp(
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_pon,
            self._generation_params.grid_shape,
            gamma=self.gamma_prior,
            bp_iterations=self.bp_iterations,
            batch_size=self.batch_size
        )

        return ray_to_occupancy_accumulated_pon, ray_to_occupancy_pon

    def estimate_depth_probabilities_from_messages(
        self,
        S,
        ray_voxel_indices,
        ray_voxel_count,
        ray_to_occupancy_accumulated_pon,
        ray_to_occupancy_pon,
        S_new
    ):
        S_new = cuda_compute_depth_distribution(
            S,
            ray_voxel_indices,
            ray_voxel_count,
            ray_to_occupancy_pon,
            ray_to_occupancy_accumulated_pon,
            S_new,
            self._generation_params.grid_shape,
            self.batch_size
        )
        return S_new


def get_bp_backend(name, generation_params, **kwargs):
    bp = None

    bp_iterations =\
        kwargs["bp_iterations"] if "bp_iterations" in kwargs.keys() else 3

    if name == "numpy":
        bp = NPBPInference(generation_params, bp_iterations=bp_iterations)
    elif name == "tf":
        if kwargs and "N" in kwargs.keys():
            bp = TFBPInference(
                generation_params,
                kwargs["N"],
                bp_iterations=bp_iterations
            )
        else:
            raise ValueError("Missing argument for TF backend")
    elif name == "cuda":
        if kwargs and "batch_size" in kwargs.keys():
            bp = CUDABPInference(
                generation_params,
                kwargs["batch_size"],
                bp_iterations=bp_iterations
            )
        else:
            raise ValueError("Missing argument for CUDA backend")

    return bp
