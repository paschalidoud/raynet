import datetime
from itertools import product
from tempfile import mkdtemp
import os

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu

from tf_implementations.forward_pass_implementations\
    import build_multi_view_cnn_forward_pass_with_features,\
    build_full_multi_view_cnn_forward_pass_with_features
from utils.geometry import project, distance

from .cuda_implementations.sample_points import\
    compute_depth_from_distribution
from .cuda_implementations.similarities import\
    perform_multi_view_cnn_forward_pass_with_depth_estimation
from .cuda_implementations.mvcnn_with_ray_marching_and_voxels_mapping import\
    batch_mvcnn_voxel_traversal_with_ray_marching_with_depth_estimation
from .cuda_implementations.raynet_fp import perform_raynet_fp


class ForwardPass(object):
    """Provide the basic interface for the different forward passes we might
    use.
    """
    def __init__(
        self,
        model,
        generation_params,
        sampling_scheme,
        image_shape,
        rays_batch=50000,
        filter_out_rays=False
    ):
        # The trained model used to export features
        self._model = model
        # Parameters used for data generation
        self._generation_params = generation_params
        self._sampling_scheme = sampling_scheme
        # The number of rays per mini-batch
        self.rays_batch = rays_batch

        # Flag used to indicate whether we want to filter out rays
        self._filter_out_rays = filter_out_rays

        self._fp = None

    @staticmethod
    def create_depth_map_from_distribution(
        scene,
        img_idx,
        S,
        truncate=800,
        sampling_scheme="sample_in_bbox"
    ):
        """ Given a set of uniformly sampled points along all rays from the
        reference image identified with iand the corresponding per-ray depth
        distributions, we want to convert them to a depth map

        Arguments:
        ----------
            scene: Scene object
                   The Scene to be processed
            img_idx, int, Index to refer to the reference image
            S: np.array(shape=(N, D), dtype=np.float32)
               The per pixel depth distribution

        Returns:
        --------
            depth_map: numpy array, with the same dimensions as the image shape
        """
        # Extract the dimensions of the camera center
        H, W = scene.image_shape
        _, D = S.shape

        # Get the camera center of the reference image
        camera_center = scene.get_image(img_idx).camera.center

        D = compute_depth_from_distribution(
            np.arange(H*W, dtype=np.int32),
            scene.get_image(img_idx).camera.P_pinv,
            camera_center,
            H,
            W,
            scene.bbox.ravel(),
            S,
            np.arange(H*W, dtype=np.float32),
            sampling_scheme,
        ).reshape(W, H).T

        return np.minimum(D, truncate)

    @staticmethod
    def create_depth_map_from_distribution_with_voting(
        scene,
        img_idx,
        points,
        S,
        truncate=800
    ):
        """Given a set of uniformly sampled points along all rays from the
        reference image identified with iand the corresponding per-ray depth
        distributions, we want to convert them to a depth map, using the
        expectation value along the depth direction.

        Arguments:
        ----------
            scene: Scene object
                   The Scene to be processed
            img_idx, int, Index to refer to the reference image
            points: np.array(shape=(4, N, D), dtype=np.float32)
                    The uniformly sampled points across all rays, where N is
                    the number of rays and D is the number of discretization
                    steps used.
            S: np.array(shape=(N, D), dtype=np.float32)
               The per pixel depth distribution

        Returns:
        --------
            depth_map: numpy array, with the same dimensions as the image shape
        """
        # Extract the dimensions of the camera center
        H, W = scene.image_shape

        # Get the camera center of the reference image
        camera_center = scene.get_image(img_idx).camera.center

        # Compute the distances form the camera center for every depth
        # hypotheses
        dists = np.sqrt(
            ((camera_center.reshape(-1, 1, 1) - points)**2).sum(axis=0)
        )
        assert dists.shape == (H*W, points.shape[-1])
        D = (S*dists).sum(axis=-1)

        return np.minimum(D.reshape(W, H).T, truncate)

    @staticmethod
    def upsample_features(features, model):
        """Parse the model for strides>1 and determine how much we should
        upsample the features.

        NOTE: The code assumes that the network is a single stream of conv -
              pool layers.

        Arguments
        ---------
            features: (N, H, W, F)
                      Array with features of N images with F dimensions
            model: Keras model
                   The model that created the features
        """
        # Collect strides
        strides = [
            l.strides[0] if hasattr(l, "strides") else 1
            for l in model.layers
        ]
        upsample = sum(s for s in strides if s > 1)

        if upsample <= 1:
            return features
        else:
            return np.kron(features, np.ones((1, upsample, upsample, 1)))

    def get_valid_rays_per_image(self, scene, i):
        H, W = scene.image_shape
        idxs = np.arange(H*W, dtype=np.int32)

        if self._filter_out_rays:
            idxs = idxs.reshape(W, H).T
            # Get the gt depth map for the current scene
            G = scene.get_depth_map(ref_idx)
            # Collect the idxs where the ground truth is non-zero
            return idxs[G != 0].ravel()
        else:
            return idxs

    def _to_list_with_zeropadded_images(self, images, inputs=None):
        # Check if the inputs is None or if it already contains elements
        if inputs is None:
            inputs = []

        # Dimensions of the image
        H, W, C = images[0].image.shape
        p = self._generation_params.padding
        zp_shape = (H+2*p, W+2*p, C)
        # Add the image one by one
        for im in images:
            # Create zeroppaded image that will be used for the forward pass
            zeropadded = np.zeros(zp_shape)
            # Apply the zeropadding
            zeropadded[p:p+H, p:p+W, :] = im.image
            inputs.append(zeropadded)

        return inputs

    def sample_points(self, scene, i):
        # TODO: DELETE THIS FUCTION
        pass

    def sample_points_batched(self, scene, i, batch):
        # TODO: DELETE THIS FUCTION
        pass

    def forward_pass(self, scene, images_range):
        """Given a scene and an image range that identify the indices of the images, we predict the
        corresponding depth maps for them.

        Arguments:
        ---------
            scene: Scene object
                   The Scene to be processed
            images_range: tuple, Indices to specify the images to be used for
                          the reconstruction

        Returns:
        --------
            depth_map: numpy array, with the same dimensions as the image shape
        """
        raise NotImplementedError()


class MultiViewCNNForwardPass(ForwardPass):
    """Perform the forward pass only for the MultiViewCNN"""
    def __init__(
        self,
        model,
        generation_params,
        sampling_scheme,
        image_shape,
        rays_batch,
        filter_out_rays=False
    ):
        super(MultiViewCNNForwardPass, self).__init__(
            model,
            generation_params,
            sampling_scheme,
            image_shape,
            rays_batch,
            filter_out_rays
        )
        self.ref_idx = -1

        # Allocate GPU memory
        D = self._generation_params.depth_planes
        self.s_gpu = to_gpu(
            np.zeros((self.rays_batch, D), dtype=np.float32)
        )
        self.points_gpu = to_gpu(
            np.zeros((self.rays_batch, D, 4), dtype=np.float32)
        )

    def sim(self, scene, feature_size):
        if self._fp is None:
            self._fp = perform_multi_view_cnn_forward_pass_with_depth_estimation(
                self._generation_params.depth_planes,
                self._generation_params.neighbors + 1,
                feature_size,
                scene.image_shape[0],
                scene.image_shape[1],
                self._generation_params.padding,
                scene.bbox.ravel(),
                self._sampling_scheme
            )
        return self._fp


    def forward_pass(self, scene, images_range):
        # Make sure that the images_range is a tuple
        assert isinstance(images_range, tuple)

        # Declare some variables that we will need
        (start_img_idx, end_img_idx, skip) = images_range
        D = self._generation_params.depth_planes  # number of depth planes
        batch_size = self.rays_batch  # number of rays in each mini-batch
        H, W = scene.image_shape

        self.ref_idx = start_img_idx  # initiaze with the first image

        while self.ref_idx < end_img_idx:
            # Get the valid rays for the current image
            ray_idxs = self.get_valid_rays_per_image(scene, self.ref_idx)

            # Based on the i index compute the multi-view Image objects
            images = scene.get_image_with_neighbors(self.ref_idx)

            # Start adding the features
            a = datetime.datetime.now()
            features = self._model.predict(
                np.stack(self._to_list_with_zeropadded_images(images), axis=0)
            )
            b = datetime.datetime.now()
            c = b - a
            print "Features computation - ", c.total_seconds()

            # Get the projection matrices of all the neighbor views, the
            # projection matrix and the camera center of the reference view
            P = [im.camera.P for im in images]
            P_inv = images[0].camera.P_pinv
            camera_center = images[0].camera.center

            a = datetime.datetime.now()
            # Move to GPU to save some time frome copying
            features_gpu = to_gpu(features.ravel())
            ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
            P_gpu = to_gpu(np.array(P).ravel())
            P_inv_gpu = to_gpu(P_inv.ravel())
            camera_center_gpu = to_gpu(camera_center)
            _, _, _, F = features.shape

            depth_map = to_gpu(
                np.zeros((H*W), dtype=np.float32)
            )

            # Start iterationg over the batch of rays
            for i in range(0, len(ray_idxs), batch_size):
                self.s_gpu.fill(0)
                self.points_gpu.fill(0)

                self.sim(scene, F)(
                    ray_idxs_gpu[i:i+batch_size],
                    features_gpu,
                    P_gpu,
                    P_inv_gpu,
                    camera_center_gpu,
                    self.s_gpu,
                    self.points_gpu,
                    depth_map[i:i+batch_size]
                )

            b = datetime.datetime.now()
            c = b - a
            print "Per-pixel depth estimation - ", c.total_seconds()

            # Move to the next image
            self.ref_idx += skip
            yield depth_map.get().reshape(W, H).T

            # TODO: Fix the memory allocation pattern so we don't delete and
            # reallocate
            del features_gpu


class MultiViewCNNVoxelSpaceForwardPass(ForwardPass):
    """Perform the forward pass only for the MultiViewCNN"""
    def __init__(
        self,
        model,
        generation_params,
        sampling_scheme,
        image_shape,
        rays_batch,
        filter_out_rays=False
    ):
        super(MultiViewCNNVoxelSpaceForwardPass, self).__init__(
            model,
            generation_params,
            sampling_scheme,
            image_shape,
            rays_batch,
            filter_out_rays
        )
        self.ref_idx = -1

        # Allocate GPU memory
        M = self._generation_params.max_number_of_marched_voxels
        self.s_gpu = to_gpu(
            np.zeros((rays_batch, M), dtype=np.float32)
        )
        self.ray_voxel_count_gpu = to_gpu(
            np.zeros((rays_batch,), dtype=np.int32)
        )
        self.ray_voxel_indices_gpu = to_gpu(
            np.zeros((rays_batch, M, 3), dtype=np.int32)
        )
        self.voxel_grid_gpu = None

    def sim(self, scene, feature_size):
        if self._fp is None:
            grid_shape = np.array(
                scene.voxel_grid(
                    self._generation_params.grid_shape
                ).shape[1:]
            )
            self._fp = batch_mvcnn_voxel_traversal_with_ray_marching_with_depth_estimation(
                self._generation_params.max_number_of_marched_voxels,
                self._generation_params.depth_planes,
                self._generation_params.neighbors + 1,
                feature_size,
                scene.image_shape[0],
                scene.image_shape[1],
                self._generation_params.padding,
                scene.bbox.ravel(),
                grid_shape,
                self._sampling_scheme
            )
        return self._fp

    def voxel_grid_to_gpu(self, scene):
        if self.voxel_grid_gpu is None:
            self.voxel_grid_gpu = to_gpu(scene.voxel_grid(
                self._generation_params.grid_shape
            ).transpose(1, 2, 3, 0).ravel())
        return self.voxel_grid_gpu

    def forward_pass(self, scene, images_range):
        # Make sure that the images_range is a tuple
        assert isinstance(images_range, tuple)

        # Declare some variables that we will need
        (start_img_idx, end_img_idx, skip) = images_range
        D = self._generation_params.depth_planes  # number of depth planes
        batch_size = self.rays_batch  # number of rays in each mini-batch
        H, W = scene.image_shape

        self.ref_idx = start_img_idx  # initiaze with the first image

        while self.ref_idx < end_img_idx:
            # Get the valid rays for the current image
            ray_idxs = self.get_valid_rays_per_image(scene, self.ref_idx)

            # Based on the i index compute the multi-view Image objects
            images = scene.get_image_with_neighbors(self.ref_idx)

            # Start adding the features
            a = datetime.datetime.now()
            features = self._model.predict(
                np.stack(self._to_list_with_zeropadded_images(images), axis=0)
            )
            b = datetime.datetime.now()
            c = b - a
            print "Features computation - ", c.total_seconds()

            # Get the projection matrices of all the neighbor views, the
            # projection matrix and the camera center of the reference view
            P = [im.camera.P for im in images]
            P_inv = images[0].camera.P_pinv
            camera_center = images[0].camera.center

            a = datetime.datetime.now()
            # Move to GPU to save some time frome copying
            features_gpu = to_gpu(features.ravel())
            ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
            P_gpu = to_gpu(np.array(P).ravel())
            P_inv_gpu = to_gpu(P_inv.ravel())
            camera_center_gpu = to_gpu(camera_center)
            _, _, _, F = features.shape

            depth_map = to_gpu(
                np.zeros((H*W), dtype=np.float32)
            )

            # Start iterationg over the batch of rays
            for i in range(0, len(ray_idxs), batch_size):
                self.s_gpu.fill(0)
                self.ray_voxel_indices_gpu.fill(0)
                self.ray_voxel_count_gpu.fill(0)

                self.sim(scene, F)(
                    ray_idxs_gpu[i:i+batch_size],
                    features_gpu,
                    P_gpu,
                    P_inv_gpu,
                    camera_center_gpu,
                    self.voxel_grid_to_gpu(scene),
                    self.ray_voxel_indices_gpu,
                    self.ray_voxel_count_gpu,
                    self.s_gpu,
                    depth_map[i:i+batch_size]
                )

            b = datetime.datetime.now()
            c = b - a
            print "Per-pixel depth estimation - ", c.total_seconds()

            # Move to the next image
            self.ref_idx += skip
            yield depth_map.get().reshape(W, H).T

            # TODO: Fix the memory allocation pattern so we don't delete and
            # reallocate
            del features_gpu


class RayNetForwardPass(ForwardPass):
    def __init__(
        self,
        model,
        generation_params,
        sampling_scheme,
        image_shape,
        rays_batch,
        filter_out_rays=False
    ):
        super(RayNetForwardPass, self).__init__(
            model,
            generation_params,
            sampling_scheme,
            image_shape,
            rays_batch,
            filter_out_rays
        )
        self.rays_batch = rays_batch
        self.ref_idx = -1
        self._de = None

        # Allocate GPU memory
        M = self._generation_params.max_number_of_marched_voxels
        grid_shape = self._generation_params.grid_shape
        gamma = self._generation_params.gamma_mrf

        self.s_gpu = to_gpu(
            np.zeros((rays_batch, M), dtype=np.float32)
        )
        self.ray_voxel_count_gpu = to_gpu(
            np.zeros((rays_batch,), dtype=np.int32)
        )
        self.ray_voxel_indices_gpu = to_gpu(
            np.zeros((rays_batch, M, 3), dtype=np.int32)
        )
        self.voxel_grid_gpu = None

        # Initialize the ray to occupancy accumulators as GPU arrays
        self.ray_to_occupancy_accumulated_pon = to_gpu(
            np.ones(
                tuple(grid_shape),
                dtype=np.float32
            ) * (np.log(gamma) - np.log(1 - gamma))
        )
        self.ray_to_occupancy_accumulated_out_pon = to_gpu(
            np.ones(
                tuple(grid_shape),
                dtype=np.float32
            ) * (np.log(gamma) - np.log(1 - gamma))
        )

        # Create a temporary directory to save the per-image
        # ray_to_occupancy_messages during the MRF inference
        self.ray_occ_msgs_dir = mkdtemp()
        print "Saving stuff to random directory %s .." % (self.ray_occ_msgs_dir,)

    def raynet_fp(self, scene, feature_size):
        if self._fp is None:
            # Declare some variables to comply with PEP8
            H, W = scene.image_shape
            bbox = scene.bbox.ravel()
            grid_shape = np.array(
                scene.voxel_grid(
                    self._generation_params.grid_shape
                ).shape[1:]
            )

            self._fp, self._de = perform_raynet_fp(
                self._generation_params.max_number_of_marched_voxels,
                self._generation_params.depth_planes,
                self._generation_params.neighbors + 1,
                feature_size,
                H,
                W,
                self._generation_params.padding,
                scene.bbox.ravel(),
                grid_shape,
                self._sampling_scheme
            )

        return [self._fp, self._de]

    def voxel_grid_to_gpu(self, scene):
        if self.voxel_grid_gpu is None:
            self.voxel_grid_gpu = to_gpu(scene.voxel_grid(
                self._generation_params.grid_shape
            ).transpose(1, 2, 3, 0).ravel())
        return self.voxel_grid_gpu
    

    def forward_pass(self, scene, images_range):
        # Make sure that the images_range is a tuple
        assert isinstance(images_range, tuple)

        # Declare some variables that we will need
        (start_img_idx, end_img_idx, skip) = images_range
        D = self._generation_params.depth_planes  # number of depth planes
        M = self._generation_params.max_number_of_marched_voxels
        gamma = self._generation_params.gamma_mrf
        batch_size = self.rays_batch  # number of rays in each mini-batch
        H, W = scene.image_shape
        bp_iterations = 3  # number of Belief Propagation updates

        # Iterate over the rays multiple times
        for it in xrange(bp_iterations):
            print "Iteration %d " % (it,)
            # Interate over each image in the images_range
            for ref_idx in xrange(start_img_idx, end_img_idx, skip):
                # Get the valid rays for the current image
                ray_idxs = self.get_valid_rays_per_image(scene, self.ref_idx)

                # Get the path to the ray_to_occupancey_messages_pon for the
                # current image
                ray_to_occupancy_messages_pon_path = os.path.join(
                    self.ray_occ_msgs_dir,
                    "ray_to_occupancey_messages_pon_%d.dat" % (ref_idx,)
                )
                ray_to_occupancy_messages_pon = np.memmap(
                    ray_to_occupancy_messages_pon_path,
                    dtype="float32",
                    mode="w+",
                    shape=(len(ray_idxs), M)
                )

                # For the first bp iteration initialize messages to 0
                if it == 0:
                    ray_to_occupancy_messages_pon.fill(0)

                # Based on the i index compute the multi-view Image objects
                images = scene.get_image_with_neighbors(ref_idx)

                # Start adding the features
                a = datetime.datetime.now()
                features = self._model.predict(
                    np.stack(self._to_list_with_zeropadded_images(images), axis=0)
                )
                b = datetime.datetime.now()
                c = b - a
                print "Features computation - ", c.total_seconds()

                # Get the projection matrices of all the neighbor views, the
                # projection matrix and the camera center of the reference view
                P = [im.camera.P for im in images]
                P_inv = images[0].camera.P_pinv
                camera_center = images[0].camera.center

                a = datetime.datetime.now()
                # Move to GPU to save some time frome copying
                features_gpu = to_gpu(features.ravel())
                ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
                P_gpu = to_gpu(np.array(P).ravel())
                P_inv_gpu = to_gpu(P_inv.ravel())
                camera_center_gpu = to_gpu(camera_center)
                _, _, _, F = features.shape

                # Start iterationg over the batch of rays
                for i in range(0, len(ray_idxs), batch_size):
                    self.s_gpu.fill(0)
                    self.ray_voxel_indices_gpu.fill(0)
                    self.ray_voxel_count_gpu.fill(0)

                    msgs = self.raynet_fp(scene, F)[0](
                        ray_idxs_gpu[i:i+batch_size],
                        features_gpu,
                        P_gpu,
                        P_inv_gpu,
                        camera_center_gpu,
                        self.voxel_grid_to_gpu(scene),
                        self.ray_voxel_indices_gpu,
                        self.ray_voxel_count_gpu,
                        self.s_gpu,
                        self.ray_to_occupancy_accumulated_pon,
                        ray_to_occupancy_messages_pon[i:i+batch_size],
                        self.ray_to_occupancy_accumulated_out_pon
                    )
                    ray_to_occupancy_messages_pon[i:i+batch_size] = msgs.get()


                b = datetime.datetime.now()
                c = b - a
                print "Message passing - ", c.total_seconds()

                # TODO: Fix the memory allocation pattern so we don't delete and
                # reallocate
                del features_gpu
        
            # Swap the accumulators for the next bp iteration
            self.ray_to_occupancy_accumulated_out_pon, self.ray_to_occupancy_accumulated_pon = \
                self.ray_to_occupancy_accumulated_pon, self.ray_to_occupancy_accumulated_out_pon
            self.ray_to_occupancy_accumulated_out_pon.fill(np.log(gamma) - np.log(1 - gamma))
    
        # Now estimate the depth map
        self.ref_idx = start_img_idx  # initiaze with the first image
        while self.ref_idx < end_img_idx:
            # Get the valid rays for the current image
            ray_idxs = self.get_valid_rays_per_image(scene, self.ref_idx)

            # Based on the i index compute the multi-view Image objects
            images = scene.get_image_with_neighbors(self.ref_idx)

            # Start adding the features
            a = datetime.datetime.now()
            features = self._model.predict(
                np.stack(self._to_list_with_zeropadded_images(images), axis=0)
            )
            b = datetime.datetime.now()
            c = b - a
            print "Features computation - ", c.total_seconds()

            # Get the projection matrices of all the neighbor views, the
            # projection matrix and the camera center of the reference view
            P = [im.camera.P for im in images]
            P_inv = images[0].camera.P_pinv
            camera_center = images[0].camera.center

            a = datetime.datetime.now()
            # Move to GPU to save some time frome copying
            features_gpu = to_gpu(features.ravel())
            ray_idxs_gpu = to_gpu(ray_idxs.astype(np.int32))
            P_gpu = to_gpu(np.array(P).ravel())
            P_inv_gpu = to_gpu(P_inv.ravel())
            camera_center_gpu = to_gpu(camera_center)
            _, _, _, F = features.shape

            depth_map = to_gpu(
                np.zeros((H*W), dtype=np.float32)
            )

            # Start iterationg over the batch of rays
            for i in range(0, len(ray_idxs), batch_size):
                self.s_gpu.fill(0)
                self.ray_voxel_indices_gpu.fill(0)
                self.ray_voxel_count_gpu.fill(0)
                
                self.raynet_fp(scene, F)[1](
                    ray_idxs_gpu[i:i+batch_size],
                    features_gpu,
                    P_gpu,
                    P_inv_gpu,
                    camera_center_gpu,
                    self.voxel_grid_to_gpu(scene),
                    self.ray_voxel_indices_gpu,
                    self.ray_voxel_count_gpu,
                    self.s_gpu,
                    self.ray_to_occupancy_accumulated_pon,
                    ray_to_occupancy_messages_pon[i:i+batch_size],
                    depth_map[i:i+batch_size]
                )

            b = datetime.datetime.now()
            c = b - a
            print "Per-pixel depth estimation - ", c.total_seconds()

            # Move to the next image
            self.ref_idx += skip
            yield depth_map.get().reshape(W, H).T

            # TODO: Fix the memory allocation pattern so we don't delete and
            # reallocate
            del features_gpu


class HartmannForwardPass(ForwardPass):
    """Perform the forward pass for the Hartmann et al. baseline.
    """
    def __init__(
        self,
        model,
        generation_params,
        sampling_scheme,
        image_shape,
        batch_size=64
    ):
        super(HartmannForwardPass, self).__init__(
            model,
            generation_params,
            sampling_scheme,
            image_shape
        )
        self.batch_size = batch_size

    def _generate_batches(self, images, neighbor_pixels, batch_size):
        # The image dimensions
        H, W = images[0].height, images[0].width

        # Create the output
        n_patches = batch_size * self._generation_params.depth_planes
        shape = (n_patches,) + self._generation_params.patch_shape
        X = [
            np.empty(shape, dtype=np.float32)
            for i in range(self._generation_params.neighbors + 1)
        ]

        pixels = np.array([
            [u, v, 1.] for u, v in product(range(W), range(H))
        ], dtype=np.float32).T
        pixels = pixels.astype(np.int32)

        cnt = 0
        for pix in pixels.T:
            ref_patch = images[0].patch(
                pix[:, np.newaxis],
                self._generation_params.patch_shape[:-1]
            )

            for d in range(self._generation_params.depth_planes):
                X[0][cnt] = ref_patch
                idx = (H*pix[0] + pix[1])*self._generation_params.depth_planes + d
                for i in range(1, self._generation_params.neighbors + 1):
                    X[i][cnt] = images[i].patch(
                        neighbor_pixels[i-1][:, idx, np.newaxis],
                        self._generation_params.patch_shape[:-1]
                    )

                cnt += 1
                if cnt == n_patches:
                    cnt = 0
                    yield X

        if cnt > 0:
            yield [
                xi[:cnt]
                for xi in X
            ]

        # Just to satisfy Keras that wants endless iterators :-)
        while True:
            yield X

    def forward_pass(self, scene, i):
        # Based on the i index compute the multi-view Image objects
        images = scene.get_image_with_neighbors(i)

        # Given a sample scheme sample_points across all rays that emanate from
        # the camera center and pass through all pixels in the reference image
        points = self.sample_points(scene, i)

        # Reproject all the points to the neighboring images
        neighbor_pixels = [
            project(images[i+1].camera.P, points.reshape(4, -1)).T
            for i in range(self._generation_params.neighbors)
        ]
        neighbor_pixels = [
            np.round(npi).astype(np.int32)
            for npi in neighbor_pixels
        ]

        # Get the dimensions of the images in the scene
        H, W = scene.image_shape
        # Do the forward pass through the network
        predictions = self._model.predict_generator(
            self._generate_batches(images, neighbor_pixels, self.batch_size),
            steps=np.ceil(float(H*W)/self.batch_size),
            verbose=1
        )

        # Do the final depth estimation
        predicted_points = points.T[
            predictions[:, 0, 0, 0].reshape(H*W, -1).argmax(axis=-1),
            np.arange(H*W)
        ].reshape(W, H, 4)

        camera_center = images[0].camera.center
        depth_map = np.sqrt((
                (predicted_points - camera_center.reshape(1, 1, 4))**2
        ).sum(axis=-1)).T

        return depth_map, points


def get_forward_pass_factory(name):
    return {
        "multi_view_cnn": MultiViewCNNForwardPass,
        "multi_view_cnn_voxel_space": MultiViewCNNVoxelSpaceForwardPass,
        "hartmann_fp": HartmannForwardPass,
        "raynet": RayNetForwardPass
    }[name]
