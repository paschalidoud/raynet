import numpy as np
from multiprocessing import Process

#import time
#from tqdm import tqdm

class RayNetBatchProvider(object):
    def __init__(
        self,
        dataset,
        sample_generator,
        batch_size=1000
    ):
        self._dataset = dataset
        self._sample_generator = sample_generator
        self._batch_size = batch_size

        # Denote variables that will be used to create the batches
        self.t_images = None
        self.t_points = None
        self.t_ray_voxel_indices = None
        self.t_ray_voxel_count = None
        self.t_S_target = None
        self.t_scene_idx = None

    def _allocate_memory_for_batch(self, generation_parameters):
        # Extract all the important variables
        patch_shape = generation_parameters.patch_shape
        D = generation_parameters.depth_planes
        M = generation_parameters.max_number_of_marched_voxels
        views = generation_parameters.neighbors + 1
        grid_shape = np.array(generation_parameters.grid_shape, dtype=np.int32)

        # Allocate space for the inputs
        images_shape = (views, self._batch_size, D) + patch_shape
        self.t_images = np.empty(shape=images_shape, dtype=np.float32)
        self.t_points = np.empty(
            shape=(self._batch_size, D, 4),
            dtype=np.float32
        )

        self.t_ray_voxel_indices = np.empty(
            shape=(self._batch_size, M, 3),
            dtype=np.int32
        )
        self.t_ray_voxel_count = np.empty(
            shape=(self._batch_size,),
            dtype=np.int32
        )

        self.t_S_target = np.empty(
            shape=(self._batch_size, M),
            dtype=np.float32
        )
        self.t_scene_idx = np.ones(shape=(self._batch_size,), dtype=np.int32)
        self.t_camera_centers = np.empty(
            shape=(self._batch_size, 4),
            dtype=np.float32
        )
        self.t_image_idxs = np.ones(shape=(self._batch_size,), dtype=np.int32)

    def _generate_mini_batch(
        self,
        t_images,
        t_points,
        t_ray_voxel_indices,
        t_ray_voxel_count,
        t_S_target,
        t_camera_centers,
        t_image_idxs,
        t_scene_idx,
        start_idx,
        end_idx
    ):
        # Run the loop to generate the mini-batch
        cnt = start_idx
        # Keep track of the scene index
        scene_idx = None
        #pbar = tqdm(total = end_idx)
        while cnt < end_idx:
            sample = self._sample_generator.get_sample(self._dataset)
            # Update the buffers based on the inputs
            if sample.X is not None and sample.y is not None:
                #pbar.update(1)
                t_images[:, cnt] = sample.X
                t_ray_voxel_indices[cnt] = sample.ray_voxel_indices
                t_ray_voxel_count[cnt] = sample.Nr
                t_points[cnt] = sample.points
                t_S_target[cnt] = sample.y
                t_scene_idx[cnt] = sample.scene_idx
                t_camera_centers[cnt] = sample.camera_center.ravel()
                t_image_idxs[cnt] = sample.img_idx

                cnt += 1
        #pbar.close()

    def get_batch_of_rays(self, generation_parameters):
        raise NotImplementedError()


class SingleThreadRayNetBatchProvider(RayNetBatchProvider):
    def get_batch_of_rays(self, generation_parameters):
        if self.t_images is None:
            self._allocate_memory_for_batch(generation_parameters)

        self._generate_mini_batch(
            self.t_images,
            self.t_points,
            self.t_ray_voxel_indices,
            self.t_ray_voxel_count,
            self.t_S_target,
            self.t_camera_centers,
            self.t_image_idxs,
            self.t_scene_idx,
            0,
            self._batch_size
        )

        # Make sure that the entire batch size contains the same scene_idx
        if np.all(self.t_scene_idx == self.t_scene_idx[0]):
            pass
        else:
            raise ValueError(
                "The same batch contains data from different scenes %r"
                % (set(self.t_scene_idx),)
            )

        grid_shape = np.array(generation_parameters.grid_shape, dtype=np.int32)
        inputs = []
        for i in range(generation_parameters.neighbors + 1):
            inputs.append(self.t_images[i])
        # Append the voxel grid for the current scene. We assume that a batch
        # contains rays from the SAME scene, this is why we only append the
        # voxel_grid once per batch
        inputs.append(
            self._dataset.get_scene(self.t_scene_idx[0]).voxel_grid(grid_shape)
        )
        inputs.append(self.t_ray_voxel_indices)
        inputs.append(self.t_ray_voxel_count)
        inputs.append(self.t_S_target)
        inputs.append(self.t_points)
        inputs.append(self.t_camera_centers)

        return inputs


class MultiThreadRayNetBatchProvider(RayNetBatchProvider):
    def __init__(
        self,
        dataset,
        sample_generator,
        n_threads=12,
        batch_size=1000
    ):
        super(MultiThreadRayNetBatchProvider, self).__init__(
            dataset,
            sample_generator,
            batch_size
        )
        self._n_threads = n_threads

        # Compute the amount of datah that every process should generate
        data_per_process = []
        if self._batch_size % self._n_threads == 0:
            d = self._batch_size / self._n_threads
            # All processes should process the same amount of data
            data_per_process = [d] * self._n_threads
        else:
            d = self._batch_size / (self._n_threads - 1)
            # All processes except for the last guy with process the same
            # amount of data
            data_per_process = [d] * (self._n_threads - 1)
            # Compute what is left to be computed
            left = self._batch_size - d * (self._n_threads - 1)
            data_per_process.append(left)

        self._start_end = []
        start = end = -1
        for i in range(self._n_threads):
            start = end + 1
            end = start + data_per_process[i] - 1
            self._start_end.append((start, end))
        # print self._start_end

    def get_batch_of_rays(self, generation_parameters):
        # Allocate memory for the batch
        if self.t_images is None:
            self._allocate_memory_for_batch(generation_parameters)

        # Start the worker processes
        processes = [
            Process(
                target=self._generate_mini_batch,
                args=(
                    self.t_images,
                    self.t_points,
                    self.t_ray_voxel_indices,
                    self.t_ray_voxel_count,
                    self.t_S_target,
                    self.t_camera_centers,
                    self.t_scene_idx,
                    self._start_end[i][0],  # start
                    self._start_end[i][1]  # end
                ),
                name="worker-thread-{}".format(i)
            ) for i in range(self._n_threads)
        ]

        # print "Starting {} worker processes".format(len(processes))
        for p in processes:
            p.daemon = True
            p.start()

        # Make sure that the entire batch size contains the same scene_idx
        if np.all(self.t_scene_idx == self.t_scene_idx[0]):
            pass
        else:
            raise ValueError(
                "The same batch contains data from different scenes %r"
                % (set(t_scene_idx))
            )

        grid_shape = np.array(generation_parameters.grid_shape, dtype=np.int32)
        voxel_grid = self._dataset.get_scene(
            self.t_scene_idx[0]
        ).voxel_grid(grid_shape)

        inputs = []
        for i in range(generation_parameters.neighbors + 1):
            inputs.append(self.t_images[i])
        # Append the voxel grid for the current scene. We assume that a batch
        # contains rays from the SAME scene, this is why we only append the
        # voxel_grid once per batch
        inputs.append(voxel_grid)
        inputs.append(self.t_ray_voxel_indices)
        inputs.append(self.t_ray_voxel_count)
        inputs.append(self.t_S_target)
        inputs.append(self.t_points)
        inputs.append(self.t_camera_centers)

        # Now that everything is done terminate the processes gracefully
        for p in processes:
            p.join()

        return inputs
