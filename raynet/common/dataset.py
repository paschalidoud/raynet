import os

import numpy as np

from scene import RestrepoScene, DTUScene


class Dataset(object):
    """Dataset represents a dataset.

    We want to be able to deal with multiple datasets such as the DTU dataset
    the aerial datasets and maybe even more. Dataset handles them in a generic
    manner.
    """
    def __init__(
        self,
        dataset_directory,
        select_neighbors_based_on="filesystem"
    ):
        self._dataset_directory = dataset_directory
        # Dictionary used to cache the scenes of a dataset
        self._cache = {}
        self._max_cache_size = 2
        self._select_neighbors_based_on = select_neighbors_based_on

    @property
    def n_scenes(self):
        return len(os.listdir(self._dataset_directory))

    @property
    def scenes(self):
        return sorted(os.listdir(self._dataset_directory))

    def get_scene(self, scene_idx):
        raise NotImplementedError()


class RestrepoDataset(Dataset):
    def __init__(self, dataset_directory, select_neighbors_based_on):
        super(RestrepoDataset, self).__init__(
            dataset_directory,
            select_neighbors_based_on
        )
        # Create a mapping from scenes to indices based on their alphabetical
        # order
        self._scene_mapping = {}
        for i, x in enumerate(self.scenes):
            self._scene_mapping[i] = x
        print self._scene_mapping

    def get_scene(self, scene_idx):
        if scene_idx in self._scene_mapping.keys():
            if scene_idx not in self._cache:
                self._cache[scene_idx] = RestrepoScene(
                    os.path.join(
                        self._dataset_directory,
                        self._scene_mapping[scene_idx]
                    ),
                    select_neighbors_based_on=self._select_neighbors_based_on
                )
            return self._cache[scene_idx]
        else:
            raise ValueError(
                "scene_idx must be %r" % (self._scene_mapping.keys())
            )


class DTUDataset(Dataset):
    def __init__(
        self,
        dataset_directory,
        illumination,
        select_neighbors_based_on
    ):
        self._illumination = illumination
        super(DTUDataset, self).__init__(
            dataset_directory,
            select_neighbors_based_on
        )

    @property
    def n_scenes(self):
        return len(
            os.listdir(os.path.join(self._dataset_directory, "Rectified"))
        )

    def get_scene(self, scene_idx):
        if scene_idx not in self._cache:
            # Check the nuber of cached elements
            keys = self._cache.keys()
            # Check if we have reached the maximum cache size
            if len(keys) + 1 > self._max_cache_size:
                try:
                    del self._cache[np.random.choice(keys)]
                except KeyError:
                    pass
            self._cache[scene_idx] = DTUScene(
                self._dataset_directory,
                scene_idx,
                self._illumination,
                select_neighbors_based_on=self._select_neighbors_based_on
            )
        return self._cache[scene_idx]
