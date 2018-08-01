import os
import unittest

import numpy as np
from scipy.misc import imsave, imread
from tempfile import mkdtemp as make_temporary_directory, TemporaryFile
from xml.etree import ElementTree

from raynet.common import scene
from raynet.common.sampling_schemes import SamplingInBboxScheme
from raynet.common.generation_parameters import GenerationParameters

class SceneTest(unittest.TestCase):
    """Contains tests for the various scenes that we need to use"""
    @staticmethod
    def get_restrepo_bbox(tmp):
        # Generate a mock scene_info.xml file to check the bounding box
        # Fix the root element
        root = ElementTree.Element("bwm_info_for_boxm2")
        # bbox element
        attr = {}
        attr["minx"] = "-2.5"
        attr["miny"] = "-2.5"
        attr["minz"] = "-0.5"
        attr["maxx"] = "2.5"
        attr["maxy"] = "2.5"
        attr["maxz"] = "0.5"
        bbox = ElementTree.SubElement(root, "bbox", attr)
        # resolution element
        attr = {}
        attr["val"] = "0.001"
        res = ElementTree.SubElement(root, "resolution", attr)
        # ntrees element
        attr = {}
        attr["ntrees_x"] = "48"
        attr["ntrees_y"] = "48"
        attr["ntrees_z"] = "48"
        ntrees = ElementTree.SubElement(root, "ntrees", attr)

        tree = ElementTree.ElementTree(root)
        tree.write(os.path.join(tmp, "scene_info.xml"))

    @staticmethod
    def get_temporary_restrepo_dataset():
        # set up a mock dataset with a dataset in the Restrepo et al. format
        tmp = make_temporary_directory()
        print tmp
        os.makedirs(os.path.join(tmp, "imgs"))
        os.makedirs(os.path.join(tmp, "cams_krt"))

        # Create 50 random views
        for i in range(50):
            D = np.random.random((200, 300, 3))
            imsave(os.path.join(tmp, "imgs", "frame_%03d.png" %(i,)), D)

            with open(os.path.join(tmp, "cams_krt", "camera_%03d.txt" %(i,)), "w") as f:
                K = np.random.random((3, 3))*200
                R = np.random.random((3, 3))*10
                t = np.random.random((1, 3))
                np.savetxt(f, K)
                f.write("\n")
                np.savetxt(f, R)
                f.write("\n")
                np.savetxt(f, t)
        
        # Generate the scene_info.xml file
        SceneTest.get_restrepo_bbox(tmp)

        return tmp

    @staticmethod
    def is_empty(x):
        """Checks if x is all filled with -1"""
        return x.sum() == -np.prod(x.shape)

    @staticmethod
    def compute_patches_from_point(images, point):
        expand_patch = False
        patch_shape = (11, 11, 3)
        # Project the point specified by the point_index in all images and get
        # the corresponding patches
        patches = [
            im.patch_from_3d(
                point.reshape(-1, 1),
                patch_shape[:2],
                expand_patch
            )
            for im in images
        ]

        # Check if we need to bail because some point is projected outside of
        # some image
        if not expand_patch and any(map(SceneTest.is_empty, patches)):
            return None

        return patches

    def test_restrepo_scene(self):
        dataset_directory = self.get_temporary_restrepo_dataset()
        s = scene.RestrepoScene(dataset_directory)
        # Make sure that the scene contains the actual number of images
        self.assertEqual(s.n_images, 50)
        self.assertEqual(s.image_shape[0], 200)
        self.assertEqual(s.image_shape[1], 300)

        # Check that the bbox is loaded properly
        bbox = s.bbox
        self.assertTrue(np.all(bbox.shape == (1, 6)))
        self.assertEqual(bbox[0, 0], -2.5)
        self.assertEqual(bbox[0, 1], -2.5)
        self.assertEqual(bbox[0, 2], -0.5)
        self.assertEqual(bbox[0, 3], 2.5)
        self.assertEqual(bbox[0, 4], 2.5)
        self.assertEqual(bbox[0, 5], 0.5)

        # Check that the neighboring indices
        n_indices = s._get_neighbor_idxs(0, 4)
        self.assertTrue(np.all(n_indices == [1, 2, 3, 4]))
        n_indices = s._get_neighbor_idxs(1, 4)
        self.assertTrue(np.all(n_indices == [0, 2, 3, 4]))
        n_indices = s._get_neighbor_idxs(35, 4)
        self.assertTrue(np.all(n_indices == [33, 34, 36, 37]))
        n_indices = s._get_neighbor_idxs(50, 4)
        self.assertTrue(np.all(n_indices == [46, 47, 48, 49]))
    
    def test_patches_from_points(self):
        # Generate a random dataset that has the Restrepo format
        dataset_directory = "./restrepo_mock_dataset/scene_1"
        s = scene.RestrepoScene(dataset_directory)

        img_idx = np.random.choice(s.n_images)
        images = s.get_image_with_neighbors(img_idx)
        # Randomly select a pixel from the reference image
        ref_img = images[0]

        patch_shape = (11, 11, 3)
        views = 5
        depth_planes = 32
        generation_params = GenerationParameters(
            patch_shape=patch_shape,
            depth_planes=depth_planes,
            neighbors=views-1
        )
        while True:
            flag = False
            px, py, _ = ref_img.random_pixel()[:, 0]
            print px, py
            points =\
                SamplingInBboxScheme(generation_params).sample_points_across_ray(
                    s,
                    img_idx,
                    py,
                    px
                )
            # Compute the patches using the patches_from_3d_points function
            X = np.empty(shape=(views, depth_planes)+patch_shape, dtype=np.float32)
            for im_idx, im in enumerate(images):
                patches = im.patches_from_3d_ponts(points, patch_shape[:2])
                if patches is None:
                    flag = True
                    break
                X[im_idx] = np.array(patches).reshape((depth_planes,) + patch_shape)

            if flag:
                continue
            
            # Compute the patches using the patches_from_point function
            X2 = np.empty(shape=(views, depth_planes)+patch_shape, dtype=np.float32)
            for idx, p in enumerate(points):
                patches = SceneTest.compute_patches_from_point(images, p)
                if patches is None:
                    flag = True
                    break
                X2[:, idx] = np.array(patches).reshape((views,) + patch_shape)

            if flag:
                continue

            # if you made it until here we are done
            break

        self.assertTrue(np.all(X.shape == X2.shape))
        for i in range(views):
            self.assertTrue(np.all(X[i] == X2[i]))

if __name__ == '__main__':
    unittest.main()
