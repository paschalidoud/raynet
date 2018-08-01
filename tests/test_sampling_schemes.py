import os
import unittest

import numpy as np

from raynet.common.generation_parameters import GenerationParameters
from raynet.common.dataset import RestrepoDataset, DTUDataset
from raynet.common.sampling_schemes import get_sampling_scheme
from raynet.train_network.sample import DefaultSampleGenerator
from raynet.utils.training_utils import dirac_distribution
from raynet.utils.geometry import project, is_collinear, is_between_simple


class SamplingSchemeTest(unittest.TestCase):
    """Contains tests for the various sampling schemes that are implemented
    """
    @staticmethod
    def default_input_output_shape(generation_params):
        neighbors = generation_params.neighbors
        depth_planes = generation_params.depth_planes
        patch_shape = generation_params.patch_shape
    
        # Find the number of image pairs given the number of the
        # adjacent images
        N = neighbors * (neighbors + 1) / 2
        dims = (depth_planes, N) + patch_shape
    
        input_shapes = [dims]*2
        output_shape = [(depth_planes,)]
    
        return input_shapes, output_shape

    def test_collinearity(self):
        """The sampled 3D points along a ray should be collinear with the
        camera center for the sample_in_bbox, sample_in_range and
        sample_in_disparity sampling methods.
        """
        print "Collinearity..."
        # Specify the dataset
        dataset = RestrepoDataset("./restrepo_mock_dataset")
        # Create a GenerationParameters object
        generation_params = GenerationParameters(
            depth_range=(3.0, 7.0),
            target_distribution_factory=dirac_distribution
        )
        # Specify the dimensiolity of the input and output
        input_shapes, output_shape = SamplingSchemeTest.default_input_output_shape(
            generation_params
        )

        sampling_policies = [
            "sample_in_bbox",
            "sample_in_range",
            "sample_in_disparity"
        ]
        for sampling_policy in sampling_policies:
            print sampling_policy
            while True:
                ss_factory = get_sampling_scheme(sampling_policy)(generation_params)
                sample_generator = DefaultSampleGenerator(
                    ss_factory,
                    generation_params,
                    (0,),
                    input_shapes,
                    output_shape
                )

                sample = sample_generator.get_sample(dataset)
                print "Scene:%d" %(sample.scene_idx,)
                scene = dataset.get_scene(sample.scene_idx)
                images = scene.get_image_with_neighbors(sample.img_idx)
                print "Image:%d" %(sample.img_idx)
                # Visualize the pixel in all neighboring images and make sure that the
                # results are resonable
                points = sample.points
                camera_center = images[0].camera.center
                if points is not None:
                    p_near = points[0, :-1].reshape(-1, 1)
                    p_far = points[-1, :-1].reshape(-1, 1)
                    self.assertTrue(is_collinear(p_near, p_far, camera_center[:-1]))
                    break

    def test_projected_points_on_reference_image(self):
        """The sampled 3D points along a ray should always be projected to the
        same pixel on the reference image.
        """
        print "Projection on reference image..."
        # Specify the dataset
        dataset = RestrepoDataset("./restrepo_mock_dataset")
        # Create a GenerationParameters object
        generation_params = GenerationParameters(
            depth_range=(3.0, 7.0),
            target_distribution_factory=dirac_distribution
        )
        # Specify the dimensiolity of the input and output
        input_shapes, output_shape = SamplingSchemeTest.default_input_output_shape(
            generation_params
        )

        sampling_policies = [
            "sample_in_bbox",
            "sample_in_range",
            "sample_in_disparity"
        ]
        for sampling_policy in sampling_policies:
            print sampling_policy
            while True:
                ss_factory = get_sampling_scheme(sampling_policy)(generation_params)
                sample_generator = DefaultSampleGenerator(
                    ss_factory,
                    generation_params,
                    (0,),
                    input_shapes,
                    output_shape
                )

                sample = sample_generator.get_sample(dataset)
                print "Scene:%d" %(sample.scene_idx,)
                scene = dataset.get_scene(sample.scene_idx)
                images = scene.get_image_with_neighbors(sample.img_idx)
                print "Image:%d" %(sample.img_idx)
                # Visualize the pixel in all neighboring images and make sure that the
                # results are resonable
                points = sample.points
                if points is not None:
                    pixels = np.round(project(images[0].camera.P, points.T))
                    for p in pixels:
                        self.assertTrue(
                            np.allclose(p[:-1], np.array([sample.patch_x, sample.patch_y]))
                        )
                    break
    
    def test_inside_bbox(self):
        """The sampled 3D points along a ray should be inside a bounding box
        that encloses a scene.
        """
        print "Inside bbox..."
        # Specify the dataset
        dataset = RestrepoDataset("./restrepo_mock_dataset")
        # Create a GenerationParameters object
        generation_params = GenerationParameters(
            depth_range=(3.0, 7.0),
            target_distribution_factory=dirac_distribution
        )
        # Specify the dimensiolity of the input and output
        input_shapes, output_shape = SamplingSchemeTest.default_input_output_shape(
            generation_params
        )

        sampling_policies = [
            "sample_in_bbox",
            "sample_in_voxel_space"
        ]
        for sampling_policy in sampling_policies:
            print sampling_policy
            while True:
                ss_factory = get_sampling_scheme(sampling_policy)(generation_params)
                sample_generator = DefaultSampleGenerator(
                    ss_factory,
                    generation_params,
                    (0,),
                    input_shapes,
                    output_shape
                )

                sample = sample_generator.get_sample(dataset)
                print "Scene:%d" %(sample.scene_idx,)
                scene = dataset.get_scene(sample.scene_idx)
                bbox = scene.bbox
                images = scene.get_image_with_neighbors(sample.img_idx)
                print "Image:%d" %(sample.img_idx)
                # Visualize the pixel in all neighboring images and make sure that the
                # results are resonable
                points = sample.points
                if points is not None:
                    p_near = points[0, :-1].reshape(-1, 1)
                    p_far = points[-1, :-1].reshape(-1, 1)
                    b_start = bbox[0, :3].reshape(-1, 1)
                    b_end = bbox[0, 3:].reshape(-1, 1)
                    self.assertTrue(is_between_simple( b_start, b_end, p_near))
                    self.assertTrue(is_between_simple(b_start, b_end, p_far))
                    break


if __name__ == '__main__':
    random_state = 10
    # First set the random state
    prng_state = np.random.get_state()
    np.random.seed(random_state)

    unittest.main()
