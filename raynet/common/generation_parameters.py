import numpy as np

from raynet.utils.training_utils import dirac_distribution,\
    gaussian_distribution


def get_target_distribution_factory(
    depth_distribution_type,
    stddev_factor=1.0,
    std_is_distance=False
):
    if depth_distribution_type == "dirac":
        return dirac_distribution
    elif depth_distribution_type == "gaussian":
        return gaussian_distribution(stddev_factor, std_is_distance)
    else:
        raise NotImplementedError()


def get_sampling_type(name):
    if "bbox" in name:
        return "sample_points_in_bbox"
    elif "range" in name:
        return "sample_points_in_range"
    elif "disparity" in name:
        return "sample_points_in_disparity"
    elif "voxel_space" in name:
        return "sample_points_in_voxel_space"


class GenerationParameters(object):
    """This class is designed to hold all generation parameters
    """
    def __init__(
        self,
        depth_planes=32,
        neighbors=4,
        patch_shape=(11, 11, 3),
        grid_shape=np.array([64, 64, 32], dtype=np.int32),
        max_number_of_marched_voxels=400,
        expand_patch=True,
        target_distribution_factory=None,
        depth_range=None,
        step_depth=None,
        padding=None,
        sampling_type=None,
        gamma_mrf=None
    ):
        self.neighbors = neighbors
        self.patch_shape = patch_shape
        self.expand_patch = expand_patch
        self.depth_planes = depth_planes
        self.grid_shape = grid_shape
        self.depth_range = depth_range
        self.step_depth = step_depth
        self.padding = padding
        self.sampling_type = sampling_type

        self.target_distribution_factory = target_distribution_factory
        self.max_number_of_marched_voxels = max_number_of_marched_voxels
        self.gamma_mrf = gamma_mrf

    @classmethod
    def from_options(cls, argument_parser):
        # Make Namespace to dictionary to be able to use it
        args = vars(argument_parser)
        # Check if argument_parser contains the grid_shape argument
        grid_shape = args["grid_shape"] if "grid_shape" in args else None
        # Check if argument_parser contains the depth_range argument
        depth_range = args["depth_range"] if "depth_range" in args else None
        # Check if argument_parser contains the step_depth argument
        step_depth = args["step_depth"] if "step_depth" in args else None

        # Check if argument_parser contains the max_number_of_marched_voxels
        # argument
        mnofmv = args["maximum_number_of_marched_voxels"]
        max_number_of_marched_voxels =\
            mnofmv if "maximum_number_of_marched_voxels" in args else None

        # Get the padding value
        patch_shape =\
            args["patch_shape"] if "patch_shape" in args else (None,)*3
        padding =\
            args["padding"] if "padding" in args and args["padding"] is not None else patch_shape[0]

        neighbors = args["neighbors"] if "neighbors" in args else None
        depth_planes = args["depth_planes"] if "depth_planes" in args else None
        if "target_distribution_factory" in args:
            tdf = get_target_distribution_factory(
                argument_parser.target_distribution_factory,
                argument_parser.stddev_factor,
                argument_parser.std_is_distance
            )
        else:
            tdf = None

        # Check if argument_parser contains the sampling_policy argument
        try:
            sampling_type = get_sampling_type(argument_parser.sampling_policy)
        except AttributeError:
            sampling_type = None

        gamma_mrf =\
            args["initial_gamma_prior"] if "initial_gamma_prior" in args else None

        return cls(
            patch_shape=patch_shape,
            depth_planes=depth_planes,
            neighbors=neighbors,
            target_distribution_factory=tdf,
            grid_shape=grid_shape,
            max_number_of_marched_voxels=max_number_of_marched_voxels,
            depth_range=depth_range,
            step_depth=step_depth,
            padding=padding,
            sampling_type=sampling_type,
            gamma_mrf=gamma_mrf
        )
