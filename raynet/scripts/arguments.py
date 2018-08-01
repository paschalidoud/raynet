import os

from ..common.dataset import RestrepoDataset, DTUDataset
from ..train_network.sample import DefaultSampleGenerator,\
    HartmannSampleGenerator, CompareWithReferenceSampleGenerator


def add_nn_arguments(parser):
    """Add arguments to a parser that are related with the NN computation.
    """
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default 1e-3)"
    )
    parser.add_argument(
        "--reducer",
        choices=["max", "average", "topK"],
        default="average",
        help=("The reducer to be used in order to compute the final "
              "similarity scores for each depth plane (defaut=average)")
    )
    parser.add_argument(
        "--merge_layer",
        choices=["dot-product", "cosine-similarity"],
        default="dot-product",
        help=("Merge layer used to compute the similarity between the features"
              " (default=dot-product)")
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Parameter used only for the topK reducer (default=5)"
    )
    parser.add_argument(
        "--optimizer",
        choices=["Adam", "SGD"],
        default="Adam",
        help="The optimizer to be used (default=Adam)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help=("Parameter used to update momentum in case of SGD optimizer"
              " (default=0.9)")
    )
    parser.add_argument(
        "--network_architecture",
        choices=[
            "simple_cnn",
            "simple_nn_for_training",
            "simple_nn_for_training_voxel_space",
            "hartmann"
        ],
        default="simple_nn_for_training",
        help="The network architecture to be used"
    )
    parser.add_argument(
        "--cnn_factory",
        choices=[
            "simple_cnn",
            "simple_cnn_ln",
            "dilated_cnn_receptive_field_25",
            "dilated_cnn_receptive_field_25_with_tanh",
            "hartmann_cnn"
        ],
        default="simple_cnn",
        help="NN architecture to be used for the Multi-View CNN"
    )
    parser.add_argument(
        "--loss",
        choices=[
            "categorical_crossentropy",
            "emd",
            "squared_emd",
            "expected_squared_error"
        ],
        default="emd",
        help="Type of loss used for training (default emd)"
    )
    parser.add_argument(
        "--padding",
        default=None,
        type=int,
        help="Zero padding around images"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 regularizer factor used for weight decay"
    )


def add_training_arguments(parser):
    """Add arguments to a parser that are related with the training process.
    """
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs (default=150)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Total number of batches of samples (default=500)"
    )
    parser.add_argument(
        "--training_cached_samples",
        type=int,
        default=500,
        help="Number of samples kept in cache (default=500)"
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=500,
        help="Number of samples used in the validation set (default=500)"
    )

    parser.add_argument(
        "--lr_epochs",
        type=lambda x: map(int, x.split(",")),
        default="50,80,100,120",
        help="Training epochs with diminishing learning rate"
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=None,
        help=("Factor according to which the learning rate will be diminished"
              " (default=None)")
    )


def add_generation_arguments(parser):
    """Add arguments to a parser that are related with the sample generation
    process.
    """
    parser.add_argument(
        "--patch_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="11,11,3",
        help="The shape of a patch (default=(11,11,3)"
    )
    parser.add_argument(
        "--depth_planes",
        type=int,
        default=32,
        help="Number of depth planes to be used (default=32)"
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=4,
        help=("Number of consecutive frames used to compute the pairs"
              " (default=4)")
    )
    parser.add_argument(
        "--target_distribution_factory",
        choices=["dirac", "guassian"],
        default="dirac",
        help="The type of the final depth distribution (default=dirac)"
    )
    parser.add_argument(
        "--stddev_factor",
        type=float,
        default=1.0,
        help=("Initial standard deviation for the gaussian depth distribution"
              " (default=1.0)")
    )
    parser.add_argument(
        "--std_is_distance",
        action="store_true",
        help=("When set the std is semantically equal to the distance between"
              " the minimum and maximum point in the ray, while otherwise it "
              "is the square of their corresponding distances")
    )
    parser.add_argument(
        "--expand_patch",
        action="store_true",
        help=("When set return also patches that are partially or entirely"
              " black if the point is outside the image boundaries")
    )
    parser.add_argument(
        "--sampling_policy",
        choices=[
            "sample_in_disparity",
            "sample_in_bbox",
            "sample_in_range",
            "sample_in_voxel_space",
            "tf_sample_in_bbox",
            "tf_sample_in_range",
            "full_tf_sample_in_bbox",
            "full_tf_sample_in_range"
        ],
        default="sample_in_bbox",
        help=("The sampling generation policy to compute the depth planes"
              " (default=sample_in_bbox)")
    )
    parser.add_argument(
        "--depth_range",
        type=lambda x: tuple(map(float, x.split(","))),
        default="3.0,7.0",
        help="The depth range used when sampling planes in range"
    )
    parser.add_argument(
        "--grid_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256,128",
        help="The number of voxels per axis (default=(256, 256, 128)"
    )
    parser.add_argument(
        "--maximum_number_of_marched_voxels",
        type=int,
        default=650,
        help=("The maximum number of voxels through which a ray can intersect"
              "(default=650)")
    )


def add_experiments_related_arguments(parser):
    parser.add_argument(
        "--training_set_name",
        default="BH",
        help="The name of the training set"
    )
    parser.add_argument(
        "--test_set_name",
        default="Downtown",
        help="The name of the test set"
    )
    parser.add_argument(
        "--credentials",
        default=os.path.join(os.path.dirname(__file__), ".credentials"),
        help="The credentials file for the Google API"
    )
    parser.add_argument(
        "--spreadsheet",
        default="Sheet1",
        help="The spreadsheet to save the results"
    )


def add_hartmann_related_arguments(parser):
    parser.add_argument(
        "--step_depth",
        default=15,
        type=int,
        help="Number of depth planes to skip before and after the gt"
    )


def add_metrics_related_arguments(parser):
    parser.add_argument(
        "--borders",
        default=40,
        type=int,
        help="The number of pixels to drop from the borders of the image"
    )
    parser.add_argument(
        "--truncate",
        default=float("inf"),
        type=float,
        help="Truncate all distances to this number if their larger"
    )
    parser.add_argument(
        "--min_distance",
        default=-1,
        type=float,
        help="Reduce the density of the points based on this distance"
    )
    parser.add_argument(
        "--consistency_threshold",
        default=0.75,
        type=float,
        help=("The value of the consistency level imposed when converting "
              " depth maps to point-clouds (default=0.75)")
    )
    parser.add_argument(
        "--n_neighbors",
        default=5,
        type=int,
        help=("Number of views considered during the consistency check "
              " (default=5)")
    )
    parser.add_argument(
        "--with_consistency_check",
        action="store_true",
        help=("When set transform depth maps to point cloud using a "
              " consistency check")
    )


def add_dataset_related_arguments(parser):
    parser.add_argument(
        "--select_neighbors_based_on",
        choices=[
            "filesystem",
            "distance"
        ],
        default="filesystem",
        help="Illumination condition used only for DTU dataset"
    )
    parser.add_argument(
        "--illumination_condition",
        choices=[
            "max",
            "0_r5000",
            "1_r5000",
            "2_r5000",
            "3_r5000",
            "4_r5000",
            "5_r5000",
            "6_r5000"
        ],
        default="max",
        help="Illumination condition used only for DTU dataset"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["restrepo", "dtu"],
        default="restrepo",
        help="Choose the type of the dataset"
    )


def add_mrf_related_arguments(parser):
    parser.add_argument(
        "--initial_gamma_prior",
        type=float,
        default=0.05,
        help="The occupancy prior probabality per voxel (default=0.05)"
    )
    parser.add_argument(
        "--bp_iterations",
        type=int,
        default=3,
        help="Number of BP iterations to be used (default=3)"
    )


def add_indexing_related_arguments(parser):
    parser.add_argument(
        "--start_end",
        type=lambda x: tuple(map(int, x.split(","))),
        default="0,5",
        help="The starting and ending frame to be used (default=(0,5)"
    )
    parser.add_argument(
        "--skip_every",
        type=int,
        default=0,
        help="The number of frames to be skipped"
    )


def add_forward_pass_factory_related_arguments(parser):
    parser.add_argument(
        "--forward_pass_factory",
        choices=[
            "multi_view_cnn",
            "multi_view_cnn_voxel_space",
            "hartmann_fp",
            "raynet"
        ],
        default="full_multi_view_cnn_cuda_fp",
        help="Choose the forward pass factory"
    )
    parser.add_argument(
        "--rays_batch",
        type=int,
        default=130000,
        help="Number of rays per batch (default=130000)"
    )


def get_actual_sampling_policy(name):
    if "sample_in_bbox" in name:
        return "sample_in_bbox"
    elif "sample_in_range" in name:
        return "sample_in_range"
    else:
        raise NotImplementedError("Not implemented yet")


def has_valid_argument(argument, valid_options):
    if argument not in valid_options:
        raise ValueError(
            "Expected argument %r but received %r" % (valid_options, argument)
        )
    else:
        pass


def get_input_output_shapes(name):
    return {
        "default": default_input_output_shape,
        "hartmann": hartmann_input_output_shape,
        "reference_wrt_others": reference_wrt_others_input_output_shape
    }[name]


def get_sample_generator(name):
    return {
        "default": DefaultSampleGenerator,
        "hartmann": HartmannSampleGenerator,
        "reference_wrt_others": CompareWithReferenceSampleGenerator
    }[name]


def default_input_output_shape(generation_params):
    neighbors = generation_params.neighbors
    depth_planes = generation_params.depth_planes
    patch_shape = generation_params.patch_shape

    # Find the number of image pairs given the number of the
    # adjacent images
    N = neighbors * (neighbors + 1) / 2
    dims = (depth_planes, N) + patch_shape

    return [dims]*2, [(depth_planes,)]


def hartmann_input_output_shape(generation_params):
    neighbors = generation_params.neighbors
    patch_shape = generation_params.patch_shape

    return [patch_shape]*(neighbors + 1), [(1, 1, 2)]


def reference_wrt_others_input_output_shape(generation_params):
    views = generation_params.neighbors
    depth_planes = generation_params.depth_planes
    patch_shape = generation_params.patch_shape

    # Find the number of image pairs given the number of the
    # adjacent images
    dims = (depth_planes, views) + patch_shape
    return [dims]*2, [(depth_planes,)]


def build_dataset(
    type,
    dir,
    illumination_condition,
    select_neighbors_based_on="filesystem"
):
    if type.lower() == "dtu":
        return DTUDataset(
            dir,
            illumination_condition,
            select_neighbors_based_on=select_neighbors_based_on
        )
    else:
        return RestrepoDataset(
            dir,
            select_neighbors_based_on=select_neighbors_based_on
        )
