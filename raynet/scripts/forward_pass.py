#!/usr/bin/env python
"""Script used to perform a forward pass given a pretrained Keras model.
"""
import argparse
import os
import sys

import keras.backend as K
import numpy as np

try:
    import pycuda
except ImportError:
    print "******PyCUDA is required for the forward pass******"
    sys.exit(0)

from ..common.generation_parameters import GenerationParameters
from ..common.sampling_schemes import get_sampling_scheme
from ..forward_pass import get_forward_pass_factory
from ..models import get_nn, cnn_factory, kernel_regularizer_factory

from .arguments import add_nn_arguments, add_generation_arguments,\
    get_input_output_shapes, add_dataset_related_arguments, build_dataset,\
    has_valid_argument, add_forward_pass_factory_related_arguments,\
    get_actual_sampling_policy, add_indexing_related_arguments,\
    add_mrf_related_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=("Do a forward pass and estimate the per pixel depth "
                     " distribution for an image in a scene")
    )
    parser.add_argument(
        "dataset_directory",
        help="Directory containing the input data"
    )
    parser.add_argument(
        "output_directory",
        help="Directory to save the output data"
    )
    parser.add_argument(
        "--weight_file",
        help="The path to the file conatining the Keras model to be used"
    )

    parser.add_argument(
        "--scene_idx",
        default=1,
        type=int,
        help="Some datasets are also split in scenes"
    )
    parser.add_argument(
        "--input_output_dimensionality",
        choices=["default", "hartmann"],
        default="default",
        help="The dimensionality of the input/output data for Keras"
    )
    parser.add_argument(
        "--filter_out",
        action="store_true",
        help="Filter out rays with zero ground-truth"
    )

    # Add additional arguments in the argument parser
    add_generation_arguments(parser)
    add_dataset_related_arguments(parser)
    add_indexing_related_arguments(parser)
    add_nn_arguments(parser)
    add_forward_pass_factory_related_arguments(parser)
    add_mrf_related_arguments(parser)
    args = parser.parse_args(argv)

    # Make sure that tensorflow won't allocate more memory than what is
    # actually needs
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=config))

    # Check if output directory exists and if not create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Sanity checks to make sure that the input arguments have the correct
    # values
    has_valid_argument(args.network_architecture, ["simple_cnn", "hartmann"])

    # Create a GenerationParameters object
    generation_params = GenerationParameters.from_options(args)
    # Create the sampling scheme factory based on the input argument
    ss_factory = get_sampling_scheme(args.sampling_policy)(generation_params)

    # Based on the input argument specify the dataset to be used
    dataset = build_dataset(
        args.dataset_type,
        args.dataset_directory,
        args.illumination_condition,
        args.select_neighbors_based_on
    )

    # Based on the scene index create a Scene object
    scene = dataset.get_scene(args.scene_idx)

    # Build the NN model that will be used to export the features
    input_shapes, output_shape = get_input_output_shapes(
        args.input_output_dimensionality
    )(generation_params)
    model = get_nn(args.network_architecture)(
        input_shapes[0],
        cnn_factory(args.cnn_factory),
        optimizer=args.optimizer,
        lr=args.lr,
        loss=args.loss,
        reducer=args.reducer,
        merge_layer=args.merge_layer,
        weight_decay=kernel_regularizer_factory(args.weight_decay),
        weight_file=args.weight_file
    )

    fp = get_forward_pass_factory(args.forward_pass_factory)(
        model,
        generation_params,
        get_actual_sampling_policy(args.sampling_policy),
        scene.image_shape,
        args.rays_batch
    )

    start_img_idx = args.start_end[0]
    end_img_idx = args.start_end[1]

    depth_images = fp.forward_pass(
        scene,
        (start_img_idx, end_img_idx, args.skip_every+1)
    )
    ref_idx = start_img_idx
    for S in depth_images:
        # Save the per-pixel depth distribution
        np.save(
            os.path.join(args.output_directory, "depth_%03d.npy" % (ref_idx,)),
            S
        )
        ref_idx += args.skip_every + 1


if __name__ == "__main__":
    main(sys.argv[1:])
