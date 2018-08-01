#!/usr/bin/env python2.7
"""Script used to perform the end-to-end training of the RayNet
"""
import argparse
import os
import sys

import json
import numpy as np

from ..common.generation_parameters import GenerationParameters
from ..common.sampling_schemes import get_sampling_scheme
from ..models import get_nn, cnn_factory, kernel_regularizer_factory
from ..tf_implementations.forward_backward_pass import\
    build_end_to_end_training
from ..train_network.raynet_batch_provider import\
    SingleThreadRayNetBatchProvider
from ..train_network.sample import RayNetSampleGenerator,\
    RayNetRandomSampleGenerator

from .arguments import add_nn_arguments, add_generation_arguments,\
    get_input_output_shapes, build_dataset, add_dataset_related_arguments,\
    has_valid_argument


def experiment_tag(args):
    lr = args.lr
    optimizer = args.optimizer
    loss = args.loss

    if args.train_with_gamma:
        gamma = (args.initial_gamma,) + args.gamma_range
    else:
        gamma = args.initial_gamma

    return "experiment_lr-%r_optimizer-%s_loss-%s_gamma_%r" % (
        lr, optimizer, loss, gamma)


def get_number_of_neighboring_rays_test_set(
    dataset_type,
    n_test_samples,
    window=4
):
    if dataset_type == "dtu":
        # return (n_test_samples / (49 - window)) + 1
        return 25
    elif dataset_type == "restrepo":
        # The Restrepo scenes contain roughly 180 images
        return n_test_samples / 180


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the RayNet in an end-to-end fashion"
    )

    parser.add_argument(
        "training_directory",
        help="Path to the folder containing the training set"
    )
    parser.add_argument(
        "testing_directory",
        help="Path to the folder containing the test set"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "weight_file",
        help="The initial weights file from the pre-trained model"
    )
    parser.add_argument(
        "train_test_scenes_range",
        default="./restrepo_train_test_splits.json",
        help="Path to the file containing the train-test splits"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100000,
        help="Number of updates (default=100000)"
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=10,
        help="Validate every updates (default=10)"
    )
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=100,
        help="Save a model every snapshot_every iteration (default=100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help=("The batch size, number of rays to be used for training "
              "(default=1000)")
    )
    parser.add_argument(
        "--repeat_from_neighboring_views",
        type=int,
        default=10,
        help=("How many times to select batches from neighboring views "
              "(default=10)")
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help=("Number of neighboring views from which to select batches "
              "(default=4)")
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=1000,
        help=("Number of ranodmly sampled rays from all dataset used for "
              "testing")
    )
    parser.add_argument(
        "--input_output_dimensionality",
        choices=["default", "hartmann"],
        default="default",
        help="The dimensionality of the input/output data for Keras"
    )

    parser.add_argument(
        "--initial_gamma",
        type=float,
        default=0.031,
        help="Initial value for gamma prior"
    )
    parser.add_argument(
        "--gamma_range",
        type=lambda x: tuple(map(float, x.split(","))),
        default="1e-3,0.99",
        help=("The allowed values of gamma during training "
              " (default=(1e-3, 0.99)")
    )
    parser.add_argument(
        "--train_with_gamma",
        action="store_true",
        help="When select also learn the gamma prior"
    )

    add_nn_arguments(parser)
    add_generation_arguments(parser)
    add_dataset_related_arguments(parser)
    args = parser.parse_args(argv)

    # Sanity checks to make sure that the input arguments have the correct
    # values
    has_valid_argument(args.network_architecture, ["simple_cnn"])

    # Check if output directory exists and if not create it
    output_directory = os.path.join(args.output_directory, experiment_tag(args))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create a GenerationParameters object
    generation_params = GenerationParameters.from_options(args)
    # Create the sampling scheme factory based on the input argument
    ss_factory = get_sampling_scheme(args.sampling_policy)(generation_params)

    # Based on the input argument specify the dataset to be used
    training_dataset = build_dataset(
        args.dataset_type,
        args.training_directory,
        args.illumination_condition,
        args.select_neighbors_based_on
    )
    testing_dataset = build_dataset(
        args.dataset_type,
        args.testing_directory,
        args.illumination_condition,
        args.select_neighbors_based_on
    )

    # Build the NN model that will be used for training
    input_shapes, output_shape = get_input_output_shapes(
        args.input_output_dimensionality
    )(generation_params)
    model = get_nn(args.network_architecture)(
        input_shapes[0],
        cnn_factory(args.cnn_factory),
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        clipnorm=0.1,
        loss=args.loss,
        reducer=args.reducer,
        merge_layer=args.merge_layer,
        weight_decay=kernel_regularizer_factory(args.weight_decay),
        weight_file=args.weight_file
    )

    # Create two sample generators for training and testing
    #train_sg = RayNetSampleGenerator(
    #    ss_factory,
    #    generation_params,
    #    tuple(json.load(open(args.train_test_scenes_range))["train"]),
    #    input_shapes,
    #    output_shape,
    #    n_rays=args.batch_size*args.repeat_from_neighboring_views,
    #    window=args.window
    #)
    train_sg = RayNetRandomSampleGenerator(
        ss_factory,
        generation_params,
        tuple(json.load(open(args.train_test_scenes_range))["train"]),
        input_shapes,
        output_shape,
        n_rays=args.n_test_samples,
        window=args.window
    )
    # NOTE: The test set will contain data just from a single scene. This
    # probably needs to be changed
    test_sg = RayNetSampleGenerator(
        ss_factory,
        generation_params,
        tuple(json.load(open(args.train_test_scenes_range))["test"]),
        input_shapes,
        output_shape,
        n_rays=get_number_of_neighboring_rays_test_set(
            args.dataset_type,
            args.n_test_samples
        )
    )
    # Create the test set
    test_bp = SingleThreadRayNetBatchProvider(
        testing_dataset,
        test_sg,
        args.n_test_samples
    )
    test_set = test_bp.get_batch_of_rays(generation_params)

    train_bp = SingleThreadRayNetBatchProvider(
        training_dataset,
        train_sg,
        args.batch_size
    )

    # Create the function that will be used for training
    _train_end_to_end_tf, _forward_pass_tf = build_end_to_end_training(
        model,
        args.batch_size,  # Number of rays
        args.neighbors + 1,  # Number of views including reference
        args.maximum_number_of_marched_voxels,
        args.depth_planes,
        args.grid_shape,
        args.patch_shape,
        args.initial_gamma,
        args.gamma_range,
        args.train_with_gamma,
        args.loss
    )

    # Create files to keep training and validation statistics
    val_f = open(os.path.join(output_directory, "val_loss.txt"), "w")
    train_f = open(
        os.path.join(output_directory, "train_statistics.txt"), "w"
    )
    train_f.write("scene_idx max_img_idx loss gamma\n")

    it = 0  # Keep track of the iterations
    m_cnt = 0  # Keep track of the saved models
    while it < args.iterations:
        loss, gamma = _train_end_to_end_tf(
            train_bp.get_batch_of_rays(generation_params)
        )
        # Get the scene_idx for the current scene
        scene_idx = train_bp.t_scene_idx[0]
        img_idxs = set(train_bp.t_image_idxs)
        train_f.write(" ".join([str(scene_idx), str(max(img_idxs)), str(loss), str(gamma)]))
        train_f.write("\n")
        train_f.flush()
        if it % args.validate_every == 0:
            val_loss = _forward_pass_tf(test_set)
            print "Validation loss in %d iteration: %f - gamma:%f" % (
                it, val_loss[0], gamma
            )
            val_f.write(str(val_loss[0]))
            val_f.write("\n")
            val_f.flush()

        if it % args.snapshot_every == 0:
            model.save_weights(os.path.join(
                output_directory,
                "weights.%d.hdf5" % (m_cnt,)
            ))
            # Update counter that keeps track of the saved models
            m_cnt += 1

        it += 1
    train_f.close()
    val_f.close()

if __name__ == "__main__":
    main(sys.argv[1:])
