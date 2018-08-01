#!/usr/bin/env python
"""Script used to pretrain the network that will be used to predict the
per-pixel depth distribution for an image given a set of images from different
views.
"""
import argparse
import os
import sys

import json
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import numpy as np

from ..common.generation_parameters import GenerationParameters
from ..common.sampling_schemes import get_sampling_scheme
from ..models import get_nn, cnn_factory, kernel_regularizer_factory
from ..train_network.batch_provider import BatchProvider

from .arguments import add_nn_arguments, add_training_arguments,\
    add_generation_arguments, add_experiments_related_arguments,\
    add_hartmann_related_arguments, get_input_output_shapes, build_dataset,\
    add_dataset_related_arguments, get_sample_generator
from .experiments_utils.experiments_manager import register_experiment,\
    set_output_directory, Metrics, save_experiment_locally


class MetricsHistory(Callback):
    def __init__(self, filepath_train, filepath_val):
        self.fd_t = open(filepath_train, "w")
        self.fd_v = open(filepath_val, "w")
        self.keys_t = []
        self.keys_v = []

    def _on_end(self, fd, keys, logs):
        if not keys:
            keys.extend(sorted(logs.keys()))
            print >>fd, " ".join(keys)

        print >>fd, " ".join(map(str, [logs[k] for k in keys]))
        fd.flush()

    def on_batch_end(self, batch, logs={}):
        self._on_end(self.fd_t, self.keys_t, logs)

    def on_epoch_end(self, epoch, logs={}):
        d = {"epoch": epoch}
        for k in logs:
            if k.startswith("val_"):
                d[k] = logs[k]
        self._on_end(self.fd_v, self.keys_v, d)


def lr_schedule(lr, factor, reductions):
    def inner(epoch):
        for i, e in enumerate(reductions):
            if epoch < e:
                return lr*factor**(-i)
        return lr*factor**(-len(reductions))

    return inner


def collect_test_set(
    test_dataset,
    test_sg,
    n_test_samples,
    input_shapes,
    output_shapes,
    batch_size=32,
    random_state=0
):
    # First set the random state
    prng_state = np.random.get_state()
    np.random.seed(random_state)
    print "Collecting test set..."

    # Specify the dimensionality of the output
    X = [
        np.empty((n_test_samples,) + shape, dtype=np.float32)
        for shape in input_shapes
    ]
    Y = [
        np.empty((n_test_samples,) + shape, dtype=np.float32)
        for shape in output_shapes
    ]

    bp = BatchProvider(
        test_dataset,
        test_sg,
        batch_size,
        cache_size=n_test_samples
    )
    bp.ready()

    cnt = 0
    while cnt < n_test_samples:
        # Get a batch
        x, y = next(bp)

        # For each input
        for Xi, xi in zip(X, x):
            # For each sample
            for j, xij in zip(range(cnt, n_test_samples), xi):
                Xi[j] = xij

        # For each output
        for Yi, yi in zip(Y, y):
            # For each sample
            for j, yij in zip(range(cnt, n_test_samples), yi):
                Yi[j] = yij

        # We either copied all of them or we 're done so add the whole batch to
        # the counter
        cnt += len(x[0])
    bp.stop()

    # We need to reset the random number generator to what it was before
    np.random.set_state(prng_state)

    # Now we 're done
    return [X, Y]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=("Train a network to predict the per-pixel depth value "
                     "for an image given a set of images from different views")
    )
    parser.add_argument(
        "training_directory",
        help="Path to the folder containing the training set"
    )
    parser.add_argument(
        "test_directory",
        help="Path to the folder containing the test set"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "train_test_scenes_range",
        help="Path to the file containing the train-test splits"
    )
    parser.add_argument(
        "--weight_file",
        help="An initial weights file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples in a batch"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_params",
        default="pretrain_network_experiment_params.txt",
        help=("Path to the file containing which parameters and "
              " with what order to be used")
    )
    parser.add_argument(
        "--input_output_dimensionality",
        choices=["default", "hartmann", "reference_wrt_others"],
        default="default",
        help="The dimensionality of the input/output data for Keras"
    )

    add_training_arguments(parser)
    add_nn_arguments(parser)
    add_generation_arguments(parser)
    add_hartmann_related_arguments(parser)
    add_experiments_related_arguments(parser)
    add_dataset_related_arguments(parser)
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    if K.backend() == "tensorflow":
        K.tf.set_random_seed(args.seed)

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
    test_dataset = build_dataset(
        args.dataset_type,
        args.test_directory,
        args.illumination_condition,
        args.select_neighbors_based_on
    )

    # Create a directory for the current experiment
    experiment_directory, experiment_tag =\
        set_output_directory(args.output_directory)
    # Keep track of the training and test evolution
    history = MetricsHistory(
        os.path.join(experiment_directory, "train.txt"),
        os.path.join(experiment_directory, "val.txt")
    )
    # Save the model weight's at the end of every epoch
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            experiment_directory,
            "weights",
            "weights.{epoch:02d}.hdf5"
        ),
        verbose=0
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
        loss=args.loss,
        reducer=args.reducer,
        merge_layer=args.merge_layer,
        weight_decay=kernel_regularizer_factory(
            args.weight_decay
        ),
        weight_file=args.weight_file
    )

    # Create two sample generators, one for the training and one for the test
    # set
    sample_generator = get_sample_generator(args.input_output_dimensionality)
    train_sg = sample_generator(
        ss_factory,
        generation_params,
        tuple(json.load(open(args.train_test_scenes_range))["train"]),
        input_shapes,
        output_shape
    )
    test_sg = sample_generator(
        ss_factory,
        generation_params,
        tuple(json.load(open(args.train_test_scenes_range))["test"]),
        input_shapes,
        output_shape
    )

    # Build to batch_providers, one for the validation set and one for the
    # training set
    test_set = collect_test_set(
        test_dataset,
        test_sg,
        args.n_test_samples,
        input_shapes,
        output_shape
    )

    print "Cache %d samples for training" % (args.training_cached_samples,)

    train_bp = BatchProvider(
        training_dataset,
        train_sg,
        args.batch_size,
        cache_size=args.training_cached_samples
    )

    callbacks = [history, checkpointer]
    # Add a callback in order to compute a diminished learning after some
    # epochs
    if args.lr_factor is not None:
        lrate_scheduler = LearningRateScheduler(
            lr_schedule(args.lr, args.lr_factor, args.lr_epochs)
        )
        callbacks.append(lrate_scheduler)

    try:
        train_bp.ready()
        model.fit_generator(
            train_bp,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            verbose=1,
            validation_data=test_set,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        pass
    train_bp.stop()

    metrics = Metrics.from_file(
        os.path.join(experiment_directory, "train.txt"),
        os.path.join(experiment_directory, "val.txt")
    ).to_list(args.steps_per_epoch)
    # Get the parameters and their ordering for the spreadsheet
    l = []
    with open(args.experiment_params) as f:
        l = f.readlines()
    params_ordering = [x.strip() for x in l]

    t = vars(args)
    params = {k: str(v) for k, v in t.iteritems()}
    # Same the parameters of the experiment
    with open(os.path.join(experiment_directory, "parameters.json"), "w") as f:
        json.dump(params, f)

    # Only register the experiment when a credentials file exists
    if os.path.exists(args.credentials):
        register_experiment(
            params_ordering,
            params,
            metrics,
            experiment_tag,
            args.credentials,
            sheet=args.spreadsheet
        )
    # Save the parameters and the results of the current experiment for future
    # use
    print "Save results before exiting in %s folder" % (experiment_directory)
    save_experiment_locally(
        params_ordering,
        params,
        metrics,
        experiment_tag,
        os.path.join(experiment_directory, "results.npy")
    )


if __name__ == "__main__":
    main(sys.argv[1:])
