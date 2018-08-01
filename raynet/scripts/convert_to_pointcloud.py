#!/usr/bin/env python
"""Transform the input depth maps to a pointcloud"""
import argparse
import os
import sys

import numpy as np

from ..metrics import VoxelMask, FiltersFactory
from ..pointcloud import get_pointcloud

from .arguments import add_dataset_related_arguments, build_dataset,\
    add_metrics_related_arguments
from .slicing import frame_idxs_type


def build_filter_factory(scene, min_distance, output_directory=None):
    filters = []
    if scene.observation_mask is not None:
        filters.append(
            VoxelMask(scene.bbox, scene.observation_mask, output_directory)
        )

    if min_distance != -1:
        filters.append(ReduceDensity(min_distance, output_directory))

    return FiltersFactory(filters)


def find_format(input_directory, key, idx):
    pattern = "_".join([key, "%d.npy" % (idx)])
    if os.path.isfile(os.path.join(input_directory, pattern)):
        return "_".join([key, "%d.npy"])
    else:
        return "_".join([key, "%03d.npy"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute the 3D reconstruction metrics"
    )
    parser.add_argument(
        "dataset_directory",
        help="The dataset to load"
    )
    parser.add_argument(
        "predictions_directory",
        help="The directory containing the model's predictions"
    )
    parser.add_argument(
        "output_directory",
        help="The directory to save the predicted point cloud"
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        default=0,
        help="Choose the scene to compute the metrics for"
    )
    parser.add_argument(
        "--frame_idxs",
        type=frame_idxs_type,
        default=":",
        help=("Choose the frames that correspond to the ordered prediction "
              "files")
    )
    parser.add_argument(
        "--pred_suffix",
        default="depth",
        help="The suffix for the predicted files (default=depth)"
    )

    add_dataset_related_arguments(parser)
    add_metrics_related_arguments(parser)
    args = parser.parse_args(argv)

    dataset = build_dataset(
        args.dataset_type,
        args.dataset_directory,
        args.illumination_condition
    )

    # Get the scene and then get the frame idxs
    scene = dataset.get_scene(args.scene_idx)
    frame_idxs = np.arange(scene.n_images)[args.frame_idxs]

    # Create the filter factory
    filter_factory = build_filter_factory(
        scene,
        args.min_distance,
        output_directory=args.output_directory
    )
    # Create the predicted point cloud
    format = find_format(
        args.predictions_directory,
        args.pred_suffix,
        frame_idxs[0]
    )
    depthmaps = [
        os.path.join(args.predictions_directory, format % (i,))
        for i in frame_idxs
    ]
    predicted_pointcloud = get_pointcloud(
        scene,
        frame_idxs,
        depthmaps,
        args.with_consistency_check,
        borders=args.borders,
        consistency_threshold=args.consistency_threshold,
        n_neighbors=args.n_neighbors
    )
    print "Saving predicted point-cloud for scene %d ..." % (args.scene_idx,)
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    predicted_pointcloud.save_ply(
        os.path.join(
            args.output_directory,
            "predicted_pc_s_%d.ply" % (args.scene_idx,)
        )
    )
    if filter_factory.has_filters:
        predicted_pointcloud.filter(filter_factory)

        predicted_pointcloud.save_ply(
            os.path.join(
                args.output_directory,
                "filtered_predicted_pc_s_%d.ply" % (args.scene_idx,)
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])
