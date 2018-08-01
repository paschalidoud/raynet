#!/usr/bin/env python
"""Compute metrics for evaluating the 3D reconstruction"""

import argparse
import os
import sys

import numpy as np

from ..metrics import PerPixelMeanDepthError, Accuracy, Completeness,\
    VoxelMask, FiltersFactory, ReduceDensity
from ..pointcloud import Pointcloud, get_pointcloud

from .arguments import add_metrics_related_arguments, build_dataset,\
    add_dataset_related_arguments
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


def build_metric(
    metric_name,
    borders,
    filter_factory,
    truncate,
    use_pc_from_depthmap
):
    return {
        "ppmde": PerPixelMeanDepthError(borders),
        "accuracy": Accuracy(
            filter_factory,
            truncate,
            borders,
            use_pc_from_depthmap
        ),
        "completeness": Completeness(
            filter_factory,
            truncate,
            borders,
            use_pc_from_depthmap
         )
    }[metric_name]


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
        "metric",
        nargs="+",
        choices=["ppmde", "accuracy", "completeness"],
        help="Choose a metric"
    )
    parser.add_argument(
        "--output_directory",
        default="/tmp/",
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
        "--predicted_files_format",
        default="depth_%03d.npy",
        help="The format for the predicted file"
    )
    parser.add_argument(
        "--use_pc_from_depthmap",
        action="store_true",
        help=("Set when we want to estimate the gt point cloud from the depth "
              "images")
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

    # Create the predicted point cloud
    depthmaps = [
        os.path.join(
            args.predictions_directory,
            args.predicted_files_format % (i,)
        )
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

    # For every metric
    for mi in args.metric:
        metric = build_metric(
            mi,
            args.borders,
            build_filter_factory(
                scene,
                args.min_distance,
                output_directory=args.output_directory
            ),
            args.truncate,
            args.use_pc_from_depthmap
        )
        values, points = metric.compute(
            scene,
            frame_idxs,
            depthmaps,
            predicted_pointcloud
        )
        # Create and save a pointcloud with the filtered points, where each
        # point will be colored based on the value of the computed metric
        if points is None:
            pass
        else:
            Pointcloud(points).save_colored_ply(
                os.path.join(
                    args.output_directory,
                    "%s_pc.ply" % (mi)
                ),
                values
            )
        print mi, " mean: ", values.mean(), " median: ", np.median(values)


if __name__ == "__main__":
    main(sys.argv[1:])
