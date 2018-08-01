"""This script contains various functions used for visualizations
"""
import matplotlib
matplotlib.use("Agg")
from itertools import combinations
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tempfile


def visualize_image(image, output_name=None):
    if output_name is None:
        tf = tempfile.NamedTemporaryFile()
        output_name = tf.name

    fig = plt.figure()
    #plt.xlim([0, image.shape[1]])
    #plt.ylim([0, image.shape[0]])
    if image.shape[2] == 1:
        plt.imshow(image[:, :, 0], cmap=cm.gray)
    else:
        plt.imshow(image)
    plt.savefig(output_name)
    plt.close()


def visualize_patch_center(image, patch_center, output_name=None):
    if output_name is None:
        tf = tempfile.NamedTemporaryFile()
        output_name = tf.name

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = plt.gca()
    if image.shape[2] == 1:
        im = plt.imshow(image[:, :, 0], cmap=cm.gray)
    else:
        im = plt.imshow(image)
    plt.axis("off")
    t = np.arange(len(patch_center))
    plt.scatter(patch_center[:, 0], patch_center[:, 1], s = 3)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="2.5%", pad=0.05);
    # plt.colorbar(im, cax=cax)
    plt.savefig(output_name)
    plt.close()


def plot_target_distribution(y, output_name=None):
    if output_name is None:
        tf = tempfile.NamedTemporaryFile()
        output_name = tf.name

    fig = plt.figure()
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.savefig(output_name)
    plt.close()


def plot_batch_distribution(y_true, y_pred, suffix):
    assert len(y_true) == len(y_pred), "y_true has different size from y_pred"
    d = np.arange(len(y_true))

    fig = plt.figure(figsize=(50, 15))
    for i in xrange(32):
        plt.subplot(4, 8, i+1)
        plt.plot(d, y_true[i], "k", label="y_true")
        plt.plot(d, y_pred[i], "b", label="y_pred")
        plt.legend()
    plt.savefig("/tmp/batch_predictions_" + str(suffix) + ".png")


def plot_depth_maps_from_single_depth(
    X1,
    X2,
    d,
    output_folder,
    views=5,
    idxs_1=None,
    idxs_2=None
):
    if idxs_1 is None or idxs_2 is None:
        idxs_1 = []
        idxs_2 = []
        for x1, x2 in combinations(xrange(views), 2):
            idxs_1.append(x1)
            idxs_2.append(x2)

    fig = plt.figure(figsize=(16, 10)) 
    fig.suptitle("%d Depth plane" % (d, ))
    cnt = 0
    for i, v in enumerate(xrange(20)):
        if cnt == 10:
            cnt = 0
        v = v + 1
        ax1 = plt.subplot(2, 10, v)
        if i < 10:
          ax1.imshow(X1[d, cnt])
          ax1.set_title("%d view" % (idxs_1[cnt], ))
        elif i >= 10:
            ax1.imshow(X2[d, cnt])
            ax1.set_title("%d view" % (idxs_2[cnt],))
        cnt += 1

    plt.savefig(os.path.join(output_folder, "depth_%d.png") %(d,))


def plot_depth_maps(
    X1_file,
    X2_file,
    output_folder="/tmp",
    views=5,
    depth_planes=32
):
    X1 = np.load(X1_file)
    X2 = np.load(X2_file)

    idxs_1 = []
    idxs_2 = []
    for x1, x2 in combinations(xrange(views), 2):
        idxs_1.append(x1)
        idxs_2.append(x2)

    for i in xrange(depth_planes):
        plot_depth_maps_from_single_depth(
            X1,
            X2,
            i,
            output_folder,
            views=views,
            idxs_1=idxs_1,
            idxs_2=idxs_2
        )
