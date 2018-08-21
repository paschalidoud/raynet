Code Walk-through
=================

The goal of this page is to briefly introduce you to our **raynet** library.
This page is mainly focused for those of you that want to have a better
understanding of the main modules implemented in our codebase in order to help
you familiarize with it.

## Using different datasets as input

We want to be able to deal with multiple datasets however, not all of them can
be parsed using the same format. To this end, we created the
[Dataset](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/common/dataset.py#L8)
class that handles various datasets in a generic manner. In principle, a dataset is defined
as a collection of scenes, however different datasets are represented with
different folder conventions. Until now, we have tested our implementation on
two challenging datasets,

* [Aerial dataset](https://www.sciencedirect.com/science/article/pii/S0924271614002354)
* [DTU Multi-view stereo benchmark](https://link.springer.com/article/10.1007/s11263-016-0902-9)

For example, the **Aerial Dataset** is represented by a directory that contains
one directory for every scene. Every scene is represented by a directory with
two inner directories containing the views and the camera poses. On the
contrary, the **DTU Dataset** is represented by a directory that contains three
subdirectories, one for the camera poses, one the raw images of every scene and
one for the ground-truth data of every scene. To this end, we created two
wrapper classes to the `Dataset` class to handle the corresponding dataset
types: `RestrepoDataset` and the `DTUDataset`. In order to create an instance
of a of `Dataset` class it simply suffices to specify two arguments,

* **dataset_directory**: The path to the folder containing the
dataset.
* **select_neighbors_based_on**: This argument is used to control how
neighbouring views are selected. We provide two options based either
on their geometrical distance or their order in the file system. They can be
selected with `distance` and `filesystem` respectively.

For instance one can create a `RestrepoDataset` object just by writing

```python
In [1]: from raynet.common.dataset import RestrepoDataset
In [2]: dataset = RestrepoDataset(
            "/path/to/folder/containing/the/probabilistic_reconstruction_data_downsampled/",
            "distance"
        )
```

Every dataset is defined as a set of scenes and every scene can be specified
with a unique scene index. Therefore, the API for the **Dataset** class is

```python
get_scene(scene_idx)
```

and it creates  a
[Scene](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/common/scene.py#L2)
object. For the **DTU Dataset** we use the provided scene indices, while for the
**Aerial Dataset** we map the scenes to indices based on their alphabetical
order.

A **Scene** is defined as a collection of raw images, camera poses,
ground-truth data as well as a bounding box that specifies the borders of the
scene. However, similar to the datasets, not all scenes are represented using
the same format. Therefore, we implement two wrappers, one for the scenes
following the format of the *Aerial Dataset* and one for the scenes following
the format of the *DTU Dataset*. Their API is the following,

* `get_image(i)`: Returns the \(i^{th}\) image of the current scene.
* `get_images()`: Returns a list of
[Image](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/common/image.py#L10)
objects one for every image of the current scene.
* `get_random_image()`: Returns an `Image` object for a random image of the
current scene.
* `get_image_with_neighbors(i, neighbors)`: Returns a list of `Image` objects,
where the first is the  \(i^{th}\) image and the rest are the neighbouring
views to this image. The neighbouring views are selected based on the
`select_neighbors_based_on` argument analysed before.
* `get_depth_for_pixel(i, y, x)`: Returns the ground-truth depth value of the
\((x, y)\) pixel of the \(i^{th}\) image.
* `get_depth_map(i)`: Returns the ground-truth depth map for the \(i^{th}\) image.
* `get_depthmaps()`: Returns a list with numpy arrays containing the
corresponding depth map for every image in the current scene.
* `get_pointcloud()`: Returns a
[Pointcloud](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/pointcloud.py#L14)
object containing the ground-truth point cloud of the current scene.

In case you want to use a dataset that follows a different format, you need to
implement a wrapper on the *Dataset* and on the *Scene* class based on your
requirements.

## Training different networks

Now that we have analysed how one can use different datasets as inputs it is
also worth mentioning how one can train different networks to perform the 3D
reconstruction task. We have built in various architectures that can be used to
extract similarity features between patches from different views. Part of the
code that defines those architectures is shown below. The full code can be
found
[here](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/models.py#L1).

* **simple_cnn**: Each layer comprises convolution, spatial batch normalization
and a ReLU non-linearity. We repeat this schemes 5 times but we remove the ReLU
from the last layer in order to retain information encoded both in the negative
and positive range. The receptive field of this architecture is \(11 \times
11\).

```python
common_params = dict(
    filters=32,
    kernel_size=3
)

Sequential([
    Conv2D(input_shape=input_shape, **common_params),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    BatchNormalization()
])
```

* **simple_cnn_ln**: This architecture is the same as the above, with the only
difference that we have replaced the spatial batch normalization with layer
normalization. The receptive field of this architecture is \(11 \times 11\).


```python
common_params = dict(
    filters=32,
    kernel_size=3,
)
Sequential([
    Conv2D(input_shape=input_shape, **common_params),
    LayerNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    LayerNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    LayerNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    LayerNormalization(),
    Activation("relu"),
    Conv2D(**common_params),
    LayerNormalization()
])
```
* **dilated_cnn_receptive_field_25**: For this architecture we also utilize
dilated convolutional layers in order to be able to increase the receptive
field without increasing the number of parameters. Again we employ RELU
non-linearity and we remove it from the last layer. The receptive field of this
architecture is \(25 \times 25\).

```python
Sequential([
    Conv2D(
        filters=32,
        kernel_size=5,
        input_shape=input_shape,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(
        filters=32,
        kernel_size=5,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(
        filters=32,
        kernel_size=5,
        kernel_regularizer=kernel_regularizer,
        dilation_rate=2
    ),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
    ),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization()
])
```

* **dilated_cnn_receptive_field_25_with_tanh**: This architecture is the same
as the above with the only difference that we have replaced the RELU
non-linearities with tanh non-linearities. Again the receptive field is \(25 \times 25 \).

```python
Sequential([
    Conv2D(
        filters=32,
        kernel_size=5,
        input_shape=input_shape,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("tanh"),
    Conv2D(
        filters=32,
        kernel_size=5,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("tanh"),
    Conv2D(
        filters=32,
        kernel_size=5,
        kernel_regularizer=kernel_regularizer,
        dilation_rate=2
    ),
    BatchNormalization(),
    Activation("tanh"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
    ),
    BatchNormalization(),
    Activation("tanh"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("tanh"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization(),
    Activation("tanh"),
    Conv2D(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    ),
    BatchNormalization()
])
```

## Inferring 3D Reconstructions

We provide three factories that can be used to test a previously trained
models. The `multi_view_cnn` factory can be used to test a Multi-View CNN
model, (namely without the MRF) and estimates discretized depth maps at
uniformly sampled depth hypotheses. Similar, the `multi_view_cnn_voxel_space`
factory is the same with the `multi_view_cnn` factory, with the only difference
that it predicts discretized depths on the voxel grid, defined using the
bounding box of the scene. Finally, the `raynet` factory can be used to infer
the 3D Model of a scene using our end-to-end trainable model. All factories
share the same API,

```python
forward_pass(scene, images_range)
```

**Arguments**

* scene: A [Scene](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/common/scene.py#L2)
that specifies the scene to be processed
* images_range: A tuple that specifies the indices of the views of the scene to
be used for the reconstruction

**Returns**

Given a `Scene` object and an image range that holds the views to be used for
the reconstruction, we predict a corresponding depth-map for every view.
