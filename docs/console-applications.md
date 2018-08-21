Overview
========

In addition to the main library that accompanies our [CVPR
paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf),
we also provide 5 useful console applications that can be used to:

* Train various models of any architecture
* Test these models 
* Assess the models based on the quality of the 3D reconstruction by computing
various metrics such as *accuracy* and *completeness*

These console applications allow for fast and easy experimentation without the
overhead of writing additional done. Below is the thorough list of these
console applications:

* [**raynet_pretrain**](#raynet_pretrain) is used to
train a Multi-View CNN model. For more details regarding what we mean by
Multi-View CNN model, please refer to our
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf).
* [**raynet_train**](#raynet_train) is used to train
the RayNet, our end-to-end trainable architecture that consists of a Multi-View
CNN and a Markov Random Field (MRF).
* [**raynet_forward**](#raynet_forward) is used to
perform inference using a  pre-trained *Keras* model, either using the
Multi-View CNN or the RayNet architecture.
* [**raynet_compute_metrics**](#raynet_compute_metrics) is used to compute
various metrics such as the *per-pixel mean depth
error*, the *accuracy* and the *completeness* of the 3D Reconstruction.
* [**raynet_to_pcl**](#raynet_to_pcl) is used to
transform a set of depth maps into a point cloud.

All console applications have an extensive help menu that will help you provide
them with the necessary arguments. To be able to parse command-line arguments
we use the [argparse module](https://docs.python.org/3/library/argparse.html).
Additional to this, below, we also give a brief overview of the corresponding
console applications. For additional examples please refer to our
[Examples](/custom-mvcnn/).


Common arguments
----------------

While every console application has different command-line arguments there
are some common arguments for all of them, which are briefly mentioned below:

* `dataset_type`: This argument is pretty self-explanatory and is used
to set the format of the dataset. Currently, our code, supports two dataset
formats, the [*DTU MVS
dataset*](https://link.springer.com/article/10.1007/s11263-016-0902-9) and the
[*Aerial
dataset*](https://www.sciencedirect.com/science/article/pii/S0924271614002354).
In case you want to use our code with a different dataset you can easily
incorporate a suitable
[Dataset](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/common/dataset.py#L8)
wrapper in our existing codebase.
* `select_neighbors_based_on`: This argument is used to control how
neighbouring views are selected. We provide two alternatives based either on
their geometrical distance or their order in the file system. They can be
selected with `distance` and `filesystem` respectively.
* `illumination_condition`: This argument is used to choose the desired
illumination condition of the input views only fro the DTU dataset.
* `input_output_dimensionality`: This argument is used to set the dimensions of
the input vector in the Multi-View CNN model as well as the dimensions of the
prediction. The `default` argument set the input dimension to be equal to \((D
\times N \times patch\_shape)\), where \(D\) is the number of depth hypotheses,
\(N\) is the number of all combinations of all views and \(patch\_shape\) is
the shape of the patch. The dimensions of the prediction is \(D\), namely one
probability per depth hypotheses.
* `loss`: This argument is used to choose the loss function to be minimized
while training a neural network architectures. The user can select one of the
following loss functions, `emd` that refers to the [Earth Mover's Distance
loss](https://arxiv.org/pdf/1611.05916.pdf), `squared_emd` that refers to its
squared variant and `categorical crossentropy` that refers to the [Cross
entropy loss](https://en.wikipedia.org/wiki/Cross_entropy).

raynet_pretrain
---------------

**raynet_pretrain** is the command-line used to train a Multi-View CNN model of
an arbitrary architecture. In our current implementation, we have developed
various architectures that can be selected by setting the `--cnn_factory`
argument using one of the following:

* ``simple_cnn``: This architecture consists of a *Convolutional layer*,
followed by a *Batch Normalization layer* followed by a *ReLU* 5 times. We
remove remove the ReLU from the last layer in order to retain information
encoded both in the negative and positive range. For this architecture we
assume that the receptive field is \(11 \times 11\).
* ``dilated_cnn_with_receptive_field_25``: In order to be able to increase the
receptive field without increasing the number of parameters in this
architecture we also use *dilated convolution layers*. For this architecture we
assume that the receptive field is \(25 \times 25\).
* ``dilated_cnn_with_receptive_field_25_with_tanh``: The same as the previous
one, with the only difference that we replace the non-linearities with
*tanh* instead of *ReLU*.

Currently, you can select one of the available cnn_factories, however if you
wish to experiment with a new architecture you can easily incorporate in the
existing codebase. For more information please have a look at the
[corresponding example](/custom-mvcnn/). When choosing your desired
architecture do not forget to also set the corresponding `--patch_shape`
argument based on the receptive field of this architecture. 

Below we summarize the mandatory arguments for the **raynet_pretrain**
command-line:

* **training_directory**: A path to the directory containing the training set.
* **test_directory**: A path to the directory containing the test set.
* **output_directory**: A path to a directory that will be used to save the
trained models as well as various training statistics such as *training
loss*, *training accuracy*, *vaidation loss*, *validation accuracy* etc.
* **train_test_scenes_range**: A json file that controls which scenes from the
current dataset will be used as training and test set. To get an idea of how
this json file should like, please check the `config folder` in our code.

In addition, we also list some optional arguments that are most commonly used:

* **n_test_samples**: The number of samples used for validation
* **training_cached_samples**: In order to further speed up the training
process, we implemented a [Producer-Consumer
pattern](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem) by creating
a
[BatchProvider](https://github.com/paschalidoud/raynet/blob/deb209064039596c321d7ddd4cb4a210b0e8a5d8/raynet/train_network/batch_provider.py#L13) class.
In more detail, one thread is in charge of **producing** batches, while the main
thread is in charge of **consuming** this batches for training. This argument
specifies the set of cached samples used to randomly choose batches from.
Depending on your system's memory constraints you might need to properly adapt it!


raynet_train
------------

**raynet_train** is the command-line used to train RayNet, namely the
Multi-View CNN with the MRF. Again, depending on the desired architecture for
the Multi-View CNN model you should choose it by appropriately setting the
argument `--cnn_factory`. For the MRF, you need to specify the argument
`--initial_gamma`, which is used to control our initial belief for a voxel
being occupied. Depending on whether you want to learn the \(\gamma\) directly
from the data you can simply add `--train_with_gamma` in your command-line
arguments.

Below we summarize the mandatory arguments for the **raynet_train**
command-line:

* **training_directory**: A path to the directory containing the training set.
* **test_directory**: A path to the directory containing the test set,
* **output_directory**: A path to a directory that will be used to save the
trained models as well as various training statistics such as *training
loss*, *training accuracy*, *vaidation loss*, *validation accuracy* etc.
* **weight_file**: A previously pre-trained *KERAS* model from the Multi-View
CNN.
* **train_test_scenes_range**: A json file that controls which scenes from the
current dataset will be used as training and test set. To get an idea of how
this json file should like, please check the `config folder` in our code.

raynet_forward
--------------

**raynet_forward** is the command-line used to test previously trained models
using various models. The type of the pre-trained model can be chosen via
setting the ``--forward_pass_factory`` argument. The `multi_view_cnn` factory
refers to a Multi-View CNN model and estimates discretized depth maps at
uniformly sampled depth hypotheses. Similar, the `multi_view_cnn_voxel_space`
factory is the same as the above, with the only difference that it predicts
discretized depths on the voxel grid, defined using the bounding box of the
scene. Finally, the `raynet` factory refers to our end-to-end trainable model,
RayNet.  The output of this application is a directory that contains a depth
map for each view of the scene.

Below we summarize the mandatory arguments for the **raynet_forward**
command-line:

* **dataset_directory**: A path to the directory containing the dataset.
* **output_directory**: A path to a directory that will be used to save a depth
map for every view in the scene that we want to reconstruct.

Some important optional arguments, worth mentioning are listed below:

* **--weight_file**: A previously pre-trained *KERAS* model.
* **--scene_idx**: The index of the scene to be reconstructed. Keep in mind that
if there is not an explicit scene indexing system, we map the scenes of a
dataset to indices based on their alphabetical order.
* **--start_end**: A tuple setting the indices of the first and the final views
to be reconstructed.
* **--skip_every**: The number of views to be skipped between the first and the
final view.

Again based on your desired `--cnn_factory` you should correctly set the
`--patch_shape` argument. Currently, this script requires **CUDA**.  However,
this constraint will be lifted in the next versions.

raynet_compute_metrics
----------------------

**raynet_compute_metrics** is the command-line used to estimate the quality of
the 3D reconstruction. Our current implementation supports three metrics, the
*accuracy*, the *completeness* and the *per pixel mean depth error*. The first
two metrics are estimated in the 3D space, while the later is defined in the
image space and averaged over all ground truth depth maps. Given a directory
that contains the estimated depth maps our console application can estimate all
the aforementioned metrics.

Below we summarize the mandatory arguments for the **raynet_compute_metrics**
command-line:

* **dataset_directory**: A path to the directory that contains the dataset.
* **predictions_directory**: A path to the directory that contains the estimated
depth maps.
* **{ppmde,accuracy,completeness}**: The implemented metrics to be used. The
user can select multiple metrics at the same time in a single run.

As we already mentioned above, the *accuracy* and *completeness* are estimated
in the 3D space, thus we need to convert the estimated depth maps into a
point-cloud before computing these metrics. However, simply estimating the
point cloud from a set of depth maps is quite noisy, thus we provide three
arguments, `--n_neighbors`, `--consistency_threshold`, `--with_consistency_check`
that enables some standard filtering methods.

raynet_to_pcl
-------------

**raynet_to_pcl** is the command-line used to convert a set of depth maps to a
point-cloud. Similar to the previous console application, also here we provide
three arguments, `--n_neighbors`, `--consistency_threshold`,
`--with_consistency_check` that can be used to filter the reconstructed
point-cloud.

Below we summarize the mandatory arguments for the **raynet_to_pcl**
command-line:

* **dataset_directory**: A path to the directory that contains the dataset.
* **predictions_directory**: A path to the directory that contains the estimated
depth maps.
* **output_directory**: A path to the directory that will be used to save the
estimated point-cloud.
