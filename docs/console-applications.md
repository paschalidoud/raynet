Console Applications
====================

Apart from the main library that we developed for the purpose of this paper, we also
provide 5 console applications that can be directly used to train models, test models and
compute various metrics such as *accuracy* and *completeness*. The available scripts are summarized bellow:

* [**raynet_pretrain**](#raynet_pretrain) is the console application used to
train the Multi-View CNN
* [**raynet_train**](#raynet_train) is the console application used to train
the RayNet (the Multi-View CNN and the MRF)
* [**raynet_forward**](#raynet_forward) is the console application used to
perform a forward pass given a pretrained Keras model, either using the
Multi-View CNN or the RayNet
* [**raynet_compute_metrics**](#raynet_compute_metrics) is the console
application used to compute various metrics such as the *per-pixel mean depth
error*, the *accuracy* and the
*completeness* of the 3D Reconstruction.
* [**raynet_to_pcl**](#raynet_to_pcl) is the console application used to
transform a set of depth maps into a point cloud.

All console applications have an extensive help menu that will help you provide
them with the necessary arguments. To be able to parse command-line arguments
we use the [argparse module](https://docs.python.org/3/library/argparse.html).

Below we give a brief overview of the corresponding console applications. For
additional examples for each individual console-application please refer to our
examples.


General arguments
-----------------

We briefly summarize some command-line arguments that are common for all
console applications:

* `dataset_type`: This argument is used to quite self-explanatory and is used
to control the dataset type used by the current command line. Currently, our
code, supports the two datasets used for our CVPR paper, namely the [*DTU MVS
dataset*](https://link.springer.com/article/10.1007/s11263-016-0902-9) and the
[*Aerial
dataset*](https://www.sciencedirect.com/science/article/pii/S0924271614002354).
In case you want to use our code with a different dataset you can easily
implement it in the existing codebase.


raynet_pretrain
---------------

**raynet_pretrain** is the command-line used to train a Multi-View CNN. In our
current implementation we provide the following architectures:

* ``simple_cnn``: It consists of a *Convolutional layer*, followed by a *Batch
Normalization layer* followed by a *Relu* 5 times. We remove remove the ReLU
from the last layer in order to retain information encoded both in the
negative and positive range. For this architecture we assume that the receptive
field is $11 \times 11$.
* ``dilated_cnn_with_receptive_field_25``: In order to be able to increase the
receptive field without increasing the number of parameters, for this
architecture we also use *dilated convolution layers*. For this architecture we
assume that the receptive field is $25 \times 25$.
* ``dilated_cnn_with_receptive_field_25_with_tanh``: The same as the previous
one, with the only difference that we replace the non-linearities with
*tanh* instead of *relu*.

The desired architecture for the Multi-View CNN can be specified using the argument
`--cnn_factory`. Currently, you can select one of the available cnn_factories,
however if you wish to experiment with a new architecture you can easily
incorporate it using the existing codebase. For more information please have a
look at the [corresponding example](/custom-mvcnn/).

Below we summarize the mandatory arguments for the **raynet_pretrain**
command-line:

* **training_directory**: A path to the directory containing the training set
* **test_directory**: A path to the directory containing the test set
* **output_directory**: A path to a directory that will be used to save the
trained models as well as various training statistics such as *training
loss*, *training accuracy*, *vaidation loss*, *validation accuracy* etc.
* **train_test_scenes_range**: A json file that controls which scenes from the
current dataset will be used as training and test set. To get an idea of how
this *.json* file should like, please check the *config* folder in our code.


raynet_train
------------

**raynet_train** is the command-line used to train RayNet, namely the
Multi-View CNN with an MRF. Again, depending on your desired architecture for
the Multi-View CNN you should specify it viw the argument `--cnn_factory`. For
the MRF, you need to specify the argument `--initial_gamma`, which is used to
control our initial belief for a voxel being occupied. Depending on whether
you want to learn the $\gamma$ directly from the data you can simply add
`--train_with_gamma` in your command-line arguments.

Below we summarize the mandatory arguments for the **raynet_pretrain**
command-line:

* **training_directory**: A path to the directory containing the training set
* **test_directory**: A path to the directory containing the test set
* **output_directory**: A path to a directory that will be used to save the
trained models as well as various training statistics such as *training
loss*, *training accuracy*, *vaidation loss*, *validation accuracy* etc.
* **weight_file**: A previously pretrained *KERAS* model from the Multi-View
* CNN.
* **train_test_scenes_range**: A json file that controls which scenes from the
current dataset will be used as training and test set. To get an idea of how
this *.json* file should like, please check the *config* folder in our code.

raynet_forward
--------------

**raynet_forward** is the command-line used to test pretrained models either
using *RayNet* or *Multi-View CNN*. This can be controlled using the
``--forward_pass_factory`` argument.

Below we summarize the mandatory arguments for the **raynet_pretrain**
command-line:

* **dataset_directory**: A path to the directory containing the dataset
* **output_directory**: A path to a directory that will be used to save the
trained models as well as various training statistics such as *training
loss*, *training accuracy*, *vaidation loss*, *validation accuracy* etc.
* **weight_file**: A previously pretrained *KERAS* model from the Multi-View
* CNN.


Currently, this script requires **CUDA**.  However, this constraint will be
lifted in the next versions.

raynet_compute_metrics
----------------------

Coming soon ...

raynet_to_pcl
-------------

Coming soon ...
