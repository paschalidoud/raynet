Getting Started
===============

The goal of this page is to briefly introduce you to our library and its
capabilities. As soon as you have successfully installed version of our library
you can start experimenting  with it to generate nice 3D Reconstructions of a
scene.

## Download data

For our [CVPR paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf) we experimented with two datasets, summarized below:

* [Aerial dataset](https://www.sciencedirect.com/science/article/pii/S0924271614002354)
* [DTU Multi-view stereo benchmark](https://link.springer.com/article/10.1007/s11263-016-0902-9)

The **DTU dataset** is an indoor dataset that contains a great variety of
objects from different materials. It can be downloaded from
[here](http://roboimagedata.compute.dtu.dk/?page_id=36). On the contrary, the
**Aerial dataset** is an outdoor dataset that contains images from urban
environments captured from an aerial platform. It can be downloaded from
[here](http://raynet-mvs.com/site/providence_data.tar.gz). In case you use any
of these datasets please do not forget to cite the corresponding papers
mentioned above!

## Train a simple Multi-View CNN model from scratch

As soon as you have downloaded any of these two datasets you can directly start
training a simple Multi-View CNN model from scratch. For this particular
example we will train a Multi-View CNN model on the Aerial dataset, which is
stored in the `/tmp/aerial_dataset/` directory for 40 epochs. Using our
`raynet_pretrain` console application you can easily start immediately training
your model using one of the provided architectures.

Here follows the terminal output for the first 15 training epochs:

<pre style="height: 300px; overflow-y: scroll;"><code class="bash">
$ CUDA_VISIBLE_DEVICES=0 raynet_pretrain /tmp/aerial_dataset/ /tmp/aerial_dataset /tmp/foo /path/to/config/restrepo_train_test_splits.json --lr 1e-3 --epochs 40
Using TensorFlow backend.
{0: 'BH', 1: 'capitol', 2: 'downtown'}
{0: 'BH', 1: 'capitol', 2: 'downtown'}
Create '0SGLLKUKPUFE48VWNUQS' folder for current experiment
Collecting test set...
Building the octree for the current scene. Be patient...
500/500 [==============================] - 171s       
Cache 500 samples for training
Building the octree for the current scene. Be patient...
499/500 [============================>.] - ETA: 0s
Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:02:00.0
Total memory: 11.91GiB
Free memory: 11.75GiB
2018-08-21 13:21:10.090922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2018-08-21 13:21:10.090928: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2018-08-21 13:21:10.090935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:02:00.0)
Epoch 1/40
500/500 [==============================] - 67s - loss: 0.0749 - acc: 0.3473 - mae: 0.0429 - mde: 2.2181 - val_loss: 0.1275 - val_acc: 0.4060 - val_mae: 0.0393 - val_mde: 4.0400
Epoch 2/40
500/500 [==============================] - 66s - loss: 0.0533 - acc: 0.3991 - mae: 0.0396 - mde: 1.5912 - val_loss: 0.0515 - val_acc: 0.5420 - val_mae: 0.0292 - val_mde: 1.5800
Epoch 3/40
500/500 [==============================] - 68s - loss: 0.0474 - acc: 0.4094 - mae: 0.0386 - mde: 1.4214 - val_loss: 0.0482 - val_acc: 0.5780 - val_mae: 0.0279 - val_mde: 1.5080
Epoch 4/40
500/500 [==============================] - 68s - loss: 0.0446 - acc: 0.4129 - mae: 0.0379 - mde: 1.3374 - val_loss: 0.0405 - val_acc: 0.6480 - val_mae: 0.0236 - val_mde: 1.2420
Epoch 5/40
500/500 [==============================] - 69s - loss: 0.0449 - acc: 0.4104 - mae: 0.0380 - mde: 1.3582 - val_loss: 0.0865 - val_acc: 0.4700 - val_mae: 0.0339 - val_mde: 2.7700
Epoch 6/40
500/500 [==============================] - 68s - loss: 0.0429 - acc: 0.4310 - mae: 0.0370 - mde: 1.2906 - val_loss: 0.0365 - val_acc: 0.6200 - val_mae: 0.0245 - val_mde: 1.1460
Epoch 7/40
500/500 [==============================] - 68s - loss: 0.0396 - acc: 0.4381 - mae: 0.0363 - mde: 1.2038 - val_loss: 0.0467 - val_acc: 0.6060 - val_mae: 0.0258 - val_mde: 1.5280
Epoch 8/40
500/500 [==============================] - 68s - loss: 0.0402 - acc: 0.4160 - mae: 0.0373 - mde: 1.2304 - val_loss: 0.0453 - val_acc: 0.6040 - val_mae: 0.0249 - val_mde: 1.5080
Epoch 9/40
500/500 [==============================] - 68s - loss: 0.0424 - acc: 0.4230 - mae: 0.0371 - mde: 1.3170 - val_loss: 0.0441 - val_acc: 0.5940 - val_mae: 0.0260 - val_mde: 1.3320
Epoch 10/40
500/500 [==============================] - 68s - loss: 0.0416 - acc: 0.4304 - mae: 0.0365 - mde: 1.2781 - val_loss: 0.0477 - val_acc: 0.5760 - val_mae: 0.0266 - val_mde: 1.5620
Epoch 11/40
500/500 [==============================] - 68s - loss: 0.0374 - acc: 0.4555 - mae: 0.0352 - mde: 1.1437 - val_loss: 0.0649 - val_acc: 0.5760 - val_mae: 0.0267 - val_mde: 2.0180
Epoch 12/40
500/500 [==============================] - 68s - loss: 0.0395 - acc: 0.4561 - mae: 0.0351 - mde: 1.2291 - val_loss: 0.0444 - val_acc: 0.5840 - val_mae: 0.0258 - val_mde: 1.4680
Epoch 13/40
500/500 [==============================] - 68s - loss: 0.0413 - acc: 0.4189 - mae: 0.0368 - mde: 1.2853 - val_loss: 0.0437 - val_acc: 0.6180 - val_mae: 0.0242 - val_mde: 1.5040
Epoch 14/40
500/500 [==============================] - 68s - loss: 0.0385 - acc: 0.4556 - mae: 0.0348 - mde: 1.1827 - val_loss: 0.0381 - val_acc: 0.6080 - val_mae: 0.0248 - val_mde: 1.2140
Epoch 15/40
301/500 [=====================>........] - ETA: 54s - loss: 0.0372 - acc: 0.4373 - mae: 0.0363 - mde: 1.2138
</code></pre>

As you can see during training, we report various metrics such as the loss and
the accuracy both on the training and the validation set to be able to check
that the network converges. Another thing worth mentioning has to do with the
output of this console application. As you can see in the terminal output above
this script creates a folder inside the output directory with name
`0SGLLKUKPUFE48VWNUQS`, where it stores various statistics regarding the
training process as well as the trained models.

## Test our previously trained model

As soon as we have finished with training our model, we can evaluate the
learned model on a scene using the `raynet_forward` console application. For
simplicity, for this example, we will test the trained model on the 20 first
images of the *DOWNTOWN* scene. Below follows the corresponding terminal output:

<pre style="height: 300px; overflow-y: scroll;"><code class="bash">
$ CUDA_VISIBLE_DEVICES=0 raynet_forward /tmp/aerial_dataset/ /tmp/foo/0SGLLKUKPUFE48VWNUQS/depth_maps/ --weight_file /tmp/foo/0SGLLKUKPUFE48VWNUQS/weights/weights.39.hdf5 --scene_idx 2 --network_architecture simple_cnn--forward_pass_factory multi_view_cnn --dataset_type restrepo --start_end 0,20
Using TensorFlow backend.
StreamExecutor works with that.
2018-08-21 14:31:06.013088: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:02:00.0
Total memory: 11.91GiB
Free memory: 11.60GiB
2018-08-21 14:31:06.013104: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2018-08-21 14:31:06.013110: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2018-08-21 14:31:06.013116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:02:00.0)
{0: 'BH', 1: 'capitol', 2: 'downtown'}
Features computation -  2.194076
Per-pixel depth estimation -  0.679471
Features computation -  0.659598
Per-pixel depth estimation -  0.065472
Features computation -  0.636852
Per-pixel depth estimation -  0.064682
Features computation -  0.630682
Per-pixel depth estimation -  0.065272
Features computation -  0.634153
Per-pixel depth estimation -  0.066705
Features computation -  0.647827
Per-pixel depth estimation -  0.067191
Features computation -  0.638852
Per-pixel depth estimation -  0.065388
Features computation -  0.631912
Per-pixel depth estimation -  0.064861
Features computation -  0.632437
Per-pixel depth estimation -  0.065154
Features computation -  0.64483
Per-pixel depth estimation -  0.064949
Features computation -  0.631402
Per-pixel depth estimation -  0.066548
Features computation -  0.700359
Per-pixel depth estimation -  0.066643
Features computation -  0.633139
Per-pixel depth estimation -  0.064806
Features computation -  0.633135
Per-pixel depth estimation -  0.064748
Features computation -  0.673216
Per-pixel depth estimation -  0.065133
Features computation -  0.685232
Per-pixel depth estimation -  0.064738
Features computation -  0.632668
Per-pixel depth estimation -  0.065855
Features computation -  0.628777
Per-pixel depth estimation -  0.065074
Features computation -  0.637438
Per-pixel depth estimation -  0.066568
Features computation -  0.6403
Per-pixel depth estimation -  0.065694
</code></pre>

This console application creates a set of depth maps, one for every input view
based on the learned model. These depth maps are stored as **numpy arrays** in
the `/tmp/foo/0SGLLKUKPUFE48VWNUQS/depth_maps/` folder. Below we visualize some
of the predicted depth maps just by using the **Multi-View CNN**. As you can
easily notice the depth predictions are quite noisy due to the small receptive
field of the architecture and the small during of training (only 40 epochs)

<div class="fig col-2">
<img src="../gfx/5.png" alt="view-5" width="330" height="500">
<img src="../gfx/1.png" alt="view-1" width="330" height="500">
<img src="../gfx/0.png" alt="view-0" width="330" height="500">
<img src="../gfx/9.png" alt="view-9" width="330" height="500">
<img src="../gfx/2.png" alt="view-2" width="330" height="500">
<img src="../gfx/15.png" alt="view-15" width="330" height="500">

<span>Depth predictions after training a Multi-View CNN with receptive field
\(11 \times 11\) for 40 epochs.</span>
</div>

