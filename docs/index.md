# RayNet: Learning Volumetric 3D Reconstruction


Recent methods based on convolutional neural networks (CNN) allow learning the
entire task from data. However, they do not incorporate the physics of image
formation such as perspective geometry and occlusion.  Instead, classical
approaches based on [Markov Random Fields
(MRF)](https://ermongroup.github.io/cs228-notes/representation/undirected/)
with ray-potentials explicitly model these physical processes, but they cannot
cope with large surface appearance variations across different viewpoints. In
this paper, we propose **RayNet**,
which combines the strengths of both frameworks. RayNet integrates a CNN that
learns view-invariant feature representations with an MRF that explicitly
encodes the physics of perspective projection and occlusion. We train RayNet
end-to-end using empirical risk minimization. We thoroughly evaluate our
approach on challenging real-world datasets and demonstrate its benefits over a
piece-wise trained baseline, hand-crafted models as well as other
learning-based approaches.

The purpose of this site is to host the documentation page of our code that
accompanies our CVPR 2018 paper with title [**RayNet:Learning Volumetric 3D
Reconstruction with Ray
Potentials**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf)

Below you can find our:

* [Supplamentary material](http://www.cvlibs.net/publications/Paschalidou2018CVPR_supplementary.pdf)
* [Poster from CVPR 2018](http://www.cvlibs.net/publications/Paschalidou2018CVPR_poster.pdf)
* [Spotlight talk from CVPR 2018](https://youtu.be/PZ0u1VZLLkU)


# Code Documentation

While this library is originally developed to accompany our CVPR publication,
we offer various scripts that can be easily used to perform 3D reconstruction
using a set of images taken from known camera poses.

You can navigate the documentation from the top navigation bar but we also
provide a list of useful links below.

* [Installation instructions](/installation/)
* [Using the console applications](/console-applications/)
* [Getting started with the library](/getting-started/)


## Citation

If you use the library please also cite our paper. The bibtex can be found below:

```
@InProceedings{Paschalidou_2018_CVPR,
    author = {
        Paschalidou, Despoina and
        Ulusoy, Osman and
        Schmitt, Carolin and
        Van Gool, Luc and Geiger, Andreas
    },
    title = {RayNet: Learning Volumetric 3D Reconstruction With Ray Potentials},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```


## License

raynet-mvs is released under the MIT license which practically allows anyone to do anything with it.
