RayNet
======

This python package provides the code that accompanies our CVPR 2018 paper with
title **RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials**.

.. image:: http://raynet-mvs.com/site/raynet_teaser.png

Dependencies & Installation
---------------------------

Normally, a ``pip install raynet`` should suffice to use our code.

If you already have a functional Keras installation there are not much left to
install :-)

* ``Keras`` > 2
* ``Tensorflow``
* ``Cython``
* ``PyCuda``
* ``backports.functools_lru_cache``
* ``imageio``
* ``googleapiclient``
* ``oauth2client``
* ``numpy``
* ``matplotlib``
* ``scikit-learn``


Depending on how you want to use our code, there are two alternatives regarding
installation. You can either use a package manager or download and install the
library manually. For those who just want to use the library, we recommend to
directly install the latest version from *PyPI*, whereas for those who want to
be able to edit the code, we recommend to install the library manually.


* *Install from Pypi with:*

.. code:: bash

    pip install --user raynet

* *Install manually:*

Clone the `latest version <https://github.com/paschalidoud/raynet>`__ of the library and run

.. code:: bash

    # Clone the repository
    git clone git@github.com:paschalidoud/raynet.git
    cd raynet
    # Local installation in development mode
    pip install --user -e .

Documentation
-------------

The dedicated documentation page can be found in our `documentation site <http://raynet-mvs.com>`__ but you can also read the
`source code <https://github.com/paschalidoud/raynet>`__  to get an
idea of how our code can be used. If you have any question regarding the code
please contact `Despoina Paschalidou <https://avg.is.tuebingen.mpg.de/person/dpaschalidou>`__.

Contribution
------------

Contributions such as bug fixes, bug reports, suggestions etc. are more than
welcome and should be submitted in the form of new issues and/or pull requests
on Github.

Relevant Research
-----------------

Below we list some papers that are relevant to the provided code.

**Ours**

* RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials [`pdf <http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf>`__]

**By Others**

* Towards Probabilistic Volumetric Reconstruction using Ray Potentials [`pdf <http://www.cvlibs.net/publications/Ulusoy2015THREEDV.pdf>`__]
* Patches, Planes and Probabilities: A Non-local Prior for Volumetric 3D Reconstruction [`pdf <http://www.cvlibs.net/publications/Ulusoy2016CVPR.pdf>`__]
* Semantic Multi-view Stereo: Jointly Estimating Objects and Voxels [`pdf <http://www.cvlibs.net/publications/Ulusoy2017CVPR.pdf>`__]
* Learned Multi Patch Similarity [`pdf <https://arxiv.org/pdf/1703.08836.pdf>`__]
* SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis [`pdf <https://arxiv.org/pdf/1708.01749.pdf>`__]

Citation
--------
If you are using our code, please cite `our paper <http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf>`__. The BibTeX reference is::

 @InProceedings{Paschalidou_2018_CVPR,
    author = {Paschalidou, Despoina and Ulusoy, Osman and Schmitt, Carolin and Van Gool, Luc and Geiger, Andreas},
    title = {RayNet: Learning Volumetric 3D Reconstruction With Ray Potentials},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
    }


