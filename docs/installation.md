Installation
============

This page contains information about installing RayNet library. RayNet has the following dependencies:

* [Keras](https://keras.io/#installation) > 2
* [Tensorflow](https://www.tensorflow.org/install/)
* numpy
* [PyCuda](https://documen.tician.de/pycuda/)
* [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html)
* [backports.functools_lru_cache](https://pypi.org/project/backports.functools_lru_cache/)
* [imageio](http://imageio.readthedocs.io/en/latest/installation.html)
* [googleapiclient](https://developers.google.com/api-client-library/python/)
* [oauth2client](https://oauth2client.readthedocs.io/en/latest/)
* [matplotlib](https://matplotlib.org/users/installing.html)

Installing library
--------------------

Depending on how you would like to use the library, there are two alternatives
regarding installing the code. You can either use a package manager or download
and install the library manually. For those who just want to use the library we
recommend to directly install the latest version from *PyPI*, whereas for
those who wish to be able to edit the code we recommend to install the library
manually.

* **Install from Pypi with:**(you might need to add sudo)

```bash
pip install --user raynet
```

* **Install manually:**

Clone the [latest version](https://github.com/paschalidoud/raynet) of the library and run
```bash
# Clone the repository
git clone git@github.com:paschalidoud/raynet.git
cd raynet
# Local installation in development mode
pip install --user -e .
```

As soon as the installation is completed you can now start using/editing our
nice library :-)

