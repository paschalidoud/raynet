"""Setup RayNet"""

from itertools import dropwhile
from os import path
from setuptools import find_packages, setup
from setuptools.extension import Extension
from subprocess import Popen


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("raynet","__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def is_cuda_on_the_machine():
    try:
        p = Popen([
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode",
                "--format=csv,noheader,nounits"
        ])
        return True
    except OSError:
        # There is no GPU on the machine
        return False


def setup_package():
    extensions = [
        Extension(
            "raynet.utils.fast_utils",
            ["raynet/utils/fast_utils.pyx"]
        ),
        Extension(
            "raynet.ray_marching.ray_tracing",
            ["raynet/ray_marching/ray_tracing.pyx"]
        )
    ]

    requirements = [
        "Cython",
        "backports.functools_lru_cache",
        "keras>=2",
        "tensorflow",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "imageio",
        "oauth2client",
        "google-api-python-client"
    
    ]

    try:
        import tensorflow
    except ImportError:
        if is_cuda_on_the_machine():
            requirements.append("tensorflow-gpu")
        else:
            requirements.append("tensorflow")

    try:
        import pycuda
    except ImportError:
        if is_cuda_on_the_machine():
            requirements.append("pycuda")

    with open("README.rst") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="raynet",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7"
        ],
        packages=find_packages(exclude=["docs", "tests", "scripts", "config"]),
        setup_requires=[
            "setuptools>=18.0",
            "Cython"
        ],
        install_requires=requirements,
        ext_modules=extensions,
        entry_points={
            "console_scripts": [
                "raynet_compute_metrics = raynet.scripts.compute_metrics:main",
                "raynet_to_pcl = raynet.scripts.convert_to_pointcloud:main",
                "raynet_forward = raynet.scripts.forward_pass:main",
                "raynet_pretrain = raynet.scripts.pretrain_network:main",
                "raynet_train = raynet.scripts.train_raynet:main"
            ]
        },
        package_data={
            "": ["*.cu", "*.pyx"]
        }
    )


if __name__ == "__main__":
    setup_package()
