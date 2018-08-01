import pkg_resources

import numpy as np
from pycuda.gpuarray import to_gpu


def _cuda_file(filepath):
    return pkg_resources.resource_filename(__name__, filepath)


def all_arrays_to_gpu(f):
    """Decorator to copy all the numpy arrays to the gpu before function
    invokation"""
    def inner(*args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                args[i] = to_gpu(args[i])

        return f(*args, **kwargs)

    return inner


def parse_cu_files_to_string(file_paths):
    # Make sure that the file_paths argument is a list
    assert isinstance(file_paths, list)

    # Generate an empty string to concatenate all cuda source files
    cu_source_code = ""
    for fp in file_paths:
        with open(_cuda_file(fp), "r") as f:
            cu_source_code += f.read()

    return cu_source_code
