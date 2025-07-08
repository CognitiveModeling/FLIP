from setuptools import setup, Extension
import numpy as np
import os
import sys
import platform
import site
import subprocess

# Function to find library paths in Conda environment
def get_conda_paths():
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        return {
            'include': os.path.join(conda_prefix, 'include'),
            'lib': os.path.join(conda_prefix, 'lib')
        }
    return None

# Function to find system library paths
def get_system_paths():
    # Common locations for HDF5 includes and libraries
    potential_hdf5_includes = [
        '/usr/include/hdf5/serial',
        '/usr/local/include',
        '/opt/local/include',
    ]
    
    potential_hdf5_libs = [
        '/usr/lib/x86_64-linux-gnu/hdf5/serial',
        '/usr/local/lib',
        '/opt/local/lib',
    ]
    
    include_path = None
    lib_path = None
    
    # Find HDF5 include path
    for path in potential_hdf5_includes:
        if os.path.exists(os.path.join(path, 'hdf5.h')):
            include_path = path
            break
    
    # Find HDF5 lib path
    for path in potential_hdf5_libs:
        if os.path.exists(os.path.join(path, 'libhdf5.so')):
            lib_path = path
            break
    
    return {
        'include': include_path,
        'lib': lib_path
    }

# Try to detect if we're in a Conda environment
conda_paths = get_conda_paths()
if conda_paths:
    print(f"Using Conda environment: {os.environ.get('CONDA_PREFIX')}")
    include_dirs = [
        np.get_include(),
        conda_paths['include'],
    ]
    library_dirs = [
        conda_paths['lib'],
    ]
    # Use conda's rpath
    extra_link_args = [
        "-Wl,-rpath," + conda_paths['lib'],
    ]
else:
    print("No Conda environment detected, using system libraries.")
    system_paths = get_system_paths()
    include_dirs = [
        np.get_include(),
    ]
    library_dirs = []
    extra_link_args = []
    
    if system_paths['include']:
        include_dirs.append(system_paths['include'])
    if system_paths['lib']:
        library_dirs.append(system_paths['lib'])
        # Add rpath for system libraries
        extra_link_args.append("-Wl,-rpath," + system_paths['lib'])

# Define the position C extension
position_module = Extension(
    "flip_position",          # Name of the module
    sources=["py_position.c", "position.c", "gaussian.c", "bbox.c", "ziggurat_inline.c"],  # Source files
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3", 
        "-march=native", 
        "-flto", 
        "-fstrict-aliasing", 
        "-ffast-math", 
        "-funroll-loops", 
        "-I" + np.get_include()
    ],
    extra_link_args=[
        "-O3", 
        "-march=native", 
        "-flto", 
        "-fopenmp"
    ]
)

# Setup script
setup(
    name="flip_position",
    version="1.0",
    description="C extension for position processing and augmentation",
    ext_modules=[position_module],
)
