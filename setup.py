
"""
NOTE about the 'setup.py' deprecation.
Despite the fact that 'setup.py' is now deprecated and replaced by 'pyproject.toml', it is still required to include C++ modules in pip packages.
Here is a minimal 'setup.py' that includes the C++ code in the package to complement the 'pyproject.toml' (this is fucked up)
"""

# Imports ----------------------------------------------------------------------
from setuptools import setup, Extension

# Extensions -------------------------------------------------------------------
# Define extension (C++ code that need to be compiled)
compute_weights_ext = Extension(
    'rsalor.weights.lib_computeWeightsBackend', # name
    sources=[ # .cpp files
        'rsalor/weights/computeWeightsBackend.cpp',
        'rsalor/weights/msa.cpp',
    ],
    include_dirs=[ # .h directories
        'rsalor/weights/include',
    ],
    extra_compile_args=['-std=c++11', '-O3'],  # optimization and other flags
    extra_link_args=['-O3'],
    language='c++',
)

# Setup ------------------------------------------------------------------------
setup(
    ext_modules = [compute_weights_ext],
)