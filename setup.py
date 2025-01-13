
# Imports ----------------------------------------------------------------------
from setuptools import setup, find_packages, Extension
#from setuptools.command.build_ext import build_ext


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
    name="rsalor",
    version="1.0.8",
    author="Matsvei Tsishyn",
    author_email="matsvei.tsishyn@protonmail.com",
    description="Combines structural data (Relative Solvent Accessibility, RSA) and evolutionary data (Log Odd Ratio, LOR from MSA) to evaluate missense mutations in proteins.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/3BioCompBio/RSALOR",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        #'llvmlite>0.30.0',
        'numpy',
        'biopython>=1.75',
    ],
    ext_modules = [compute_weights_ext],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: C",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    #entry_points={
    #    "console_scripts":[
    #        "rsalor=rsalor.main:run_mutation",
    #    ],
    #},
)