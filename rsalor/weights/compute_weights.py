
# Imports ----------------------------------------------------------------------
import os.path
from typing import List
import glob
import numpy as np
import ctypes


# Main -------------------------------------------------------------------------
def compute_weights(
        msa_path: str,
        msa_len: int,
        msa_depth: int,
        seqid: float=0.80,
        count_target_sequence: bool=True,
        num_threads: int=1,
        verboses: bool=False,
    ) -> List[float]:
    """Compute weights for all sequences of an MSA.
    Use C++ backend for time-performance. Implementation inspired from python package 'pycofitness'.
    
    Arguments:
        msa_path               (str):   path to msa '.fasta' file
        msa_len                (int):   length of the MSA (length of target sequence)
        msa_depth              (int):   depth of the MSA (number of sequences in the MSA)
        seqid                (float):   sequence identity threshold to consider two sequences as similar (default=0.80)
        count_target_sequence (bool):   count target sequence in weights computations
        num_threads            (int):   number of threads (CPUs) used by C++ backend (default=1)
        verboses              (bool):   set True to log steps of execution (default=False)
    
    Return:
        weights (List[float])
    """

    # Find C++ computeWeightsBackend compiled executable file
    backend_so_paths  = glob.glob(os.path.join(os.path.dirname(__file__), "lib_computeWeightsBackend*"))
    try:
        BACKEND_SO_PATH = backend_so_paths[0]
    except IndexError:
        print("\nERROR in compute_weights: ")
        print("\tUnable to find computeWeightsBackend dynamic library path.'")
        print("\tAre you running as a module before installation?")
        print("\tIn this case you need to build the shared object.")
        raise ValueError
    
    # Init C++ bridge
    computeWeightsBackend = ctypes.CDLL(BACKEND_SO_PATH)
    computeWeightsFunction = computeWeightsBackend.computeWeightsBackend
    computeWeightsFunction.argtypes = (
        ctypes.c_char_p,  # msa_path
        ctypes.c_uint,    # msa_len
        ctypes.c_uint,    # msa_depth
        ctypes.c_float,   # seqid
        ctypes.c_bool,    # count_target_sequence
        ctypes.c_uint,    # num_threads
        ctypes.c_bool     # verboses
    )
    computeWeightsFunction.restype = ctypes.POINTER(ctypes.c_float * msa_depth)
    freeWeights = computeWeightsBackend.freeWeights
    #freeWeights.argtypes = (???)
    freeWeights.restype = None

    # Run backend
    weights_ptr = computeWeightsFunction(
        msa_path.encode('utf-8'),
        msa_len,
        msa_depth,
        seqid,
        count_target_sequence,
        num_threads,
        verboses,
    )

    # Convert to list
    weights = np.zeros((msa_depth), dtype=np.float32)
    for i, x in enumerate(weights_ptr.contents):
        weights[i]= x

    # Free memory
    weights_ptr_casted = ctypes.cast(weights_ptr, ctypes.POINTER(ctypes.c_void_p))
    freeWeights(weights_ptr_casted)

    # Return
    return weights


def write_weights(weights: List[float], weights_path: str) -> None:
    """Read weights list from a file."""

    # Guardians
    assert os.path.isdir(os.path.dirname(weights_path)), f"ERROR in write_weights(): directory of weights_path='{weights_path}' does not exist."
    assert len(weights) > 0, f"ERROR in write_weights(): weigths list can not be of length zero."

    # Write
    weights_str = "\n".join([str(w) for w in weights])
    with open(weights_path, "w") as fs:
        fs.write(weights_str)


def read_weights(weights_path: str) -> List[float]:
    """Write weights list to a file."""

    # Guardians (for input)
    assert os.path.isfile(weights_path), f"ERROR in read_weights(): weights_path='{weights_path}' file does not exist."

    # Read from file
    with open(weights_path, "r") as fs:
        lines = fs.readlines()

    # Parse
    weights: List[float] = []
    for i, line in enumerate(lines):
        if len(line) > 1 and line[0] != "#":
            try:
                weights.append(float(line))
            except:
                raise ValueError(f"ERROR in read_weights(): failed to parse line {i+1}/{len(lines)} as a float.\n * weights_path='{weights_path}'\n * line='{line.replace('\n', '')}'")
    if len(weights) == 0:
        raise ValueError(f"ERROR in read_weights(): no parsable weights line found in weights_path='{weights_path}'.")
    return weights