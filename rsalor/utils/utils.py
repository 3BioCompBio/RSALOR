
# Imports ----------------------------------------------------------------------
import os.path
from shutil import which
from typing import List, Union


# Base functions ---------------------------------------------------------------
def is_convertable_to(input_object, input_type) -> bool:
    """Return if input_object is convertable to input_type."""
    try:
        _ = input_type(input_object)
        return True
    except:
        return False
    
def memory_str(n_bytes: int) -> str:
    """Return a human readable string for a memory size measure (input in bytes)."""
    if n_bytes / 1000**3 > 1.0:
        return f"{n_bytes / 1000**3:.3f} GB"
    elif n_bytes / 1000**2 > 1.0:
        return f"{n_bytes / 1000**2:.3f} MB"
    elif n_bytes / 1000 > 1.0:
        return f"{n_bytes / 1000:.3f} kB"
    else:
        return f"{n_bytes} B"
    
def time_str(n_sec: float) -> str:
    """Return a human readable string for a time measure (input in seconds)."""
    if n_sec / (60*60*24) > 1.0:
        return f"{n_sec / (60*60*24):.3f} d."
    elif n_sec / (60*60) > 1.0:
        return f"{n_sec / (60*60):.3f} h."
    elif n_sec / 60 > 1.0:
        return f"{n_sec / 60:.3f} min."
    else:
        return f"{n_sec:.3f} sec."
    
def find_file(path_list: List[str], is_software: bool, name: str, description: Union[str, None]=None, verbose: bool=False,) -> str:
    """Find first existing file among path_list."""
    
    # Find valid path among candidates
    output_path = None
    for candidate_path in path_list:

        # Find as a path to a file
        if os.path.isfile(candidate_path):
            output_path = candidate_path
            if verbose:
                print(f" * Set path for [{name}] (AS PATH TO A FILE): '{output_path}'")
            break

    # Find valid bash command executable in PATH
    if output_path is None and is_software:
        for candidate_path in path_list:
            basename = os.path.basename(candidate_path)
            which_candidate_path = which(basename)
            if which_candidate_path is not None:
                output_path = which_candidate_path
                if verbose:
                    print(f" * set path for [{name}] (AS EXECUTABLE): '{output_path}'")
                break

    # Raise error if no valid path is found
    if output_path is None:

        # Init error message
        instance_name = "software" if is_software else "file"
        error_str = f"\nERROR in find_file(): no valid path found for {instance_name} '{name}':"
        error_str += "\nPath to file not found among: "

        # List failed candidates
        for candidate_path in path_list:
            error_str += f"\n - '{candidate_path}'"
        if is_software:
            error_str += "\nCommand not found in the system PATH among: "
            for candidate_path in path_list:
                error_str += f"\n - '{os.path.basename(candidate_path)}'"

        # Add recommendaiton
        if is_software:
            error_str += f"\n   -> Please install software '{name}' and provide the path to its executable file or add it to system PATH."

        # Add description
        if description is not None:
            error_str += f"\nDescription: \n{description}"
        raise ValueError(error_str)

    # Return first found valid path
    return output_path
    
    