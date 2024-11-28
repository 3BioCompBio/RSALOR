
# Imports ----------------------------------------------------------------------
import os.path
from os import remove
from typing import Union
from Bio import SeqIO

# MSA files processing functions -----------------------------------------------
def ali_to_fasta(input_path: str, output_path: str, delete_input: bool=False) -> Union[None, str]:
    """Convert '.ali' (Stickholm format) file to a '.fasta' file.
        * Then, deletes input '.ali' file if required.
        * Returns output_path or None if execution failed.
    Source: https://stackoverflow.com/questions/24156578/using-bio-seqio-to-write-single-line-fasta
    """

    # Guardians
    error_log = f"ERROR in ali_to_fasta(): "
    error_log += f"\n * input_path  : '{input_path}'"
    error_log += f"\n * output_path : '{output_path}'\n"
    if not os.path.isfile(input_path):
        print(f"{error_log} -> input file does not exists.")
        return None
    if not is_nonempty_file(input_path):
        print(f"{error_log} -> input file is empty.")
        return None
    if not input_path.endswith(".ali"):
        print(f"{error_log} -> input file should end with '.ali'.")
        return None
    if not output_path.endswith(".fasta"):
        print(f"{error_log} -> output file should end with '.fasta'.")
        return None

    # Run convertion
    try:
        records = SeqIO.parse(input_path, "stockholm")
    except Exception as error:
        print(f"{error_log} -> input file parsing failed.")
        print(error)
        return None
    try:
        SeqIO.FastaIO.FastaWriter(output_path, wrap=None).write_file(records)
    except Exception as error:
        print(f"{error_log} -> file convertion + writing failed.")
        print(error)
        return None
    
    # Detect errors
    if not is_nonempty_file(output_path):
        print(f"{error_log} -> converted output file is empty.")
        return None
    
    # Delete initial '.ali' file if required
    if delete_input:
        if os.path.isfile(input_path):
            remove(input_path)

    # Return
    return output_path

# Dependency -------------------------------------------------------------------
def is_nonempty_file(input_path: str) -> bool:
    """Check if 'input_path' is an existing non-empty file."""
    if not os.path.isfile(input_path):
        return False
    with open(input_path, "r") as fs:
        line = fs.readline()
        return len(line) > 0