
# Imports ----------------------------------------------------------------------
import os.path
import sys
from typing import Dict, Union
import tempfile
from contextlib import contextmanager
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from rsalor.utils import find_file
from rsalor.rsa.rsa_solver import RSASolver

# RSADSSP ----------------------------------------------------------------------
class RSADSSP(RSASolver):
    """
    RSADSSP(): Solver for RSA (Relative Solvent Accessibility) from a '.pdb' file using DSSP software.
    
    usage:
        rsa_map = RSADSSP('./soft/DSSP/dssp').run('./my_pdb.pdb')
    """

    # Constants ----------------------------------------------------------------
    CANDIDATES_PATHS = ["mkdssp", "dssp"]
    HELPER_LOG = """-------------------------------------------------------
RSA Solver: DSSP issue:
In order to solve Relative Solvent Accessiblity (RSA), RSALOR package uses:
Python package biopython -> interface with the DSSP algorithms (https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html).
The DSSP software (free for academic use) has to be installed on your computer.
Please install DSSP (https://swift.cmbi.umcn.nl/gv/dssp/) and specify the path to its executable or add it to system PATH.
DSSP source code can be found here: https://github.com/cmbi/hssp

NOTE: you can still use the RSALOR package without DSSP if you only want LOR values of the MSA without using RSA (just set pdb_path=None).
-------------------------------------------------------"""

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return "RSASolver['DSSP']"

    def execute_solver(self, pdb_path: str) -> Dict[str, float]:
        """Compute RSA by running DSSP.

        args:
        pdb_path (str):         path to PDB file   

        output:
        {resid: str => RSA: float}     (such as {'A13': 48.57, ...})
        """

        # Init DSSP path (check DSSP executable existance only if software is executed)
        self._init_dssp_path()

        # Run DSSP
        pdb_with_cryst1_line = self._inject_cryst1_line(pdb_path) # Manage CRYST1 line
        # Case: CRYST1 line is already present or this version of DSSP does not requires it
        if pdb_with_cryst1_line is None:
            rsa_map = self._run_dssp_backend(pdb_path)
        # Case: inject CRYST1 line and run DSSP on modified PDB
        else:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                tmp_pdb_path = temp_file.name
                with open(tmp_pdb_path, "w") as fs:
                    fs.write(pdb_with_cryst1_line)
                rsa_map = self._run_dssp_backend(tmp_pdb_path)

        return rsa_map

    # Dependencies -------------------------------------------------------------
    def _init_dssp_path(self) -> str:
        """Find an existing executable file for DSSP on the computer."""
        if self.executable_path is not None:
            dssp_path_list = [self.executable_path] + self.CANDIDATES_PATHS
        else:
            dssp_path_list = self.CANDIDATES_PATHS
        dssp_path = find_file(dssp_path_list, is_software=True, name="DSSP", description=self.HELPER_LOG, verbose=self.verbose)
        self.dssp_path = dssp_path

    def _inject_cryst1_line(self, pdb_path: str) -> Union[None, str]:
        """Inject CRYST1 line in a PDB file if there is not one.
            -> If CRYST1 line is present, return None
            -> Else return a string of the PDB file with the CRYST1 line
        """

        # No need to inject CRYST1 line with mkdssp
        if self.dssp_path.endswith("mkdssp"):
            return None

        # Constants
        CRYST1_HEADER = "CRYST1"
        ATOM_HEADER = "ATOM"
        DEFAULT_CRYST1_LINE = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"

        # Read lines
        new_lines = []
        with open(pdb_path, "r") as fs:
            line = fs.readline()

            # Read lines to detect CRYST1 line
            while line:
                if line.startswith(CRYST1_HEADER): # Return None to specify that CRYST1 line is already here
                    return None
                if line.startswith(ATOM_HEADER):
                    new_lines.append(DEFAULT_CRYST1_LINE)
                    new_lines.append(line)
                    line = fs.readline()
                    break
                new_lines.append(line)
                line = fs.readline()
            
            # After injecting CRYST1 line, continue following lines
            while line:
                new_lines.append(line)
                line = fs.readline()

        # Return pdb string with injected CRYST1 line
        return "".join(new_lines)

    def _run_dssp_backend(self, pdb_path: str) -> Dict[str, float]:
        """Run DSSP software with the BioPython interface."""

        # Parse PDB with BioPython
        pdb_name = os.path.basename(pdb_path).removesuffix(".pdb")
        structure = PDBParser(QUIET=True).get_structure(pdb_name, pdb_path)
        model = structure[0]

        # Run DSSP
        if not self.verbose: # Run DSSP with WARNINGS desabled
            with suppress_stderr():
                dssp = DSSP(model, pdb_path, dssp=self.dssp_path)
        else: # Run DSSP normally
            dssp = DSSP(model, pdb_path, dssp=self.dssp_path)

        # Parse Residues
        resid_set = set()
        residues_keys = list(dssp.keys())
        rsa_map: Dict[str, float] = {}
        for res_key in residues_keys:
            chain, (res_insertion, res_id, res_alternate_location) = res_key
            resid = f"{chain}{res_insertion}{res_id}".replace(" ", "")
            if resid not in resid_set:
                res_data = dssp[res_key]
                resid_set.add(resid)
                rsa = res_data[3]
                if isinstance(rsa, float):
                    rsa = round(rsa * 100.0, 4)
                    rsa_map[resid] = rsa

        # Return
        return rsa_map

# Just to delete WARNINGS from DSSP and BioPython ------------------------------
# Because BioPython and DSSP does not provide a disable WARNINGS option ...
@contextmanager
def suppress_stderr():
    """Redirect standard error to null (with some magic)"""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr