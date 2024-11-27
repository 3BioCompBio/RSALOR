
# Imports ----------------------------------------------------------------------
import os.path
import sys
from typing import Union, List, Dict
import tempfile
from contextlib import contextmanager
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from rsalor.sequence import AminoAcid
from rsalor.sequence import Sequence
from rsalor.structure import Residue
from rsalor.utils import find_file


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


# Execution --------------------------------------------------------------------
class Structure:
    """Structure container for a chain from a PDB file and its RSA values assigned by DSSP.
        uses BioPython interface (https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html) with the DSSP algorithm (https://swift.cmbi.umcn.nl/gv/dssp/)

    usage:
    structure = Structure('./my_pdb.pdb', 'A', './softwares/dssp')
    for residue in ststructure: print(residue)
    """


    # Constants ----------------------------------------------------------------
    DSSP_CANDIDATES_PATHS = ["mkdssp", "dssp"]
    DSSP_HELPER_LOG = """-------------------------------------------------------
In order to solve Relative Solvent Accessiblity (RSA), RSALOR package uses:
Python package BioPython: interface with the DSSP algorithms (https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html).
The DSSP software (free for academic use) has to be installed on your computer.
Please install DSSP (https://swift.cmbi.umcn.nl/gv/dssp/) and specify the path to its executable or add it to the system PATH.
DSSP source code can be found here: https://github.com/cmbi/hssp
Note: you can still use the RSALOR package without DSSP if you only want LOR values of the MSA without using RSA.
-------------------------------------------------------"""


    # Constructor --------------------------------------------------------------
    def __init__(self, pdb_path: str, chain: str, dssp_path: Union[None, str]=None, verbose: bool=False):

        # Guardians
        assert os.path.isfile(pdb_path), f"ERROR in Structure(): pdb_path='{pdb_path}' file does not exist."
        assert pdb_path.endswith(".pdb"), f"ERROR in Structure(): pdb_path='{pdb_path}' should end with '.pdb'."
        assert len(chain) == 1 and chain != " ", f"ERROR in Structure(): chain='{chain}' should be a string of length 1 and not ' '."

        # Init base properties
        self.pdb_path = pdb_path
        self.pdb_name = os.path.basename(pdb_path.removesuffix('.pdb'))
        self.chain = chain
        self.name = f"{self.pdb_name}_{self.chain}"
        self.verbose = verbose

        # Init Structure data
        self.residues: List[Residue] = []
        self.residues_map: Dict[str, Residue] = {}
        self.sequence: Sequence

        # Init DSSP path
        if dssp_path is not None:
            dssp_path_list = [dssp_path] + self.DSSP_CANDIDATES_PATHS
        else:
            dssp_path_list = self.DSSP_CANDIDATES_PATHS
        self.dssp_path = find_file(dssp_path_list, is_software=True, log_steps=self.verbose, name="DSSP", description=self.DSSP_HELPER_LOG)

        # Inject CRYST1 line (because DSSP has decided to reject PDBs without CRYST1 line lol wtf why)
        pdb_with_cryst1_line = self._inject_cryst1_line()
        if pdb_with_cryst1_line is None: # CRYST1 line is already in PDB
            self._run_dssp(self.pdb_path)
        else: # Inject CRYST1 line in PDB
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                tmp_pdb_path = temp_file.name
                with open(tmp_pdb_path, "w") as fs:
                    fs.write(pdb_with_cryst1_line)
                self._run_dssp(self.pdb_path)

        # Set other properties
        for residue in self.residues:
            self.residues_map[residue.resid] = residue
        seq_name = f"{self.name} (ATOM-lines)"
        seq_str = "".join(res.amino_acid.one for res in self.residues)
        self.sequence = Sequence(seq_name, seq_str)


    # Base properties ----------------------------------------------------------
    def __str__(self) -> str:
        return f"Structure('{self.name}', l={len(self)})"

    def __len__(self) -> int:
        return len(self.residues)
    
    def __contains__(self, resid: str) -> bool:
        return resid in self.residues_map
    
    def __getitem__(self, id: int) -> dict:
        return self.residues[id]
    
    def __iter__(self):
        return iter(self.residues)


    # Dependencies -------------------------------------------------------------
    def _inject_cryst1_line(self) -> Union[None, str]:
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
        with open(self.pdb_path, "r") as fs:
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
    
    def _run_dssp(self, pdb_path: str):
        """Run DSSP software with the BioPython interface."""

        # Parse PDB with BioPython
        structure = PDBParser().get_structure(self.pdb_name, pdb_path)
        model = structure[0]

        # Run DSSP
        if not self.verbose: # Run DSSP with WARNINGS desabled
            with suppress_stderr():
                dssp = DSSP(model, self.pdb_path, dssp=self.dssp_path)
        else: # Run DSSP normally
            dssp = DSSP(model, self.pdb_path, dssp=self.dssp_path)

        # Parse Residues
        resid_set = set()
        residues_keys = list(dssp.keys())
        for res_key in residues_keys:
            chain, (res_insertion, res_id, res_alternate_location) = res_key
            if chain != self.chain:
                continue
            resid = f"{res_insertion}{res_id}".replace(" ", "")
            if resid not in resid_set:
                res_data = dssp[res_key]
                resid_set.add(resid)
                aa_one = res_data[1]
                if aa_one in AminoAcid.ONE_2_ID:
                    aa = AminoAcid(res_data[1])
                else:
                    aa = AminoAcid.get_unknown()
                rsa = res_data[3]
                if isinstance(rsa, float):
                    rsa = round(rsa * 100.0, 2)
                    residue = Residue(self.chain, resid, aa, rsa)
                else:
                    residue = Residue(self.chain, resid, aa)
                self.residues.append(residue)
