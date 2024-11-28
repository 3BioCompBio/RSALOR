
# Imports ----------------------------------------------------------------------
import os.path
from abc import ABC, abstractmethod
from typing import Dict, Union


# Abstract RSASolver class -----------------------------------------------------
class RSASolver(ABC):
    """
    Abstract container class for RSASolver: to compute RSA (Relative Solvent Accessibility) from a PDB file.

    usage:
        rsa_map = RSASolver('./soft/software_executable').run('./my_pdb.pdb')
    """

    # Constants ----------------------------------------------------------------
    COMMENT_CHAR = "#"

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            executable_path: Union[None, str]=None,
            verbose: bool=False,
        ):
        self.executable_path = executable_path
        self.verbose = verbose
    
    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return "RSASolver['AbstractSolver']"
    
    def run(
            self,
            pdb_path: str,
            rsa_cache_path: Union[None, str]=None,
        ) -> Dict[str, float]:
        """Compute RSA by running the solver or using the caced file.

        args:
        pdb_path (str):         path to PDB file
        rsa_cache_path (str):   path to/from where save/read RSA (if file exists, solver execution will be skipped and output directly read from file)

        output:
        {resid: str => RSA: float}     (such as {'A13': 48.57, ...})
        """
        
        # Parse RSA if cache file exists
        if rsa_cache_path is not None and os.path.isfile(rsa_cache_path):
            if self.verbose:
                print(f" * read RSA values from rsa_cache_path '{rsa_cache_path}'")
            rsa_map = self.read(rsa_cache_path)
            return rsa_map
        
        # PDB file Guardians
        assert os.path.isfile(pdb_path), f"ERROR in {self}.execute_solver(): pdb_path='{pdb_path}' file does not exists."
        assert pdb_path.endswith(".pdb"), f"ERROR in {self}.execute_solver(): pdb_path='{pdb_path}' should be a '.pdb' file."

        # Run solver
        if self.verbose:
            print(f" * execute RSA solver: {self}")
        rsa_map = self.execute_solver(pdb_path)

        # Write RSA map
        if rsa_cache_path is not None and not os.path.isfile(rsa_cache_path):
            if self.verbose:
                print(f" * save RSA values in rsa_cache_path '{rsa_cache_path}'")
            self.write(rsa_cache_path, rsa_map)

        # Return
        return rsa_map

    @abstractmethod
    def execute_solver(self, pdb_path: str) -> Dict[str, float]:
        """Compute RSA by running the solver.

        args:
        pdb_path (str):         path to PDB file   

        output:
        {resid: str => RSA: float}     (such as {'A13': 48.57, ...})
        """
        pass

    def read(self, file_path: str) -> Dict[str, float]:
        """Read rsa_map file and return RSA mapping: {resid: str => RSA: float}."""

        # Guardians
        assert os.path.isfile(file_path), f"ERROR in {self}.read(): file file_path='{file_path}' does not exist."

        # Parse and return
        rsa_map: Dict[str, float] = {}
        with open(file_path, "r") as fs:
            lines = [line.split() for line in fs.readlines() if len(line) > 2 and line[0] != self.COMMENT_CHAR]
        for line in lines:
            resid, rsa = line[0], line[1]
            rsa_map[resid] = float(rsa)

        # Guardian and return
        assert len(rsa_map) > 0, f"ERROR in {self}.read(): No RSA data found in file_path='{file_path}'."
        return rsa_map
    
    def write(self, file_path: str, rsa_map: Dict[str, float]) -> "RSASolver":
        """Write rsa_map to a file."""

        # Guardians
        file_path = os.path.abspath(file_path)
        assert os.path.isdir(os.path.dirname(file_path)), f"ERROR in {self}.write(): directory of file_path='{file_path}' does not exists."

        # Stringity
        rsa_map_str = "\n".join(f"{resid} {rsa}" for resid, rsa in rsa_map.items())

        # Write
        with open(file_path, "w") as fs:
            fs.write(rsa_map_str)

        return self