
# Imports ----------------------------------------------------------------------
import os.path
from typing import Dict
import tempfile
import subprocess
from rsalor.utils import find_file
from rsalor.rsa.rsa_solver import RSASolver

# RSAMuSiC ---------------------------------------------------------------------
class RSAMuSiC(RSASolver):
    """
    RSAMuSiC(): Solver for RSA (Relative Solvent Accessibility) from a '.pdb' file using MuSiC software.
    
    usage:
        rsa_map = RSAMuSiC('./soft/MuSiC/music').run('./my_pdb.pdb')
    """

    # Constants ----------------------------------------------------------------

    # Saved condidate paths to simplify execution on different machines
    CANDIDATES_PATHS = [
        "music", "music_retro",                             # MuSiC executables
        "/home/Softs/MuSiC-4.1/music",                      # Nautilus and Santorin
    ]

    # Helper: hint to solve problem form the user: install software please
    HELPER_LOG = """-------------------------------------------------------
RSA Solver: MuSiC issue:
In order to solve Relative Solvent Accessiblity (RSA), RSALOR package uses MuSiC software.
MuSiC is our in house protein structure software (https://soft.dezyme.com/).
Please install MuSiC and specify the path to its executable or add it to system PATH.

Alternatively, if you do not have access to MuSiC, set rsa_solver='DSSP' and install DSSP (free for academic uses)
DSSP: https://swift.cmbi.umcn.nl/gv/dssp/

NOTE: you can still use the RSALOR package without MuSiC if you only want LOR values of the MSA without using RSA (just set pdb_path=None).
-------------------------------------------------------"""

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return "RSASolver['MuSiC']"
    
    def execute_solver(self, pdb_path: str) -> Dict[str, float]:
        """Compute RSA by running MuSiC: 'music -cat'

        args:
        pdb_path (str):         path to PDB file   

        output:
        {resid: str => RSA: float}     (such as {'A13': 48.57, ...})
        """

        # Init MuSiC path (check MuSiC executable existance only if software is executed)
        self._init_music_path()

        # Using temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Init
            name = os.path.basename(pdb_path).removesuffix(".pdb")
            pdb_dir = os.path.abspath(os.path.dirname(pdb_path))
            path_in_path = os.path.join(tmp_dir, "path.in")
            cat_path = os.path.join(tmp_dir, f"{name}.cat")
            log_path = os.path.join(tmp_dir, f"log_{name}.txt")

            # Generate path.in file
            path_in = "\n".join([
                f"DATA    {os.path.join(os.path.dirname(self.music_path), 'MuSiC/Data/')}",
                f"PDB     {pdb_dir}/",
                f"OUTPUT  {tmp_dir}/",
                f"CAT     {tmp_dir}/\n"
            ])
            with open(path_in_path, "w") as fs:
                fs.write(path_in)

            # Adapt run to MuSiC 4.0 or 4.1 assuming version is specified in MuSiC folder name
            music_last_folder_name = os.path.basename(os.path.dirname(self.music_path))
            sidechain_parameter = "FULLATOM" if "4.1" in music_last_folder_name else ""

            # Run MuSiC cat
            music_cmd = f"{self.music_path} -cat {name} {sidechain_parameter} {name} -init {path_in_path} -log {name}"
            if self.verbose:
                print(f" * run MuSiC command: '{music_cmd}'")
            process = subprocess.run(music_cmd.split(), shell=False, capture_output=True, text=True)

            # Check execution errors
            if process.returncode != 0:
                self._log_music_execution_error(process, log_path, music_cmd)
                raise ValueError(f"ERROR in {self}.execute_solver(): execution of MuSiC failed.")
            if not os.path.isfile(cat_path):
                self._log_music_execution_error(process, log_path, music_cmd)
                raise ValueError(f"ERROR in {self}.execute_solver(): execution of MuSiC succeeded but no output '.cat' file is generated at '{cat_path}'.")
            
            # Log output
            #if self.verbose:
            #    print("\nMuSiC logs: ")
            #    music_lines = result.stdout.split("\n")
            #    music_lines = [line for line in music_lines if len(line) > 0]
            #    print("\n".join(music_lines) + "\n")

            # Parse MuSiC -cat output
            try:
                rsa_map = self._parse_cat(cat_path)
            except:
                raise ValueError(f"ERROR in {self}.execute_solver(): failed to parse RSA from generated file '{cat_path}'.")
            assert len(rsa_map) > 0, f"ERROR in {self}.execute_solver(): no valid RSA data found in file '{cat_path}'."

        # Return
        return rsa_map
    
    # Dependencies -------------------------------------------------------------
    def _init_music_path(self) -> str:
        """Find an existing executable file for MuSiC on the computer."""
        if self.executable_path is not None:
            music_path_list = [self.executable_path] + self.CANDIDATES_PATHS
        else:
            music_path_list = self.CANDIDATES_PATHS
        music_path = find_file(music_path_list, is_software=True, name="MuSiC", description=self.HELPER_LOG, verbose=self.verbose)
        self.music_path = music_path

    def _parse_cat(self, file_path: str) -> Dict[str, float]:
        """Parse music '.cat' file and return RSA mapping: {resid: str => RSA: float}."""

        # Guardians
        assert os.path.isfile(file_path), f"ERROR in {self}._parse_cat(): file_path='{file_path}' does not exists."
        assert file_path.endswith(".cat"), f"ERROR in {self}._parse_cat(): file_path='{file_path}' should end with '.cat'."

        # Read cat file
        with open(file_path, "r") as fs:

            # Skip lines before #RESIDUES section
            line = fs.readline()
            while line and not line.startswith("#RESIDUES"):
                if line.startswith("#RESIDUES"): break
                line = fs.readline()
            line = fs.readline()

            # Read #RESIDUES section
            rsa_map: Dict[str, float] = {}
            while line:
                
                # Break after #RESIDUES section ends
                if line.startswith("#"): break

                # Parse values from line
                resid = line[0:6].replace(" ", "")
                rsa = float(line[30:40])

                # Save values
                rsa_map[resid] = rsa
                line = fs.readline()

        # Sanity chech and return
        return rsa_map
    
    def _log_music_execution_error(self, music_run_process, log_path: str, music_cmd: str) -> None:
        print("\nERROR in MuSiC execution.")
        print(" * MuSiC command: ")
        print(f" $ {music_cmd}")
        print(" * Standard Output (stdout): ")
        print(music_run_process.stdout)
        print(" * Error Output (stderr): ")
        print(music_run_process.stderr)
        print(" * Log file content: ")
        if os.path.isfile(log_path):
            with open(log_path, "r") as fs:
                log_lines = "\n".join(list(fs.readlines()))
                print(log_lines)