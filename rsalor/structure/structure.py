
# Imports ----------------------------------------------------------------------
import os.path
from typing import Union, List, Dict, Literal
from rsalor.sequence import AminoAcid
from rsalor.structure import Residue
from rsalor.sequence import Sequence
from rsalor.rsa import RSASolver, RSABiopython, RSADSSP, RSAMuSiC

# Execution --------------------------------------------------------------------
class Structure:
    """Structure object for parsing all Residues from ATOM lines and assign RSA (with biopython (Shrake & Rupley), DSSP or MuSiC).

    usage:
    structure = Structure('./my_pdb.pdb', 'A')
    """

    # Constants ----------------------------------------------------------------
    RSA_SOLVERS: Dict[str, RSASolver] = {
        "biopython": RSABiopython,
        "DSSP": RSADSSP,
        "MuSiC": RSAMuSiC,
    }

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            pdb_path: str,
            chain: str,
            rsa_solver: Literal["biopython", "DSSP", "MuSiC"]="biopython",
            rsa_solver_path: Union[None, str]=None,
            rsa_cache_path: Union[None, str]=None,
            verbose: bool=False,
        ):
        """Structure object for parsing all Residues from ATOM lines and assign RSA (with biopython, DSSP or MuSiC).

        arguments:
        pdb_path (str):                                   path to PDB file
        chain (str):                                      target chain in the PDB
        rsa_solver ('biopython'/'DSSP'/'MuSiC'):          solver to use to compute RSA
        rsa_solver_path (Union[None, str]=None):          path to solver executable
        rsa_cache_path (Union[None, str]=None):           path to write/read to/from RSA values
        verbose (bool=False):                             set True for logs
        """

        # Guardians
        assert os.path.isfile(pdb_path), f"ERROR in Structure(): pdb_path='{pdb_path}' file does not exist."
        assert pdb_path.endswith(".pdb"), f"ERROR in Structure(): pdb_path='{pdb_path}' should end with '.pdb'."
        assert len(chain) == 1 and chain != " ", f"ERROR in Structure(): chain='{chain}' should be a string of length 1 and not ' '."
        solver_list = list(self.RSA_SOLVERS.keys())
        assert rsa_solver in solver_list, f"ERROR in Structure(): rsa_solver='{rsa_solver}' should be in {solver_list}."

        # Init base properties
        self.pdb_path = pdb_path
        self.pdb_name = os.path.basename(self.pdb_path).removesuffix(".pdb")
        self.chain = chain
        self.name = f"{self.pdb_name}_{self.chain}"
        self.rsa_solver = rsa_solver
        self.rsa_solver_path = rsa_solver_path
        self.verbose = verbose

        # Parse structure
        self.residues: List[Residue] = []
        self.chain_residues: List[Residue] = []
        self.residues_map: Dict[str, Residue] = {}
        self._parse_structure()

        # Set sequence
        self.sequence = Sequence(f"{self.name} (PDB, ATOM-lines)", "".join(res.amino_acid.one for res in self.chain_residues))

        # Assign RSA
        solver: RSASolver = self.RSA_SOLVERS[rsa_solver]
        rsa_map = solver(self.rsa_solver_path, self.verbose).run(self.pdb_path, rsa_cache_path=rsa_cache_path)
        n_assigned = 0
        for residue in self.residues:
            resid = residue.resid
            if resid in rsa_map:
                n_assigned += 1
                residue.rsa = rsa_map[resid]

        # Log
        if self.verbose:
            print(f" * {n_assigned} / {len(self.chain_residues)} assigned RSA values for chain '{self.chain}'")


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
    
    # Deendencies --------------------------------------------------------------
    def _parse_structure(self) -> None:
        """Parse residues data from PDB file."""
        
        # Init
        model_counter = 0
        current_chain = None
        closed_chains = set()

        # Parse PDB residues
        with open(self.pdb_path, "r", encoding="ISO-8859-1") as fs:
            line = fs.readline()
            while line:
                prefix = line[0:6]
                
                # Atom line
                if prefix == "ATOM  " or prefix == "HETATM":
                    current_chain = line[21]
                    if current_chain in closed_chains: # discard ATOM line if chain is closed
                        line = fs.readline()
                        continue
                    position = line[22:27].replace(" ", "")
                    aa_three = line[17:20]
                    aa = AminoAcid.parse_three(aa_three)
                    if aa.is_unknown(): # discard non amino acid ATOM lines
                        line = fs.readline()
                        continue
                    resid = current_chain + position
                    if resid not in self.residues_map:
                        residue = Residue(current_chain, position, aa)
                        self.residues.append(residue)
                        self.residues_map[resid] = residue
                
                # Manage multiple models: consider only model 1
                elif prefix == "MODEL ":
                    model_counter += 1
                    if model_counter > 1:
                        #print(f"WARNING in {self}: PDB contains multiple models, but only model 1 will be considered.")
                        break

                # Manage closed chains: ATOMS that appears after the chain is closed are not part of the protein chain
                elif prefix == "TER   " or prefix == "TER\n":
                    if current_chain is not None:
                        closed_chains.add(current_chain)
                
                # Take next line
                line = fs.readline()
        
        # Set residues list of target chain
        self.chain_residues = [res for res in self.residues if res.chain == self.chain]

        # No target chain error
        if len(self.chain_residues) == 0:
            error_log = f"ERROR in {self}._parse_structure(): target chain '{self.chain}' not found in PDB file."
            error_log += f"\n * pdb_path: '{self.pdb_path}'"
            error_log += f"\n * num total residues: {len(self.residues)}"
            error_log += f"\n * existing chains: {list(set([res.chain for res in self.residues]))}"
            raise ValueError(error_log)