
# Imports ----------------------------------------------------------------------
import os
import gzip
import warnings
from typing import Union, List, Dict
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Structure import Structure as BPStructure
from Bio.PDB.Model import Model as BPModel
from Bio.PDB.Chain import Chain as BPChain
from Bio.PDB.Residue import Residue as BPResidue
from Bio.PDB.SASA import ShrakeRupley
from rsalor.sequence import AminoAcid
from rsalor.structure import Residue
from rsalor.sequence import Sequence
from rsalor.utils import Logger


# Execution --------------------------------------------------------------------
class Structure:
    """Structure object for parsing Residues from ATOM lines and assign RSA (using Shrake & Rupley).
    - rely on biopython parser
    - accepts '.pdb', '.ent' and '.cif' files
    - accepts '.gz' compressed files

    usage:
    structure = Structure('./my_pdb.pdb', 'A')
    """


    # Constants ----------------------------------------------------------------
    
    # Base properties
    ACCEPTED_EXTENTIONS = [
        "pdb", "ent", "cif",
        "pdb.gz", "ent.gz", "cif.gz",
    ]

    # Knowledge based
    # Maps aa-types to knowledge-based maximum surface area
    # Taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/#pone-0080635-t001
    MAX_SURFACE_MAP = {
        "ALA": 1.29,
        "ARG": 2.74,
        "ASN": 1.95,
        "ASP": 1.93,
        "CYS": 1.67,
        "GLN": 2.23,
        "GLU": 2.25,
        "GLY": 1.04,
        "HIS": 2.24,
        "ILE": 1.97,
        "LEU": 2.01,
        "LYS": 2.36,
        "MET": 2.24,
        "PHE": 2.40,
        "PRO": 1.59,
        "SER": 1.55,
        "THR": 1.55,
        "TRP": 2.85,
        "TYR": 2.63,
        "VAL": 1.74,
    }
    MAX_SURFACE_DEFAULT = 2.01 # mean value

    # Atoms values
    HYDROGEN_ATOMS_PREFIXES = ["H", "1H", "2H", "3H"]


    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            pdb_path: str,
            chain: Union[str, None],
            ignore_hydrogen_atoms: bool=True,
            rsa_solver=None, # deprecated, kept to not break existing codes
            rsa_solver_path=None, # deprecated, kept to not break existing codes
            rsa_cache_path: Union[None, str]=None,
            verbose: Union[bool, Logger]=False,
        ):
        """Structure object for parsing Residues from ATOM lines and assign RSA (using Shrake & Rupley).

        Arguments:
            pdb_path (str):                                   path to PDB file
            chain (str | None):                               target chain in the PDB (or None to ignore)
            ignore_hydrogen_atoms (bool=Tue):                 ignore hydrogen atoms to compute RSA
            rsa_cache_path (None | str=None):                 path to write/read to/from RSA values
            verbose (bool=False):                             set True for logs or provide Logger instance
        """

        # Init logger
        if isinstance(verbose, Logger):
            self.logger = verbose
        else:
            self.logger = Logger(verbose=bool(verbose), disable_warnings=not verbose)

        # Guardians
        pdb_path = str(pdb_path)
        assert os.path.isfile(pdb_path), f"ERROR in Structure(): pdb_path='{pdb_path}' file does not exist."
        if not self.has_valid_extention(pdb_path):
            raise ValueError(
                f"ERROR in Structure(): "
                f"pdb_path='{pdb_path}' should end with any of {self.ACCEPTED_EXTENTIONS}."
            )
        if chain is not None:
            assert len(chain) == 1 and chain != " ", f"ERROR in Structure(): chain='{chain}' should be a string of length 1 and not ' '."

        # Deprecation warning
        if rsa_solver is not None or rsa_solver_path is not None:
            self.logger.warning(
                "Parameters <rsa_solver> and <rsa_solver_path> are deprecated: \n"
                " -> they are ignored and only BioPython is used to resolve RSA."
            )

        # Init base properties
        self.pdb_path = pdb_path
        self.pdb_name = self.get_pdb_name(self.pdb_path)
        self.chain = chain
        if chain is not None:
            self.name = f"{self.pdb_name}_{self.chain}"
        else:
            self.name = self.pdb_name
        self.ignore_hydrogen_atoms = bool(ignore_hydrogen_atoms)
        self.rsa_solver = None # deprecated, kept to not break existing codes
        self.rsa_solver_path = None # deprecated, kept to not break existing codes
        self.rsa_cache_path = rsa_cache_path
        self.verbose = self.logger.verbose

        # Parse structure
        self.logger.log(f" * parse 3D structure")
        self.residues: List[Residue] = []
        self.chain_residues: List[Residue] = []
        self.residues_map: Dict[str, Residue] = {}
        self._parse_structure()

        # Set sequence
        self.sequence = Sequence(
            f"{self.name} (PDB, ATOM-lines)",
            "".join(res.amino_acid.one for res in self.chain_residues)
        )

        # Log
        n_assigned_in_chain = sum(isinstance(residue.rsa, float) for residue in self.chain_residues)
        self.logger.log(f" * {n_assigned_in_chain} / {len(self.chain_residues)} assigned RSA values for chain '{self.chain}'")


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


    # RSA cache -----------------------------------------------------------------
    def read_rsa_map(self, file_path: str) -> Dict[str, float]:
        """Read rsa_map cache file and return RSA mapping: {resid: str => RSA: float}."""

        # Guardians
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"ERROR in {self}.read_rsa_map(): RSA cache file file_path='{file_path}' does not exist.")

        # Parse and return
        COMMENT_CHAR = "#"
        rsa_map: Dict[str, float] = {}
        with open(file_path, "r") as fs:
            lines = [line.split() for line in fs.readlines() if len(line) >= 3 and line[0] != COMMENT_CHAR]
        for line in lines:
            if len(line) < 2: continue
            resid, rsa = line[0], line[1]
            rsa_map[resid] = float(rsa)

        # Guardian and return
        assert len(rsa_map) > 0, f"ERROR in {self}.read(): No RSA data found in file_path='{file_path}'."
        return rsa_map

    def get_rsa_map(self) -> Dict[str, float]:
        """Get RSA mapping: {resid: str => RSA: float}.
        - skips residues without assigned RSA value
        """
        return {res.resid: res.rsa for res in self.residues if res.rsa is not None}
        
    def write_rsa_map(self, file_path: str) -> None:
        """Write rsa_map to a cache file."""

        # Init file system
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        # Compute
        rsa_map = self.get_rsa_map()
        rsa_map_str = "\n".join(f"{resid} {rsa}" for resid, rsa in rsa_map.items()) + "\n"

        # Write
        with open(file_path, "w") as fs:
            fs.write(rsa_map_str)


    # Dependencies --------------------------------------------------------------
    @classmethod
    def has_valid_extention(cls, pdb_path: str) -> bool:
        """Return True if 'pdb_path' has a file extention amoung Structure.ACCEPTED_EXTENTIONS."""
        return any([str(pdb_path).endswith(f".{ext}") for ext in cls.ACCEPTED_EXTENTIONS])

    @classmethod
    def get_pdb_name(clf, pdb_path: str) -> str:
        """Get 'pdb_name' from 'pdb_path' by removing file extention and directory prefix."""
        pdb_name = os.path.basename(str(pdb_path))
        for ext in clf.ACCEPTED_EXTENTIONS:
            if pdb_name.endswith(f".{ext}"):
                pdb_name = pdb_name.removesuffix(f".{ext}")
                break
        return pdb_name

    @classmethod
    def read_biopython_structure(cls, pdb_path: str) -> BPStructure:
        """Return a BioPython Structure object from 'pdb_path' path to a 3D structure file.
            - handle '.pdb', '.ent' and '.cif' formats
            - handle '.gz' compressed files
        """

        # Guardians
        pdb_path = str(pdb_path)
        if not cls.has_valid_extention(pdb_path):
            raise ValueError(
                f"ERROR in Structure.read_biopython_structure(): "
                f"pdb_path='{pdb_path}' should end with any of {cls.ACCEPTED_EXTENTIONS}."
            )

        # Select parser
        if pdb_path.endswith(".cif") or pdb_path.endswith(".cif.gz"):
            pdb_parser = MMCIFParser(QUIET=True)
        else:
            pdb_parser = PDBParser(QUIET=True)

        # Select file handler
        if pdb_path.endswith(".gz"):
            custom_open = gzip.open
        else:
            custom_open = open

        # Parse structure with biopython
        pdb_name = cls.get_pdb_name(pdb_path)
        with custom_open(pdb_path, mode="rt", encoding="ISO-8859-1") as fs:
            bp_structure: BPStructure = pdb_parser.get_structure(pdb_name, fs)
        return bp_structure

    def _parse_structure(self) -> None:
        """Parse residues data from PDB file."""

        # Parse structure with biopython
        bp_structure: BPStructure = self.read_biopython_structure(self.pdb_path)
        bp_model_0: BPModel = bp_structure[0] # consider only model 0

        # Remove hydrogen atoms for consistency between 3D structures with and without them
        # -> for example, X-ray 3D structures have no hydrogen atoms but some AlphaFold models do
        # -> do it before evaluating SASA with biopython ShrakeRupley
        if self.ignore_hydrogen_atoms:
            for bp_chain in bp_model_0:
                for bp_residue in bp_chain:
                    atoms_to_remove = []
                    for bp_atom in bp_residue:
                        atom_id = bp_atom.id
                        if any([atom_id.startswith(hp) for hp in self.HYDROGEN_ATOMS_PREFIXES]):
                            atoms_to_remove.append(atom_id)
                    for atom_id in atoms_to_remove:
                        bp_residue.detach_child(atom_id)

        # Compute ASA
        rsa_map = None
        if self.rsa_cache_path is not None and os.path.isfile(self.rsa_cache_path):
            self.logger.log(f" * read RSA values from rsa_cache_path '{self.rsa_cache_path}'")
            rsa_map = self.read_rsa_map(self.rsa_cache_path)
            for chain in bp_model_0: # guarantee residue.sasa property to avoid eventual bugs
                for residue in chain:
                    residue.sasa = None
        else:
            self.logger.log(f" * compute RSA values using Shrake & Rupley")
            ShrakeRupley().compute(bp_model_0, level="R")
        
        # Extract residues information
        bp_chain: BPChain
        bp_residue: BPResidue
        n_residues_failed_to_parse = 0
        warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.Polypeptide")
        for bp_chain in bp_model_0:
            peptides = PPBuilder().build_peptides(bp_chain, aa_only=0) # use PPBuilder to keep only protein chains and exclude ligands
            for peptide in peptides:
                for bp_residue in peptide:
                    try:
                        residue = self._parse_bp_residue(bp_residue, rsa_map)
                    except:
                        n_residues_failed_to_parse += 1
                        continue
                    self.residues.append(residue)
                    self.residues_map[residue.resid] = residue
                    if residue.chain == self.chain:
                        self.chain_residues.append(residue)

        # Error and warning
        if self.chain is not None and len(self.chain_residues) == 0:
            raise ValueError(
                f"ERROR in {self}._parse_structure(): target chain '{self.chain}' not found in PDB file."
                f"\n * pdb_path: '{self.pdb_path}'"
                f"\n * num total residues: {len(self.residues)}"
                f"\n * existing chains: {list(set([res.chain for res in self.residues]))}"
            )
        if n_residues_failed_to_parse > 0:
            self.logger.warning(
                f"failed to parse some residues from structure:"
                f" {n_residues_failed_to_parse} / {n_residues_failed_to_parse + len(self.residues)}"
            )

        # Write RSA cache
        if self.rsa_cache_path is not None and not os.path.isfile(self.rsa_cache_path):
            self.logger.log(f" * save RSA values to rsa_cache_path '{self.rsa_cache_path}'")
            self.write_rsa_map(self.rsa_cache_path)

    def _parse_bp_residue(
            self,
            bp_residue: BPResidue,
            rsa_map: Union[None, Dict[str, float]]=None
        ) -> Residue:
        """Parse a Residue object from a BioPython Residue.
        - parses RSA values from rsa_map if it is provided
        - alternatively compute if from biopython SASA value
        """
        bp_chain: BPChain = bp_residue.get_parent()
        chain_id = str(bp_chain.id)
        position = str(bp_residue.id[1]) + str(bp_residue.id[2]).replace(" ", "")
        resid = chain_id + position
        amino_acid = AminoAcid.parse_three(str(bp_residue.get_resname()))
        if rsa_map is not None:
            rsa = rsa_map.get(resid, None)
        else:
            rsa = self._get_bp_residue_rsa(bp_residue)
        plddt = self._get_bp_residue_plddt(bp_residue)
        return Residue(chain_id, position, amino_acid, rsa=rsa, plddt=plddt)

    def _get_bp_residue_rsa(self, bp_residue: BPResidue, ) -> Union[float, None]:
        """Get RSA of a BioPython Residue
        - using biopython assigned SASA and by-AA Max surface table
        """
        sasa = bp_residue.sasa
        if sasa is None:
            return None
        aa_three = bp_residue.resname
        aa_three_standardized = AminoAcid._NON_STANDARD_AAS.get(aa_three, aa_three)
        max_surf = self.MAX_SURFACE_MAP.get(aa_three_standardized, self.MAX_SURFACE_DEFAULT)
        return float(sasa) / max_surf

    def _get_bp_residue_plddt(self, bp_residue: BPResidue) -> Union[float, None]:
        """Get pLDDT (or B-factor) of a BioPython Residue.
            First try to look at main backbone atoms N, CA or C;
            then fall back to any other atom (in order of apparition).
        """
        if "N" in bp_residue:
            return bp_residue["N"].bfactor
        if "CA" in bp_residue:
            return bp_residue["CA"].bfactor
        if "C" in bp_residue:
            return bp_residue["C"].bfactor
        for atom in bp_residue.get_atoms():
            return atom.bfactor
        return None
