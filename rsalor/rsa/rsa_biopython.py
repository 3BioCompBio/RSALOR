
# Imports ----------------------------------------------------------------------
import os.path
from typing import Dict
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from rsalor.sequence import AminoAcid
from rsalor.rsa.rsa_solver import RSASolver

# RSAMuSiC ---------------------------------------------------------------------
class RSABiopython(RSASolver):
    """
    RSABiopython(): Solver for RSA (Relative Solvent Accessibility) from a '.pdb' file using python package biopython.
    Uses the “rolling ball” algorithm developed by Shrake & Rupley algorithm
        doc: https://biopython.org/docs/dev/api/Bio.PDB.SASA.html
    
    usage:
        rsa_map = RSABiopython().run('./my_pdb.pdb')
    """

    # Constants ----------------------------------------------------------------
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

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return "RSASolver['biopython' (Shrake & Rupley algorithm)]"
    
    def execute_solver(self, pdb_path: str) -> Dict[str, float]:
        """Compute RSA by running biopython python package: Bio.PDB.SASA: ShrakeRupley
            doc: https://biopython.org/docs/dev/api/Bio.PDB.SASA.html

        args:
        pdb_path (str):         path to PDB file   

        output:
        {resid: str => RSA: float}     (such as {'A13': 48.57, ...})
        """

        # Parse PDB file
        pdb_name = os.path.basename(pdb_path).removesuffix(".pdb")
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(pdb_name, pdb_path)

        # Compute ASA
        shrake_rupley = ShrakeRupley(
            #probe_radius=1.40, # radius of the probe in A. Default is 1.40, roughly the radius of a water molecule.
            #n_points=200,      # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.
            #radii_dict=None,   # user-provided dictionary of atomic radii to use in the calculation. Values will replace/complement those in the default ATOMIC_RADII dictionary.
        )
        shrake_rupley.compute(structure, level="R")

        # Convert to RSA and format
        rsa_map: Dict[str, float] = {}
        for chain_obj in structure[0]:
            chain = chain_obj.id
            chain_structure = structure[0][chain]
            for residue in chain_structure:
                
                # Find 'resid' = {chain}{res_position}
                (res_insertion, res_id, res_alternate_location) = residue.id
                resid = f"{chain}{res_insertion}{res_id}".replace(" ", "")

                # Get AA 3-letter code and standardize if required
                aa_three = residue.resname
                aa_three = AminoAcid._NON_STANDARD_AAS.get(aa_three, aa_three)

                # Get RSA
                asa = residue.sasa
                if isinstance(asa, float):
                    rsa_map[resid] = asa / self.get_max_surf(aa_three)
        return rsa_map

    @classmethod
    def get_max_surf(cls, aa_three: str) -> float:
        return cls.MAX_SURFACE_MAP.get(aa_three, cls.MAX_SURFACE_DEFAULT)
    