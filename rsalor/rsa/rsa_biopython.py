
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
    MAX_SURFACE_MAP = {
        "ALA": 1.181,
        "ARG": 2.560,
        "ASN": 1.655,
        "ASP": 1.587,
        "CYS": 1.461,
        "GLN": 1.932,
        "GLU": 1.862,
        "GLY": 0.881,
        "HIS": 2.025,
        "ILE": 1.810,
        "LEU": 1.931,
        "LYS": 2.258,
        "MET": 2.034,
        "PHE": 2.228,
        "PRO": 1.468,
        "SER": 1.298,
        "THR": 1.525,
        "TRP": 2.663,
        "TYR": 2.368,
        "VAL": 1.645,
    }
    MAX_SURFACE_DEFAULT = 1.8186

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
    