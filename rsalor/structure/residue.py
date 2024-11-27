
# Imports ----------------------------------------------------------------------
from typing import Union
from rsalor.sequence import AminoAcid

# Main -------------------------------------------------------------------------
class Residue:
    """
    Container class for a PDB residue.
        res = Residue('A', '113', AminoAcid('K'))
    """

    def __init__(self, chain: str, position: str, amino_acid: AminoAcid, rsa: Union[None, float]=None):

        # Guardians
        assert len(chain) == 1 and chain != " ", f"ERROR in Residue(): invalid chain='{chain}'."

        # Set properties
        self.chain = chain
        self.position = position
        self.amino_acid = amino_acid
        self.rsa = rsa

    @property
    def resid(self) -> str:
        return self.chain + self.position

    def __str__(self) -> str:
        return f"Residue('{self.resid}', '{self.amino_acid.three}', RSA={self.rsa})"
    
    def __eq__(self, other: "Residue") -> bool:
        return self.resid == other.resid
    
    def __neq__(self, other: "Residue") -> bool:
        return self.resid != other.resid
