
# Imports ----------------------------------------------------------------------
from typing import Union
from rsalor.sequence import AminoAcid

# Main -------------------------------------------------------------------------
class Residue:
    """Container class for a PDB residue.
    
    usage:
    res = Residue('A', '113', AminoAcid('K'))
    """

    # Constructor --------------------------------------------------------------
    def __init__(self, chain: str, position: str, amino_acid: AminoAcid, rsa: Union[None, float]=None):

        # Guardians
        assert len(chain) == 1 and chain != " ", f"ERROR in Residue(): invalid chain='{chain}'."
        if rsa is not None:
            assert rsa >= 0.0, f"ERROR in Residue(): rsa='{rsa}' should be positive."

        # Set properties
        self.chain: str = chain
        self.position: str = position
        self.amino_acid: AminoAcid = amino_acid
        self.rsa: Union[None, float] = rsa

    # Properties ---------------------------------------------------------------
    @property
    def resid(self) -> str:
        return self.chain + self.position

    def __str__(self) -> str:
        return f"Residue('{self.resid}', '{self.amino_acid.three}', RSA={self.rsa})"