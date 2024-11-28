
# Imports ----------------------------------------------------------------------
from rsalor.utils import is_convertable_to
from rsalor.sequence import AminoAcid

# Mutation ---------------------------------------------------------------------
class Mutation:
    """Container class for a single missence/synonymous mutation on a protein (FASTA) sequence.
    
    NOTE: Use FASTA residue position convention: so resdue position is an integer and starts at 1.
    NOTE: Trivial mutations are accepter (like 'A14A').

    usage:
    mutation: Mutation = Mutation('A14G')
    """

    # Constructor --------------------------------------------------------------
    def __init__(self, mutation_str: str):

        # Unpack and guardians
        assert len(mutation_str) >= 3, f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': should be of length 3 or more."
        wt_aa, position, mt_aa = mutation_str[0], mutation_str[1:-1], mutation_str[-1]
        assert AminoAcid.one_exists(wt_aa), f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': wild-type amino acid is incorrect."
        assert AminoAcid.one_exists(mt_aa), f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': mutant amino acid is incorrect."
        assert is_convertable_to(position, int), f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': position must be a stricktly positive integer."
        position = int(position)
        assert position > 0, f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': position must be a stricktly positive integer."

        # Set
        self.wt_aa: AminoAcid = AminoAcid(wt_aa)
        self.position: int = position
        self.mt_aa: AminoAcid = AminoAcid(mt_aa)

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return f"{self.wt_aa.one}{self.position}{self.mt_aa.one}"
    
    def __int__(self) -> int:
        """Return unique integer code for each mutation."""
        return self.position*10000 + self.wt_aa.id*100 + self.mt_aa.id
    
    def is_trivial(self) -> bool:
        """Return if mutation is trivial (like 'A14A')."""
        return self.wt_aa == self.mt_aa
