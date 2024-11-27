
# Imports ----------------------------------------------------------------------
from typing import List
from rsalor.utils import is_convertable_to
from rsalor.sequence import AminoAcid

# Mutation ---------------------------------------------------------------------

class Mutation:
    """Contained class for a single missence mutation."""

    # Constructor --------------------------------------------------------------
    def __init__(self, mutation_str: str):

        # Unpack and guardians
        assert len(mutation_str) >= 3, f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': should be of length 3 or more."
        wt_aa, position, mt_aa = mutation_str[0], mutation_str[1:-1], mutation_str[-1]
        assert AminoAcid.one_exists(wt_aa), f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': wild-type amino acid is incorrect."
        assert AminoAcid.one_exists(mt_aa), f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': mutant amino acid is incorrect."
        assert is_convertable_to(position, int) and int(position) > 0, f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': position must be a positive integer."

        # Set
        self.wt_aa = AminoAcid(wt_aa)
        self.position = int(position)
        self.mt_aa = AminoAcid(mt_aa)

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return f"{self.wt_aa.one}{self.position}{self.mt_aa.one}"
    
    def __int__(self) -> int:
        """Return unique integer code for each mutation."""
        return self.position*10000 + self.wt_aa.id*100 + self.mt_aa.id
    
    def is_trivial(self) -> bool:
        """Return if mutation is trivial (like 'A14A')."""
        return self.wt_aa == self.mt_aa
    
    def to_mutations(self) -> "Mutations":
        return Mutations(str(self))


class Mutations:
    """Contained class for a multiple missence mutation(s)."""

    # Constructor --------------------------------------------------------------
    def __init__(self, mutation_str: str):
        
        # Set
        self.mutations: List[Mutation] = [Mutation(m) for m in mutation_str.split(":")]
        self.mutations.sort(key=lambda m: int(m))

        # Guardians
        assert len(self) == len({m.position for m in self.mutations}), f"ERROR in Mutations('{mutation_str}'): Different single mutations have to be at different positions."

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return ":".join([str(m) for m in self.mutations])

    def __len__(self) -> int:
        return len(self.mutations)
    
    def __iter__(self):
        return iter(self.mutations)
    
    def __getitem__(self, id: int) -> Mutation:
        return self.mutations[id]
    
    def is_trivial(self) -> bool:
        return all([m.is_trivial() for m in self.mutations])
    
    def remove_trivial_mutations(self) -> "Mutations":
        """Remove all trivial mutations (like 'A12G:H13H' -> 'A12G')."""
        self.mutations = [mut for mut in self.mutations if not mut.is_trivial()]
        return self
    
    def to_mutation(self) -> "Mutation":
        assert len(self) == 1, f"ERROR in Mutations('{str(self)}'): impossible to convert Mutations of length {len(self)} to a single Mutation."
        return Mutation(str(self))