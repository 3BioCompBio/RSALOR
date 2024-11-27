
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Union
from rsalor.sequence import AminoAcid
from rsalor.sequence import Mutations, Mutation


# Sequence ----------------------------------------------------------------
class Sequence():
    """Container class for a single sequence (name and sequence as a string)."""


    # Constructor --------------------------------------------------------------
    HEADER_START_CHAR = ">"
    GAP_CHAR = "-"
    def __init__(self, name: str, sequence: str, weight: float=1.0, to_upper: bool=True):
        if name.startswith(self.HEADER_START_CHAR):
            name = name.removeprefix(self.HEADER_START_CHAR)
        if to_upper:
            sequence = sequence.upper()
        self.name: str = name
        self.sequence: str = sequence
        self.weight: float = weight
    
    # Base properties ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequence)
    
    def __str__(self) -> str:
        MAX_PRINT_LEN = 15
        seq_str = self.sequence
        if len(seq_str) > MAX_PRINT_LEN:
            seq_str = f"{seq_str[0:MAX_PRINT_LEN]}..."
        name_str = self.name
        if len(name_str) > MAX_PRINT_LEN:
            name_str = f"{name_str[0:MAX_PRINT_LEN]}..."
        return f"Sequence('{name_str}', seq='{seq_str}', l={len(self)})"
    
    def __eq__(self, other: "Sequence") -> bool:
        return self.sequence == other.sequence
    
    def __neq__(self, other: "Sequence") -> bool:
        return self.sequence != other.sequence
    
    def __hash__(self) -> int:
        return hash(self.sequence)
    
    def __iter__(self):
        return iter(self.sequence)
    
    def __getitem__(self, id: int) -> str:
        return self.sequence[id]
    
    def __contains__(self, char: str) -> bool:
        return char in self.sequence
    
    # Base Methods -------------------------------------------------------------
    def get_at_position(self, i: int) -> str:
        """Get Amino Acid (one-letter-code) at position i according to Fasta standard position id (1 = first AA, at [i-1])"""
        return self.sequence[i-1]
    
    def position_exists(self, i: int) -> bool:
        """Return is a position i exists in the sequence according to Fasta standard position id (1 = first AA)."""
        return 1 <= i <= len(self)
    
    def n_gaps(self) -> int:
        return len([char for char in self.sequence if char == self.GAP_CHAR])
    
    def n_non_gaps(self) -> int:
        return len([char for char in self.sequence if char != self.GAP_CHAR])
    
    def contains_gaps(self) -> bool:
        for char in self.sequence:
            if char == self.GAP_CHAR:
                return True
        return False
    
    def is_all_amino_acids(self) -> bool:
        for char in self.sequence:
            if not AminoAcid.one_exists(char):
                return False
        return True
    
    def to_fasta_string(self) -> str:
        """Return string of the sequence as it is in a fasta file."""
        return f"{self.HEADER_START_CHAR}{self.name}\n{self.sequence}\n"

    def get_non_gap_positions(self) -> List[bool]:
        """Get the array (List[:bool]) where a position is True if it is non-gap."""
        return [char != self.GAP_CHAR for char in self.sequence]
    
    def get_gap_positions(self) -> List[bool]:
        """Get the array (List[:bool]) where a position is True if it is gap."""
        return [char == self.GAP_CHAR for char in self.sequence]
    
    def list_single_mutations(self, exclude_trivial_mutations: bool=False, as_string: bool=False) -> List[Mutation]:
        all_aa_list = [aa.one for aa in AminoAcid.get_all()]
        mutations: List[Mutation] = []
        for i, wt_aa in enumerate(self.sequence):
            for mt_aa in all_aa_list:
                mutation = Mutation(f"{wt_aa}{i+1}{mt_aa}")
                if exclude_trivial_mutations and mutation.is_trivial():
                    continue
                mutations.append(mutation)
        if as_string:
            mutations = [str(mutation) for mutation in mutations]
        return mutations
    
    def list_double_mutations(self, exclude_trivial_mutations: bool=False, exclude_single_mutations: bool=False, as_string: bool=False) -> List[Mutations]:
        all_aa_list = [aa.one for aa in AminoAcid.get_all()]
        mutations: List[Mutations] = []
        for i1, wt_aa1 in enumerate(self.sequence):
            for i2, wt_aa2 in enumerate(self.sequence):
                if i2 <= i1: continue
                for mt_aa1 in all_aa_list:
                    for mt_aa2 in all_aa_list:
                        double_mutation = Mutations(f"{wt_aa1}{i1+1}{mt_aa1}:{wt_aa2}{i2+1}{mt_aa2}")
                        if exclude_trivial_mutations and double_mutation.is_trivial():
                            continue
                        if exclude_single_mutations and (double_mutation[0].is_trivial() ^ double_mutation[1].is_trivial()):
                            continue
                        mutations.append(double_mutation)
        if as_string:
            mutations = [str(mutation) for mutation in mutations]
        return mutations
    
    def mutations_are_compatible(self, mutations: Union[str, Mutation, Mutations]):
        """Return if mutations are compatible with the sequence."""

        # Convert any type to Mutations
        if isinstance(mutations, str):
            mutations = Mutations(mutations)
        elif isinstance(mutations, Mutation):
            mutations = mutations.to_mutations()

        # Verify each mutation
        for mutation in mutations:
            if not self.position_exists(mutation.position):
                return False
            if mutation.wt_aa.one != self.get_at_position(mutation.position):
                return False
        return True

    # IO Methods ---------------------------------------------------------------
    def write(self, fasta_path: str) -> "Sequence":
        """Save fasta in a file"""
        assert os.path.isdir(os.path.dirname(fasta_path)), f"ERROR in Sequence('{self.name}').write(): directory of '{fasta_path}' does not exists."
        assert fasta_path.endswith(".fasta"), f"ERROR in Sequence('{self.name}').write(): fasta_path='{fasta_path}' should end with '.fasta'."
        with open(fasta_path, "w") as fs:
            fs.write(self.to_fasta_string())
        return self

    # Mutate Methods -----------------------------------------------------------
    def mutated(self, mutations: Union[str, Mutation, Mutations]) -> "Sequence":
        """Return a new mutated sequence by missence mutation(s)."""

        # Uniformize input type
        if isinstance(mutations, str):
            mutations = Mutations(mutations)
        elif isinstance(mutations, Mutation):
            mutations = mutations.to_mutations()

        # Mutate
        mutated_sequence = self.sequence
        for mutation in mutations:
            position = mutation.position
            assert self.position_exists(position), f"ERROR in Sequence().mutated(): position of mutation='{mutations}' does not exists in sequence."
            assert mutation.wt_aa.one == self.get_at_position(position), f"ERROR in Sequence().mutated(): wt amino acid of mutation='{mutations}' does not match amino acid at this position ({self.get_at_position(position)})."
            mutated_sequence = mutated_sequence[:position-1] + mutation.mt_aa.one + mutated_sequence[position:]
        return Sequence(self.name, mutated_sequence)
    
    def mutate(self, mutations_str: str) -> "Sequence":
        """Mutate sequence by missence mutation(s)."""
        new_sequence = self.mutated(mutations_str)
        self.name = new_sequence.name
        self.sequence = new_sequence.sequence
        return self
    
    def trim(self, keep_positions: List[bool]) -> "Sequence":
        """Trim sequence (filter on positions) according to keep_positions (array of bool indicating which position to keep)."""
        assert len(keep_positions) == len(self), f"ERROR in {self}.trim(): length of keep_positions (={len(keep_positions)}) does not match length of sequence (={len(self)})."
        self.sequence = "".join([char for char, keep in zip(self.sequence, keep_positions) if keep])
        return self