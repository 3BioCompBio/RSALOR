
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Union
from rsalor.sequence import AminoAcid
from rsalor.sequence import Mutation

# Sequence ---------------------------------------------------------------------
class Sequence:
    """Container class for a single sequence (name, sequence and weight).
    
    usage:
    
    seq: Sequence = Sequence('seq1', 'MQIFVKTLTGKTI--T') \n
    seq_name: str = seq.name \n
    seq_str: str = seq.sequence \n
    seq.write('./fasta/seq1.fasta')
    """

    # Constants ----------------------------------------------------------------
    HEADER_START_CHAR = ">"
    GAP_CHAR = AminoAcid.GAP_ONE

    # Constructor --------------------------------------------------------------
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
    def n_gaps(self) -> int:
        """Return number of gaps in sequence."""
        return len([char for char in self.sequence if char == self.GAP_CHAR])
    
    def n_non_gaps(self) -> int:
        """Return number of non-gaps in sequence."""
        return len([char for char in self.sequence if char != self.GAP_CHAR])
    
    def gap_ratio(self) -> float:
        """Return gap ratio."""
        return self.n_gaps() / len(self)

    def contains_gaps(self) -> bool:
        """Return is sequence contains gaps."""
        for char in self.sequence:
            if char == self.GAP_CHAR:
                return True
        return False
    
    def is_all_amino_acids(self) -> bool:
        """Returns is sequence is composed of only standard amino acids."""
        for char in self.sequence:
            if not AminoAcid.one_exists(char):
                return False
        return True
    
    def to_fasta_string(self) -> str:
        """Return string of the sequence in FASTA format."""
        return f"{self.HEADER_START_CHAR}{self.name}\n{self.sequence}\n"
    
    def mutation_is_compatible(self, mutation: Union[str, Mutation]) -> bool:
        """Return if mutation is compatible with the sequence."""

        # Convert to Mutation type
        if isinstance(mutation, str):
            mutation = Mutation(mutation)

        # Verify if mutatoin position is in sequence
        if not (1 <= mutation.position <= len(self)):
            return False
        # Verify if wild-type amino acid corresponds to sequence
        if mutation.wt_aa.one != self.sequence[mutation.position-1]:
            return False
        return True

    # IO Methods ---------------------------------------------------------------
    def write(self, fasta_path: str) -> "Sequence":
        """Save sequence in a FASTA file."""

        # Guardians
        fasta_path = os.path.abspath(fasta_path)
        assert os.path.isdir(os.path.dirname(fasta_path)), f"ERROR in Sequence('{self.name}').write(): directory of '{fasta_path}' does not exists."
        assert fasta_path.endswith(".fasta"), f"ERROR in Sequence('{self.name}').write(): fasta_path='{fasta_path}' should end with '.fasta'."
        
        # Save FASTA and return self
        with open(fasta_path, "w") as fs:
            fs.write(self.to_fasta_string())
        return self

    # Mutate Methods -----------------------------------------------------------
    def trim(self, keep_positions: List[bool]) -> "Sequence":
        """Trim sequence (filter on positions) according to keep_positions (array of bool indicating which position to keep)."""

        # Guardians
        assert len(keep_positions) == len(self), f"ERROR in {self}.trim(): length of keep_positions ({len(keep_positions)}) does not match length of sequence ({len(self)})."

        # Trim and return self
        self.sequence = "".join([char for char, keep in zip(self.sequence, keep_positions) if keep])
        return self