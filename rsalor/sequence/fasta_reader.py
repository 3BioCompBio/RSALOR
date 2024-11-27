
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Union
from rsalor.sequence import Sequence


# FastaReader ------------------------------------------------------------------ 
class FastaReader:
    """Class to parse (read) fasta sequence(s) from a fasta file."""

    @classmethod
    def read_first_sequence(cls, fasta_path: str) -> Sequence:
        """Read first sequence of a fasta file."""
        reader = FastaStream(fasta_path)
        sequence = reader.get_next()
        reader.close()
        return sequence
    
    @classmethod
    def read_sequences(cls, fasta_path: str) -> List[Sequence]:
        """Read all sequences in a fasta file."""
        reader = FastaStream(fasta_path)
        sequences = reader.get_all()
        reader.close()
        return sequences
    
    @classmethod
    def count_sequences(cls, fasta_path: str) -> int:
        """Count number of sequences in a fasta file."""

        # Guardians
        assert os.path.isfile(fasta_path), f"ERROR in FastaReader.count_sequences(): fasta_path='{fasta_path}' does not exists."
        assert fasta_path.endswith(".fasta"), f"ERROR in FastaReader.count_sequences(): fasta_path='{fasta_path}' should end with '.fasta'."

        # Count
        HEADER_START_CHAR = Sequence.HEADER_START_CHAR
        n = 0
        with open(fasta_path) as fs:
            line = fs.readline()
            while line:
                if line.startswith(HEADER_START_CHAR):
                    n += 1
                line = fs.readline()
        return n


# Dependency: FastaStream ------------------------------------------------------
class FastaStream:
    """Low level class to stream fasta sequences file one by one (and avoid to cache the whole file)."""

    # Constructor --------------------------------------------------------------
    HEADER_START_CHAR = Sequence.HEADER_START_CHAR

    def __init__(self, fasta_path: str):

        # Guardians
        assert os.path.isfile(fasta_path), f"ERROR in FastaStream(): fasta_path='{fasta_path}' does not exists."
        assert fasta_path.endswith(".fasta"), f"ERROR in FastaStream(): fasta_path='{fasta_path}' should end with '.fasta'."

        # Init
        self.fasta_path = fasta_path
        self.file = open(fasta_path, "r")
        self.current_id = -1
        self.current_line = self._next_line()
        
        # First sequence sanity check
        assert self.current_line is not None, f"ERROR in FastaStream(): no sequence(s) found in file '{fasta_path}'."
        assert self._is_current_line_header(), f"ERROR in FastaStream(): first line of file '{fasta_path}' sould be a fasta header (thus start with '{self.HEADER_START_CHAR}').\nline='{self.current_line}'"

    @property
    def is_open(self) -> bool:
        """Returns True if current file/stream is still open"""
        return self.current_line is not None

    # Methods ------------------------------------------------------------------
    def close(self) -> None:
        """Close file"""
        self.file.close()
        self.current_line = None

    def get_next(self) -> Union[None, Sequence]:
        """Get next Fasta sequence"""
        if self.current_line is None:
            return None
        self.current_id += 1
        header = self.current_line.removesuffix("\n")
        seq_arr = []
        self.current_line = self._next_line()
        while self.current_line:
            if self._is_current_line_header():
                break
            seq_arr.append(self.current_line.removesuffix("\n"))
            self.current_line = self._next_line()

        seq = "".join(seq_arr)
        return Sequence(header, seq)
    
    def get_all(self) -> List[Sequence]:
        """Get all remaining Fasta sequences"""
        fasta_sequence_list = []
        fasta_sequence = self.get_next()
        while fasta_sequence is not None:
            fasta_sequence_list.append(fasta_sequence)
            fasta_sequence = self.get_next()
        return fasta_sequence_list
    
    # Dependencies -------------------------------------------------------------
    def _next_line(self) -> Union[None, str]:
        line = self.file.readline()
        if line == "":
            self.close()
            return None
        return line
    
    def _is_current_line_header(self) -> bool:
        return self.current_line.startswith(Sequence.HEADER_START_CHAR)
