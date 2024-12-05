
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Tuple, Dict, Union
from Bio.Align import PairwiseAligner
from rsalor.sequence import Sequence

# PairwiseAlignment ------------------------------------------------------------
class PairwiseAlignment:
    """Class to perform pairwise alignments based on sequence identity.
    The aim is to reconsile slightly different inputs of the same protein sequence but with possibly small incoherences like missing residues.
    Like the SEQRES and ATOM lines of a PDB.
    Or a sequence extracted from a PDB and a sequence from an MSA(for instance, MSA or PDB could cover a different range of the sequence).

    NOTE: Put sequence which is expected to contain the least gaps first
    
    usage:
    
    seq1 = FastaSequence("msa_seq", "HHALYDYEARTK") \n
    seq2 = FastaSequence("pdb_seq",   "ALYDYEART") \n
    align = PairwiseAlignment(seq1, seq2) \n
    align.show() \n
    seq1_to_seq2_id_mapping = align.get_mapping()
    """

    # Constants ----------------------------------------------------------------
    GAP_CHAR = "-"
    MATCH_CHAR = "|"
    MISMATCH_CHAR = "x"

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            sequence1: Sequence,
            sequence2: Sequence,
            match_score: float=1.0,
            mismatch_score: float=-3.0,
            open_gap_score: float=-2.5,
            extend_gap_score: float=-2.0,
            tail_gap_score: float=-2.0,
            query_insertion_multiplier: float=3.0,
        ):

        # Length Guardians
        if len(sequence1) == 0 or len(sequence2) == 0:
            print(f" * sequence1: {sequence1}")
            print(f" * sequence2: {sequence2}")
            raise ValueError("ERROR in PairwiseAlignment(): input target or query sequence can not be of length 0.")

        # Init base properties
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.len1 = len(sequence1)
        self.len2 = len(sequence2)
        self.len_min = min(self.len1, self.len2)
        self.len_max = max(self.len1, self.len2)
        self.len_ratio = self.len_min / self.len_max

        # Init aligner
        self.aligner = PairwiseAligner()
        self.aligner.mode = 'global'
        self.aligner.match_score = match_score
        self.aligner.mismatch_score = mismatch_score
        self.aligner.target_internal_open_gap_score = open_gap_score
        self.aligner.target_internal_extend_gap_score = extend_gap_score
        self.aligner.target_right_open_gap_score = tail_gap_score
        self.aligner.target_right_extend_gap_score = tail_gap_score
        self.aligner.target_left_open_gap_score = tail_gap_score
        self.aligner.target_left_extend_gap_score = tail_gap_score
        self.aligner.query_internal_open_gap_score = open_gap_score * query_insertion_multiplier
        self.aligner.query_internal_extend_gap_score = extend_gap_score * query_insertion_multiplier
        self.aligner.query_left_open_gap_score = tail_gap_score
        self.aligner.query_left_extend_gap_score = tail_gap_score
        self.aligner.query_right_open_gap_score = tail_gap_score
        self.aligner.query_right_extend_gap_score = tail_gap_score

        # Align
        alignments = self.aligner.align(self.sequence1.sequence, self.sequence2.sequence)
        try: # For Biopython versions 1.80 and later
            self.align1: str = alignments[0][0]
            self.align2: str = alignments[0][1]
        except: # For legacy Biopython versions
            alignment_str_list = str(alignments[0]).split()
            self.align1: str = alignment_str_list[0]
            self.align2: str = alignment_str_list[2]
        self.score: float = alignments.score
        
        # Alignment properties
        self.match: int = 0
        self.gap: int = 0
        self.mismatch: int = 0
        comparator_list = []
        for aa1, aa2 in zip(self.align1, self.align2):
            if aa1 == self.GAP_CHAR or aa2 == self.GAP_CHAR:
                self.gap += 1
                comparator_list.append(self.GAP_CHAR)
            elif aa1 == aa2:
                self.match += 1
                comparator_list.append(self.MATCH_CHAR)
            else:
                self.mismatch += 1
                comparator_list.append(self.MISMATCH_CHAR)
        self.comparator = "".join(comparator_list)
        self.match_ratio: float = self.match / len(self)
        self.gap_ratio: float = self.gap / len(self)
        self.mismatch_ratio: float = self.mismatch / len(self)

        # Failed alignment error
        if self.match == 0:
            print("PairwiseAlignment(): failed to align sequences: zero matching positions is the alignemnt.")
            print(f" * sequence1: {sequence1}")
            print(f" * sequence2: {sequence2}")
            raise ValueError("ERROR in PairwiseAlignment(): alignment failed.")

        # Count gap types
        self.left_gap, self.right_gap = _count_tail_characters(self.comparator, self.GAP_CHAR)
        self.tail_gap: int = self.left_gap + self.right_gap
        self.internal_gap: int = self.gap - self.tail_gap

        # Count gap types by sequence
        self.gap1 = len(self) - self.len1
        self.gap2 = len(self) - self.len2
        self.left_gap1, self.right_gap1 = _count_tail_characters(self.align1, self.GAP_CHAR)
        self.tail_gap1: int = self.left_gap1 + self.right_gap1
        self.internal_gap1: int = self.gap1 - self.tail_gap1
        self.left_gap2, self.right_gap2 = _count_tail_characters(self.align2, self.GAP_CHAR)
        self.tail_gap2: int = self.left_gap2 + self.right_gap2
        self.internal_gap2: int = self.gap2 - self.tail_gap2

        # Some final measures
        self.sequence_identity: float = self.match / (self.match + self.mismatch) # excluding gapped positions
        self.coverage1: float = (self.match + self.mismatch) / self.len1
        self.coverage2: float = (self.match + self.mismatch) / self.len2
        self.coverage: float = (self.match + self.mismatch) / len(self)

    # Base Properties ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.align1)
    
    def __str__(self) -> str:
        return f"PairwiseAlignment('{self.sequence1.name}' vs. '{self.sequence2.name}', l={len(self)}, ({self.match} |, {self.gap} -, {self.mismatch} x))"
    
    def show(self, n_lines: int=120, only_critical_chunks: bool=False) -> "PairwiseAlignment":
        """Show the complete alignemnt."""
        assert n_lines > 0, f"ERROR in {self}.show(): n_lines={n_lines} should be > 0."
        print(self)
        l = len(self)
        i = 0
        while i < l:
            range_line = f"{i+1} - {min(i+n_lines, l)}"
            ali1_line = self.align1[i:i+n_lines]
            comp_line = self.comparator[i:i+n_lines]
            ali2_line = self.align2[i:i+n_lines]
            if only_critical_chunks:
                comp_line = comp_line.replace(self.MISMATCH_CHAR, f"\033[91m{self.MISMATCH_CHAR}\033[0m")
                ali2_line = ali2_line.replace(self.GAP_CHAR, f"\033[91m{self.GAP_CHAR}\033[0m")
            if not only_critical_chunks or self.MISMATCH_CHAR in comp_line or self.GAP_CHAR in ali2_line:
                print(range_line)
                print(ali1_line)
                print(comp_line)
                print(ali2_line)
            i += n_lines
        return self

    # Methods ------------------------------------------------------------------
    def write(self, save_path: str) -> "PairwiseAlignment":
        """Save alignment to a '.fasta' file."""
        save_path = os.path.abspath(save_path)
        assert save_path.endswith(".fasta"), f"ERROR in {self}.write(): save_path='{save_path}' sould be a '.fasta' file."
        assert os.path.isdir(os.path.dirname(save_path)), f"ERROR in {self}.write(): directory of save_path='{save_path}' does not exists."
        align_str = f">{self.sequence1.name}\n{self.align1}\n>{self.sequence2.name}\n{self.align2}\n"
        with open(save_path, "w") as fs:
            fs.write(align_str)
        return self

    def get_mapping(
            self,
            ids1: Union[None, List[Union[str, int]]]=None,
            ids2: Union[None, List[Union[str, int]]]=None,
            reversed: bool=False,
        ) -> Dict[Union[str, int], Union[str, int]]:
        """
        Return mapping of the aligment between mathing residues from seq1 to seq2.
            * By default the ids are just consecutive integers starting at 1.

        args:
            ids1: overwrite ids for seq1 (default is [1, 2, 3, ...])
            ids2: overwrite ids for seq2 (default is [1, 2, 3, ...])
            revered: if True, give mapping from seq2 to seq1
        """

        # Init ids
        if ids1 is None:
            ids1 = list(range(1, len(self.sequence1.sequence)+1))
        else:
            assert len(ids1) == len(self.sequence1), f"ERROR in {self}.get_mapping(): length of ids1={len(ids1)} does not match length of sequence1={len(self.sequence1)}."
        if ids2 is None:
            ids2 = list(range(1, len(self.sequence2.sequence)+1))
        else:
            assert len(ids2) == len(self.sequence2), f"ERROR in {self}.get_mapping(): length of ids2={len(ids2)} does not match length of sequence2={len(self.sequence2)}."

        # Manage reversed 
        align1, align2 = self.align1, self.align2
        if reversed:
            align1, align2 = align2, align1
            ids1, ids2 = ids2, ids1

        # Generate mapping
        mapping = {}
        i1, i2 = 0, 0
        for aa1, aa2 in zip(align1, align2):
            if aa1 != self.GAP_CHAR:
                i1 += 1
            if aa2 != self.GAP_CHAR:
                i2 += 1
            if aa1 != self.GAP_CHAR and aa2 != self.GAP_CHAR:
                mapping[ids1[i1-1]] = ids2[i2-1]
        return mapping
    
    @classmethod
    def get_gaps_ranges(cls, align: str, tail_gaps: bool=True) -> List[Tuple[int, int]]:
        """Return gaps ranges of alignment string."""
        
        # Detect gaps
        gaps_ranges = []
        is_previous_gap = False
        for i, aa in enumerate(align):
            is_current_gap = aa == cls.GAP_CHAR
            # Open gap range
            if not is_previous_gap and is_current_gap:
                current_gap_rang = [i]
            # Close gap range
            elif is_previous_gap and not is_current_gap:
                current_gap_rang.append(i)
                gaps_ranges.append(current_gap_rang)
            is_previous_gap = is_current_gap
        
        # Mange right tail gap
        if align[-1] == cls.GAP_CHAR:
            current_gap_rang.append(len(align))
            gaps_ranges.append(current_gap_rang)

        # Remove tail gaps if required
        if not tail_gaps:
            if align[0] == cls.GAP_CHAR:
                gaps_ranges = gaps_ranges[1:]
            if align[-1] == cls.GAP_CHAR:
                gaps_ranges = gaps_ranges[:-1]

        return gaps_ranges

# Dependencies -----------------------------------------------------------------
def _count_tail_characters(input_sequence: str, count_char: str) -> Tuple[int, int]:
    c1, c2 = 0, 0
    # Left tail
    for char in input_sequence:
        if char == count_char:
            c1 += 1
        else:
            break
    # Right tail
    for char in input_sequence[::-1]:
        if char == count_char:
           c2 += 1
        else:
            break
    return c1, c2
