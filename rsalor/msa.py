
# Imports ----------------------------------------------------------------------
import os.path
from typing import Union, List, Dict, Literal, Callable
import tempfile
import numpy as np
from rsalor.utils import time_str
from rsalor.sequence import AminoAcid
from rsalor.sequence import Mutation
from rsalor.sequence import Sequence
from rsalor.sequence import FastaReader, FastaStream
from rsalor.sequence import PairwiseAlignment
from rsalor.structure import Structure
from rsalor.weights import compute_weights, read_weights, write_weights
from rsalor.utils import CSV
from rsalor.utils import Logger


# Main -------------------------------------------------------------------------
class MSA:
    """Class MSA: Combines structural data (RSA) and evolutionary data (Log Odd Ratio, LOR from MSA) to evaluate missense mutations in proteins.
    Main class of the RSALOR package.
    """


    # Constants ----------------------------------------------------------------
    N_STATES = len(AminoAcid.ONE_2_ID) + 1
    GAP_ID = N_STATES - 1
    GAP_CHAR = AminoAcid.GAP_ONE
    ONE_2_ID = {aa_one: aa_id for aa_one, aa_id in AminoAcid.ONE_2_ID.items()}
    ONE_2_ID_GAP = {aa_one: aa_id for aa_one, aa_id in AminoAcid.ONE_2_ID.items()}
    ONE_2_ID_GAP[GAP_CHAR] = GAP_ID


    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            msa_path: str,
            pdb_path: Union[None, str]=None,
            chain: Union[None, str]=None,
            theta_regularization: float=0.01,
            n_regularization: float=0.0,
            count_target_sequence: bool=True,
            remove_redundant_sequences: bool=True,
            seqid_weights: Union[None, float]=0.80,
            min_seqid: Union[None, float]=0.35,
            num_threads: int=1,
            rsa_solver: Literal["biopython", "DSSP", "MuSiC"]="biopython",
            rsa_solver_path: Union[None, str]=None,
            trimmed_msa_path: Union[None, str]=None,
            allow_msa_overwrite: bool=False,
            weights_cache_path: Union[None, str]=None,
            rsa_cache_path: Union[None, str]=None,
            verbose: bool=False,
            disable_warnings: bool=False,
            name: Union[None, str]=None,
        ):
        """\nRSA*LOR: Combines structural data (RSA) and evolutionary data (Log Odd Ratio, LOR from MSA) to evaluate missense mutations in proteins.
        
    ----------------------------------------------------------------------------
    usage (Python):
      from rsalor import MSA                        # Import pip package
      msa = MSA('./msa1.fasta', './pdb1.pdb', 'A')  # Initialize MSA object with an MSA file, a PDB file and corresponding chain in the PDB
      scores = msa.get_scores()                     # Compute RSA*LOR scores of all single-site missense mutations
      msa.save_scores("./msa1_scores.csv")          # Save scores to a '.csv' file

    ----------------------------------------------------------------------------
    Main arguments:
      msa_path (str)                            path to MSA '.fasta' file

    Structure arguments:
      pdb_path (None | str, None)               path to PDB '.pdb' file (leave empty to ignore structure)
      chain (None | str, None)                  chain in the PDB to consider
    
    LOR/LR arguments:
      theta_regularization (float, 0.01)        regularization term for LOR/LR at amino acid frequencies level
      n_regularization (float, 0.0)             regularization term for LOR/LR at amino acid counts level
      count_target_sequence (bool, True)        count target (first) sequence of the MSA in frequencies
      remove_redundant_sequences (bool, True)   pre-process MSA to remove redundent sequences
      seqid_weights (None | float, 0.80)        seqid threshold to consider two sequences in the same cluster for weighting (set None to ignore)
      min_seqid (None | float, 0.35)            sequences which seqid with target sequence is below will be discarded (set None to ignore)
      num_threads (int, 1)                      number of threads (CPUs) for weights evaluation (in the C++ backend)

    RSA arguments:
      rsa_solver ('biopython'/'DSSP'/'MuSiC')   used solver to compute RSA (DSSP and MuSiC require the software to be installed)
      rsa_solver_path (None | str, None)        path to DSSP/MuSiC executable to compute RSA (leave empty if software is in system PATH)
      
    Files management arguments:
      trimmed_msa_path (None | str, None)       set to save the trimmed + non-redundent MSA file (leave empty to ignore)
      allow_msa_overwrite (bool, False)         allow to overwrite initial MSA file with the trimmed + non-redundent MSA file
      
    Cache arguments:
      weights_cache_path (None | str, None)     set to read (is file exists) or write (is files does not exists) weights (leave empty to ignore)
      rsa_cache_path (None | str, None)         set to read (is file exists) or write (is files does not exists) rsa values (leave empty to ignore)

    Logging arguments:
      verbose (bool, False)                     log execution steps
      disable_warnings (bool, False)            disable logging of Warnings
      name (None | str, None)                   name of the MSA object (for logging)
        """

        # MSA path Guardians
        self.name = "" # Required for logs, so we set directly.
        self._verify_input_msa_path(msa_path)

        # Fill basic properties
        self.msa_path: str = msa_path
        self.msa_filename: str = os.path.basename(self.msa_path)
        self.name: str = name
        if self.name is None:
            self.name = self.msa_filename.removesuffix(".fasta")
        self.pdb_path: str = pdb_path
        self.chain: str = chain
        self.rsa_solver: str = rsa_solver
        self.rsa_solver_path: str = rsa_solver_path
        self.rsa_cache_path: str = rsa_cache_path
        self.theta_regularization: float = theta_regularization
        self.n_regularization: float = n_regularization
        self.remove_redundant_sequences: bool = remove_redundant_sequences
        self.count_target_sequence: bool = count_target_sequence
        self.seqid_weights: Union[None, float] = seqid_weights
        self.min_seqid: Union[None, float] = min_seqid
        self.num_threads: int = num_threads
        self.weights_cache_path: str = weights_cache_path
        self.trimmed_msa_path: Union[None, str] = trimmed_msa_path
        self.allow_msa_overwrite: bool = allow_msa_overwrite
        self.verbose: bool = verbose
        self.disable_warnings: bool = disable_warnings
        self.logger = Logger(verbose, disable_warnings, step_prefix="RSALOR", warning_note=f" in {self}", error_note=f" in {self}")

        # Init structure (if pdb_path is specified)
        self._init_structure()

        # Read sequences
        self._read_sequences()

        # Filter sequences that are too far from target sequence
        if min_seqid is not None:
            self._remove_far_seqid_sequences()

        # Align Structure and Sequence (if pdb_path is specified)
        self._align_structure_to_sequence()

        # Save trimmed MSA (if trimmed_msa_path is specified)
        if self.trimmed_msa_path is not None:
            self.logger.step("save trimmed MSA (without target sequence gaps and non-std AAs, without redundent sequences) to a file.")
            self.logger.log(f" * trimmed_msa_path: '{trimmed_msa_path}'")
            self._verify_trimmed_seq_path()
            self.write(trimmed_msa_path)

        # Assign weights
        self._init_weights()

        # Counts and Frequencies
        self._init_counts()


    # Constructor dependencies -------------------------------------------------
    def _init_structure(self) -> None:
        """Parse PDB file and compute RSA (Relative Solvent Accessibility)."""

        # Case: pdb_path is None -> just log some warnings and continue
        if self.pdb_path is None:
            if self.chain is not None:
                warning_log = "pdb_path is not set, so structure and RSA are ignored."
                warning_log += f"   However chain is set to '{self.chain}'."
                warning_log += f"   Please specify pdb_path to consider structure and RSA."
                self.logger.warning(warning_log)
            if self.rsa_solver_path is not None:
                warning_log = "pdb_path is not set, so structure and RSA are ignored."
                warning_log += f"   However rsa_solver_path is set to '{self.rsa_solver_path}'."
                warning_log += f"   Please specify pdb_path to consider structure and RSA."
                self.logger.warning(warning_log)
            self.structure = None
            return None
        
        # Set Structure
        self.logger.step(f"parse PDB structure '{os.path.basename(self.pdb_path)}' (chain '{self.chain}') and compute RSA.")
        assert self.chain is not None, f"{self.error_prefix}: pdb_path='{self.pdb_path}' is set, so please set also the PDB chain to consider."
        self.structure = Structure(
            self.pdb_path,
            self.chain,
            rsa_solver=self.rsa_solver,
            rsa_solver_path=self.rsa_solver_path,
            rsa_cache_path=self.rsa_cache_path,
            verbose=self.verbose,
        )

        # Non assigned RSA warnings
        self._verify_rsa_values()

    def _read_sequences(self) -> None:
        """Read sequences from MSA FASTA file."""
        
        # Read MSA
        self.logger.step(f"read sequences from MSA file '{self.msa_filename}'.")

        # Inspect target sequence for gaps and non-standard AAs
        # Also set up alignment between MSA and trimmed MSA positions
        target_sequence = FastaReader.read_first_sequence(self.msa_path)
        self.fasta_to_fasta_trimmed: Dict[str, str] = {}
        self.fasta_trimmed_to_fasta: Dict[str, str] = {}
        tgt_seq_len = len(target_sequence)
        n_gaps = 0
        non_standard = []
        keep_position: List[bool] = []
        i_res_trimmed = 0
        for i_res, res in enumerate(target_sequence):
            if res in self.ONE_2_ID: # Standard AA -> keep
                fasta_res = str(i_res+1)
                fasta_trimmed_res = str(i_res_trimmed+1)
                self.fasta_to_fasta_trimmed[fasta_res] = fasta_trimmed_res
                self.fasta_trimmed_to_fasta[fasta_trimmed_res] = fasta_res
                i_res_trimmed += 1
                keep_position.append(True)
            elif res == self.GAP_CHAR: # Gap -> remove
                n_gaps += 1
                keep_position.append(False)
            else: # Other -> remove
                non_standard.append(res)
                keep_position.append(False)
        n_remove = n_gaps + len(non_standard)
        do_trimming = n_remove > 0
        n_keep = len(target_sequence) - n_remove
        if n_keep < 1:
            raise ValueError(f"{self.error_prefix}: target sequence does not contain any standard amino acid residues.")
        if do_trimming:
            self.logger.warning(f"target sequence contains some gaps or non-standard amino acids: MSA will be trimmed: {len(target_sequence)} -> {n_keep} (num trimmed positions: {n_remove}).")
        if n_gaps > 0:
            self.logger.warning(f"target sequence contains {n_gaps} gaps -> those positions will be trimmed.")
        if len(non_standard) > 0:
            non_std_str = "".join(non_standard)
            if len(non_std_str) > 10:
                non_std_str = non_std_str[0:7] + "..."
            self.logger.warning(f"target sequence contains {len(non_standard)} non-standard amino acids ('{non_std_str}') -> those positions will be trimmed.")

        # Read sequences from file
        self.sequences: List[Sequence] = []
        fasta_stream = FastaStream(self.msa_path) # Caution with this one
        n_tot_sequences = 0
        # Keep redundant sequences
        if not self.remove_redundant_sequences:
            sequence = fasta_stream.get_next()
            while sequence is not None:
                self._verify_sequence_length(sequence, tgt_seq_len, n_tot_sequences)
                if do_trimming:
                    sequence.trim(keep_position)
                if len(sequence) == 0:
                    continue
                self.sequences.append(sequence)
                sequence = fasta_stream.get_next()
                n_tot_sequences += 1
        # Keep only non-redundant sequences
        # the filter is done during execution to optimize time and RAM (could help with huge MSAs)
        else:
            sequences_set = set()
            sequence = fasta_stream.get_next()
            while sequence is not None:
                self._verify_sequence_length(sequence, tgt_seq_len, n_tot_sequences)
                if do_trimming:
                    sequence.trim(keep_position)
                if len(sequence) == 0:
                    continue
                sequence_str = sequence.sequence
                if sequence_str not in sequences_set:
                    self.sequences.append(sequence)
                    sequences_set.add(sequence_str)
                sequence = fasta_stream.get_next()
                n_tot_sequences += 1
            self.logger.log(f" * remove redundant sequences  : {n_tot_sequences} -> {len(self.sequences)}")
        fasta_stream.close()

        # Verify MSA consisency
        assert self.depth > 1, f"{self.error_prefix}: MSA contains no or only 1 sequence."
        assert self.length > 0, f"{self.error_prefix}: MSA target (first) sequence is of length 0."

        # Log
        self.logger.log(f" * MSA length (tgt seq length) : {len(self.target_sequence)}")
        self.logger.log(f" * MSA depth (num sequences)   : {len(self.sequences)}")

        # Set target sequence name
        self.target_sequence.name += " (trimmed MSA)"

    def _remove_far_seqid_sequences(self) -> None:
        """Filter sequences that are too far from target sequence by sequence identity."""

        # Guardian
        assert 0.0 <= self.min_seqid < 1.0, f"{self.error_prefix}: min_seqid={self.min_seqid} should be stricktly between 0 and 1."

        # Log
        self.logger.step(f"filter sequences that are too far from target sequence.")

        # Compute sequences to keep
        keep_sequences: List[Sequence] = []
        target_sequence_str = self.sequences[0].sequence
        for current_sequence in self.sequences:
            current_sequence_str = current_sequence.sequence

            # Compute seqid with target sequence
            current_seqid = self._seqid_to_target(target_sequence_str, current_sequence_str)

            if current_seqid > self.min_seqid:
                keep_sequences.append(current_sequence)

        # Update MSA sequences
        l1, l2 = len(self.sequences), len(keep_sequences)
        self.sequences = keep_sequences

        # Log results
        self.logger.log(f" * filter: {l1} -> {l2} (min_seqid={self.min_seqid:.2f})")

        # Guardians
        if l2 == 0:
            error_log = f"{self.error_prefix}: remove_far_seqid_sequences(): no sequence left."
            error_log += f"\n - No sequences left in the MSA after removing sequences that are too far from target sequence (by sequence indentity)"
            error_log += f"\n - min_seqid={self.min_seqid}: please increase value or set to None."
            raise ValueError(error_log)
        
    def _seqid_to_target(self, seq1: str, seq2: str) -> float:
        """Computes sequence identity between two sequences in the MSA."""
        gap = self.GAP_CHAR
        num_identical_residues = sum([int(aa1 == aa2) for aa1, aa2 in zip(seq1, seq2)])
        num_aligned_residues = sum([int(aa != gap) for aa in seq2])
        return num_identical_residues / num_aligned_residues
        
    def _align_structure_to_sequence(self) -> None:
        """Align residues position between PDB sequence and target sequence of the MSA."""

        # Init
        self.str_seq_align: PairwiseAlignment
        self.pdb_to_fasta_trimmed: Dict[str, str] = {}
        self.fasta_trimmed_to_pdb: Dict[str, str] = {}
        self.rsa_array: List[Union[None, float]] = [None for _ in range(self.length)]
        self.rsa_factor_array: List[Union[None, float]] = [None for _ in range(self.length)]
        if self.structure is None:
            return None
        
        # Log
        self.logger.step("align Structure (from PDB) and Sequence (from MSA).")
        
        # Init alignment
        self.str_seq_align = PairwiseAlignment(self.structure.sequence, self.target_sequence)
        
        # Map positions
        i_pdb, i_fasta_trimmed = 0, 0
        n_no_rsa, n_no_residue = 0, 0
        for aa_pdb, aa_fasta_trimmed in zip(self.str_seq_align.align1, self.str_seq_align.align2):
            if aa_pdb != self.GAP_CHAR and aa_fasta_trimmed != self.GAP_CHAR:
                residue = self.structure.chain_residues[i_pdb]
                fasta_trimmed_id = str(i_fasta_trimmed+1)
                self.pdb_to_fasta_trimmed[residue.resid] = fasta_trimmed_id
                self.fasta_trimmed_to_pdb[fasta_trimmed_id] = residue.resid
                self.rsa_array[i_fasta_trimmed] = residue.rsa
                if residue.rsa is None:
                    n_no_rsa += 1
            if aa_pdb != self.GAP_CHAR:
                i_pdb += 1
            if aa_fasta_trimmed != self.GAP_CHAR:
                if aa_pdb == self.GAP_CHAR: # Position in MSA but not is PDB
                    n_no_residue += 1
                i_fasta_trimmed += 1

        # Log
        n_assigned = len([rsa for rsa in self.rsa_array if rsa is not None])
        self.logger.log(f" * {n_assigned} / {len(self.rsa_array)} assigned RSA values for positions in trimmed MSA")

        # Set RSA factor
        self.set_rsa_factor()
        
        # Alignment Warnings
        if n_no_residue:
            self.logger.warning(f"{n_no_residue} / {len(self.rsa_array)} positions in trimmed MSA with no corresponding residues in PDB structure.")
        if n_no_rsa:
            self.logger.warning(f"{n_no_rsa} / {len(self.rsa_array)} positions in trimmed MSA corresponding to PDB residues without assigned RSA.")
        critical_alignment_warning = False
        if self.str_seq_align.mismatch > 0:
            critical_alignment_warning = True
            self.logger.warning(f"{self.str_seq_align.mismatch} / {len(self.rsa_array)} mismatch between trimmed MSA and PDB.", critical=True)
        if self.str_seq_align.internal_gap2 > 0:
            critical_alignment_warning = True
            self.logger.warning(f"{self.str_seq_align.internal_gap2} internal residues in the PDB do not correspond to a position in trimmed MSA.", critical=True)
        if critical_alignment_warning and not self.disable_warnings:
            self.str_seq_align.show(n_lines=80, only_critical_chunks=True)
            self.logger.warning("Please, make sure the first sequence in your MSA file is the target sequence to mutate.", critical=True)

    def set_rsa_factor(self, rsa_factor_function: Union[Callable[[float], float], None]=None) -> None:

        # Set default function
        if rsa_factor_function is None:
            rsa_factor_function = self.inverse_rsa

        # Log
        self.logger.step(f"set RSA factor (RSA -> w(RSA) with w='{rsa_factor_function.__name__}').")

        # Set RSA factor
        for i, rsa in enumerate(self.rsa_array):
            if rsa is not None:
                self.rsa_factor_array[i] = (1.0 - min(rsa, 100.0) / 100.0)

    def _init_weights(self) -> None:
        """Initialize weights for all sequences of the MSA (using C++ backend or from a cache file)."""

        # Case: keep all weights to 1
        if self.seqid_weights is None:
            # Put weight of first sequence to 0.0 manually to ignore it if required
            if not self.count_target_sequence:
                self.sequences[0].weight = 0.0
            return None

        # Read from cached file case
        if self.weights_cache_path is not None and os.path.isfile(self.weights_cache_path):
            self.logger.step("read weights from cached file.")
            self.logger.log(f" * weights_cache_path: '{self.weights_cache_path}'")
            weights = read_weights(self.weights_cache_path)
            if len(weights) != len(self.sequences):
                error_log = f"{self.error_prefix}: read_weights(weights_cache_path='{self.weights_cache_path}'): "
                error_log += f"\nnumber of parsed weights ({len(weights)}) does not match number of sequences ({len(self.sequences)}) in MSA."
                error_log += f"\n   * Please remove current weights_cache file and re-run weights or set weights_cache_path to None."
                raise ValueError(error_log)
        
        # Re-compute case weights case
        else:
            self.logger.step("compute weights using C++ backend.")
            dt = (0.00000000015 * self.length * self.depth**2) / self.num_threads
            dt_str = time_str(dt)
            self.logger.log(f" * seqid (to compute clusters) : {self.seqid_weights}")
            self.logger.log(f" * expected computation-time   : {dt_str} (with {self.num_threads} CPUs)")

            # Case when processed+trimmed MSA in saved 
            if self.trimmed_msa_path is not None:
                weights = compute_weights(
                    self.trimmed_msa_path,
                    self.length,
                    self.depth,
                    self.seqid_weights,
                    self.count_target_sequence,
                    self.num_threads,
                    self.verbose
                )
            # Case when processed+trimmed MSA is not saved
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_msa_path = os.path.join(tmp_dir, f"{self.name}.fasta")
                    self.write(tmp_msa_path)
                    weights = compute_weights(
                        tmp_msa_path,
                        self.length,
                        self.depth,
                        self.seqid_weights,
                        self.count_target_sequence,
                        self.num_threads,
                        self.verbose
                    )

            # Verify coherence of computed weights
            if len(weights) != len(self.sequences):
                error_log = f"{self.error_prefix}: compute_weights(): "
                error_log += f"number of computed weights ({len(weights)}) does not match number of sequences ({len(self.sequences)}) in MSA."
                raise ValueError(error_log)

        # Assign weights
        for i, wi in enumerate(weights):
            self.sequences[i].weight = wi

        # Save weights in cache file if required
        if self.weights_cache_path is not None and not os.path.isfile(self.weights_cache_path):
            self.logger.step(f"save computed weights to file '{self.weights_cache_path}'.")
            self.logger.log(f" * weights_cache_path: '{self.weights_cache_path}'")
            write_weights(weights, self.weights_cache_path)
        
    def _init_counts(self) -> None:
        """Initialize residues counts and frequences from the MSA."""

        # Log
        self.logger.step("initialize residues counts and frequencies.")

        # Set Neff
        self.Neff: float = sum([sequence.weight for sequence in self.sequences])
        self.logger.log(f" * Neff (sum of weights): {self.Neff:.2f}")

        # Counts
        self.counts = np.zeros((self.length, self.N_STATES), float)
        for sequence in self.sequences:
            for l, aa in enumerate(sequence):
                aa_id = self.ONE_2_ID.get(aa, self.GAP_ID)
                self.counts[l, aa_id] += sequence.weight
        self.gap_counts = self.counts[:, self.GAP_ID]
        self.nongap_counts = self.Neff - self.gap_counts

        # Frequencies
        self.frequencies = self.counts / self.Neff
        self.gap_frequencies = self.frequencies[:, self.GAP_ID]
        self.nongap_frequencies = 1.0 - self.gap_frequencies

        # CI (Conservation Index)
        self.global_aa_frequencies = np.sum(self.frequencies, axis=0) / self.length
        self.CI = np.sqrt(0.5 * np.sum(((self.frequencies - self.global_aa_frequencies)[:, 0:20])**2, axis=1))

        # Manage regularization and LOR/LR scores
        self.update_regularization(self.theta_regularization, self.n_regularization)

    def update_regularization(self, theta_regularization: float, n_regularization: float) -> "MSA":
        """Update regularization parameters and recompute regularized frequencies.

        Arguments:
        theta_regularization (float):  Regularization at the level of frequencies (add theta to all positional frequencies and normalize)
        n_regularization     (float):  Regularization at the level of counts (add n to all positional counts and normalize)
        """

        # Log
        self.logger.step("compute regularized frequencies.")
        self.logger.log(f" * theta_regularization : {theta_regularization}")
        self.logger.log(f" * n_regularization     : {n_regularization}")

        # Regularization Guardians
        assert theta_regularization >= 0.0, f"{self.error_prefix}: theta_regularization={theta_regularization} should be positive."
        assert n_regularization >= 0.0, f"{self.error_prefix}: n_regularization={n_regularization} sould be positive."
        assert theta_regularization > 0.0 or n_regularization > 0.0, f"{self.error_prefix}: both theta_regularization and n_regularization can not be zero to avoid divering values."

        # Set regularization properties
        self.theta_regularization = theta_regularization
        self.n_regularization = n_regularization

        # Apply n_regularization
        self.frequencies_reg = (self.counts + self.n_regularization) / (self.Neff + (float(self.N_STATES) * self.n_regularization))

        # Apply theta_regularization
        reg_term: float = self.theta_regularization / float(self.N_STATES)
        reg_factor: float = 1.0 - self.theta_regularization
        self.frequencies_reg = reg_factor * self.frequencies_reg + reg_term

        # Compute dependent values
        self.gap_frequencies_reg = self.frequencies_reg[:, self.GAP_ID]
        self.nongap_frequencies_reg = 1.0 - self.gap_frequencies_reg

        # Set LOR and LR
        self.LR = np.log(self.frequencies_reg)
        self.LOR = np.log(self.frequencies_reg / (1.0 - self.frequencies_reg))

        return self


    # Base Properties ----------------------------------------------------------
    @classmethod
    def help(cls) -> None:
        """Log main usage (help) of MSA class in the 'rsalor' package."""
        print(cls.__init__.__doc__)
    
    def __str__(self) -> str:
        return f"MSA('{self.name}')"
    
    def __iter__(self):
        return iter(self.sequences)
    
    def __getitem__(self, id: int) -> str:
        return self.sequences[id]
    
    @property
    def target_sequence(self) -> Sequence:
        return self.sequences[0]

    @property
    def length(self) -> int:
        """Length of each sequence from the MSA."""
        return len(self.target_sequence)

    @property
    def depth(self) -> int:
        """Number of sequences in the MSA."""
        return len(self.sequences)
    
    @property
    def error_prefix(self) -> str:
        """Return error in MSA prefix."""
        return f"\033[91mERROR\033[0m in {self}"
    
    @staticmethod
    def inverse_rsa(rsa_value: float) -> float:
        return 1.0 - min(rsa_value, 100.0) / 100.0

    # Scores (such as LOR) Properties ------------------------------------------
    def get_frequency(self, residue_id: int, amino_acid_one_char: str, regularized: bool=True):
        """Get a given amino acid (regularized) frequency at a given position:

        NOTE: residue_id in FASTA convention (first position is 1) on the trimmed MSA

        Arguments:
        residue_id (int):              position index in fasta convention (first residues is 1)
        amino_acid_one_char (str):     amino acid one-letter-code or gap code '-'
        regularized (bool):            set True for regularized frequencies        
        """
        if regularized:
            return self.frequencies_reg[residue_id - 1, self.ONE_2_ID_GAP[amino_acid_one_char]]
        else:
            return self.frequencies[residue_id - 1, self.ONE_2_ID_GAP[amino_acid_one_char]]
        
    def eval_mutations(
            self,
            mutations_list: List[str],
            mutations_reference: Literal["fasta_trimmed", "fasta", "pdb"]="fasta_trimmed",
            metric: Literal["LOR", "LR"]="LOR",
            use_rsa_factor: bool=False,
            disable_wt_warning: bool=False,
        ) -> List[float]:
        """Return list of LOR (log-add ratio) or LR (log ratio) for each mutation in mutations_list
            * for a mutation: LOR('H13K') = log(freq(H, 13) / 1 - freq(H, 13)) - log(freq(K, 13) / 1 - freq(K, 13))
            * by default, position of the mutation is given in the fasta convention (first residue position is 1) on the trimmed MSA

        NOTE: mutation can be indicated in 3 different references:
            - 'fasta': residues are numbered using the FASTA convention (first residue is 1) using the input MSA target sequence as reference
            - 'fasta_trimmed': residues are numbered using the FASTA convention from the trimmed MSA (without target sequence gaps and non-std AAs)
            - 'pdb': residues are numbered as in the PDB file

        Arguments:
        mutations_list (List[str]):             list of mutations as strings
        mutations_reference (str):              "fasta_trimmed", "fasta", "pdb" to specify which mutation convention to use
        metric (str):                           "LOR" or "LR" to specify which metric to compute
        use_rsa_factor (bool):                  set True to multiply the score by the RSA factor at this position
        disable_wt_warning (bool):              set True to not throw WARNING is mutation wt-aa does not match aa in the target sequence
        """

        # Set metric
        ALLOWED_METRICS = ["LOR", "LR"]
        assert metric in ALLOWED_METRICS, f"{self.error_prefix}.eval_mutations(): metric='{metric}' should be in {ALLOWED_METRICS}."
        if metric == "LOR":
            E_matrix = self.LOR
        else:
            E_matrix = self.LR

        # Uniformize mutations to 'fasta_trimmed' reference
        ALLOWED_MUTATIONS_TYPES = ["fasta_trimmed", "fasta", "pdb"]
        assert mutations_reference in ALLOWED_MUTATIONS_TYPES, f"{self.error_prefix}: mutations_reference='{mutations_reference}' sould be in {ALLOWED_MUTATIONS_TYPES}."
        if mutations_reference == "fasta" or mutations_reference == "pdb":
            residues_map = self.fasta_to_fasta_trimmed if mutations_reference == "fasta" else self.pdb_to_fasta_trimmed
            mutations_list_converted = []
            for mutation in mutations_list:
                wt, resid, mt = mutation[0], mutation[1:-1], mutation[-1]
                if resid not in residues_map:
                    error_log = f"{self.error_prefix}.eval_mutations():"
                    error_log += f"\nMutation '{mutation}' can not be converted from '{mutations_reference}' reference to 'fasta_trimmed' reference."
                    error_log += f"\n - residue '{resid}' may be outside of the range of the MSA"
                    if mutations_reference == "pdb":
                        error_log += f"\n - residue '{resid}' may be missing in the PDB structure"
                    elif mutations_reference == "fasta":
                        error_log += f"\n - residue '{resid}' may be a gap or a non-standard amino acid in the target sequence of initial MSA"
                    raise ValueError(error_log)
                mutation_converted = wt + residues_map[resid] + mt
                mutations_list_converted.append(mutation_converted)
            mutations_list_reference = [Mutation(mut) for mut in mutations_list_converted]
        else:
            mutations_list_reference = [Mutation(mut) for mut in mutations_list]

        # Compute mutations
        dE_arr = []
        for i, mutation in enumerate(mutations_list_reference):
            assert 1 <= mutation.position <= self.length, f"{self.error_prefix}.eval_mutations(): position of mutation='{mutation}' is out of range of target sequence of the MSA."
            if not disable_wt_warning:
                aa_target = self.target_sequence[mutation.position-1]
                aa_mutation = mutation.wt_aa.one
                # Trigger incorrect wt aa warning
                if aa_mutation != aa_target:
                    mutation_description = f"'{mutation}'"
                    if mutations_reference != "fasta_trimmed":
                        mutation_description = f"{mutation_description} ('{mutations_list[i]}' in '{mutations_reference}' reference)"
                    self.logger.warning(f"eval_mutations(): mutation {mutation_description}: wt-aa does not match target sequence aa '{aa_target}'.")
            dE = E_matrix[mutation.position-1, mutation.wt_aa.id] - E_matrix[mutation.position-1, mutation.mt_aa.id]
            dE_arr.append(dE)

        # Modulate by RSA factor
        if use_rsa_factor:
            for i, (mutation, dE) in enumerate(zip(mutations_list_reference, dE_arr)):
                rsa_factor = self.rsa_factor_array[mutation.position-1]
                if rsa_factor is None:
                    dE_arr[i] = None
                else:
                    dE_arr[i] = rsa_factor * dE

        return dE_arr
    
    def get_scores(self, round_digit: Union[None, int]=None, log_results: bool=False,) -> List[dict]:
        """Compute scores (gap_freq, wt_freq, mt_freq, RSA, LOR, RSA*LOR, ...) for each single-site mutation.

        NOTE: mutation are indicated in 3 different references:
            - 'fasta': residues are numbered using the FASTA convention (first residue is 1) using the input MSA target sequence as reference
            - 'fasta_trimmed': residues are numbered using the FASTA convention from the trimmed MSA (without target sequence gaps and non-std AAs)
            - 'pdb': residues are numbered as in the PDB file

        output: List of dictionary with the scores:
            [{mutation_fasta: 'A13G', LOR: 0.4578, ...}, ...]
        """

        # Log
        self.logger.step("compute scores for all single-site mutations.")

        # Compute scores for each single site mutation
        all_aas = AminoAcid.get_all()
        scores = []
        for i, wt in enumerate(self.target_sequence.sequence):
            wt_i = AminoAcid.ONE_2_ID[wt]
            resid_fasta_trimmed = str(i+1)
            resid_fasta = self.fasta_trimmed_to_fasta[resid_fasta_trimmed]
            resid_pdb = self.fasta_trimmed_to_pdb.get(resid_fasta_trimmed, None)
            RSA = self.rsa_array[i]
            RSA_factor = self.rsa_factor_array[i]
            CI = self.CI[i]
            gap_freq = self.gap_frequencies[i]
            wt_freq = self.frequencies[i, wt_i]
            for mt_aa in all_aas:
                mt = mt_aa.one
                mt_i = mt_aa.id
                mutation_fasta_trimmed = wt + resid_fasta_trimmed + mt
                mutation_fasta = wt + resid_fasta + mt
                mutation_pdb = None
                if resid_pdb is not None:
                    mutation_pdb = wt + resid_pdb + mt
                mt_freq = self.frequencies[i, mt_i]
                LOR = self.LOR[i, wt_i] - self.LOR[i, mt_i]
                LR = self.LR[i, wt_i] - self.LR[i, mt_i]
                RSALOR, RSALR = None, None
                if RSA_factor is not None:
                    RSALOR = RSA_factor * LOR
                    RSALR = RSA_factor * LR
                score = {
                    "mutation_fasta": mutation_fasta,
                    "mutation_fasta_trimmed": mutation_fasta_trimmed,
                    "mutation_pdb": mutation_pdb,
                    "gap_freq": gap_freq,
                    "wt_freq": wt_freq,
                    "mt_freq": mt_freq,
                    "CI": CI,
                    "RSA": RSA,
                    "LOR": LOR,
                    "LR": LR,
                    "RSA*LOR": RSALOR,
                    "RSA*LR": RSALR,
                }
                scores.append(score)

        # Round float values if required
        if round_digit is not None:
            for score in scores:
                for prop in ["gap_freq", "wt_freq", "mt_freq", "CI", "RSA", "LOR", "LR", "RSA*LOR", "RSA*LR"]:
                    val = score[prop]
                    if val is not None:
                        score[prop] = round(val, round_digit)

        # Log
        if log_results:
            scores_csv = CSV(list(scores[0].keys()), name=self.name)
            scores_csv.add_entries(scores[0:40])
            scores_csv.show(n_entries=40, max_colsize=23)

        return scores
    
    def save_scores(
            self,
            scores_path: str,
            round_digit: Union[None, int]=None,
            sep: str=";",
            missing_value: Union[None, str]="XXX",
            log_results: bool=False
        ) -> List[dict]:
        """Compute scores (gap_freq, wt_freq, mt_freq, RSA, LOR, RSA*LOR, ...) for each single-site mutation and save it to scores_path as a '.csv' file.

        NOTE: mutation are indicated in 3 different references:
            - 'fasta': residues are numbered using the FASTA convention (first residue is 1) using the input MSA target sequence as reference
            - 'fasta_trimmed': residues are numbered using the FASTA convention from the trimmed MSA (without target sequence gaps and non-std AAs)
            - 'pdb': residues are numbered as in the PDB file

        output: List of dictionary with the scores:
            [{mutation_fasta: 'A13G', LOR: 0.4578, ...}, ...]
        """
        
        # Compute scores
        scores = self.get_scores(round_digit)

        # Log
        self.logger.step("save scores to a file.")
        self.logger.log(f" * scores_path: '{scores_path}'")

        # Format in CSV
        scores_properties = list(scores[0].keys())
        scores_csv = CSV(scores_properties, name=self.name)
        scores_csv.set_sep(sep)
        scores_csv.add_entries(scores)

        # Change None to missing_value
        if missing_value is not None:
            for entry in scores_csv:
                for prop in scores_properties:
                    if entry[prop] is None:
                        entry[prop] = missing_value

        # Log
        if log_results:
            scores_csv.show(n_entries=40, max_colsize=23)

        # Save and return
        if scores_path is not None:
            scores_csv.write(scores_path)
        return scores


    # IO Methods ---------------------------------------------------------------
    def write(self, msa_path: str) -> "MSA":
        """Save MSA to a FASTA MSA file."""

        # Guardians
        msa_path = os.path.abspath(msa_path)
        assert os.path.isdir(os.path.dirname(msa_path)), f"{self.error_prefix}.write(): directory of msa_path='{msa_path}' does not exists."
        assert msa_path.endswith(".fasta"), f"{self.error_prefix}.write(): msa_path='{msa_path}' should end with '.fasta'."

        # Write
        with open(msa_path, "w") as fs:
            fs.write("".join([seq.to_fasta_string() for seq in self.sequences]))
        return self
    

    # Guardians Dependencies ---------------------------------------------------
    # Helpers to verify coherence of inputs and current state

    def _verify_input_msa_path(self, msa_path: str) -> None:
        """For correct format and existance of input msa_path."""

        # Existance
        assert os.path.exists(msa_path), f"{self.error_prefix}: msa_path='{msa_path}' files does not exist."

        # FASTA format
        if not msa_path.endswith(".fasta"):
            error_log = f"{self.error_prefix}: msa_path='{msa_path}' should end with '.fasta'."
            # Hint for '.ali' format
            if msa_path.endswith(".ali"):
                error_log += f"\n * msa_path: '{msa_path}'"
                error_log += f"\n * input msa_path is expected to be a MSA file in FASTA ('.fasta') format."
                error_log += f"\n * Please convert the MSA to '.fasta' with python script: "
                error_log += "\nfrom rsalor.utils import ali_to_fasta"
                error_log += "\nali_to_fasta('./my_msa.ali', './my_msa.fasta')\n"
            raise ValueError(error_log)

    def _verify_sequence_length(self, sequence: Sequence, target_length: int, i: int) -> None:
        """For coherence of all sequences in the MSA."""
        if len(sequence) != target_length:
            seq_str = sequence.sequence
            if len(seq_str) > 40:
                seq_str = seq_str[0:37] + "..."
            error_log = f"{self.error_prefix}._read_sequences(): msa_path='{self.msa_path}':"
            error_log += f"\n -> length of sequence [{i+1}] l={len(sequence)} ('{seq_str}') does not match length of target sequence l={target_length}."
            raise ValueError(error_log)

    def _verify_trimmed_seq_path(self) -> None:
        """For coherence of trimmed_msa_path and for safety to not overwrite initial input MSA."""
        trimmed_msa_path = str(os.path.abspath(self.trimmed_msa_path))
        assert os.path.isdir(os.path.dirname(trimmed_msa_path)), f"{self.error_prefix}: directory of trimmed_msa_path='{trimmed_msa_path}' does not exists."
        assert trimmed_msa_path.endswith(".fasta"), f"{self.error_prefix}: trimmed_msa_path='{trimmed_msa_path}' should end with '.fasta'."
        if os.path.normpath(self.msa_path) == os.path.normpath(trimmed_msa_path) and not self.allow_msa_overwrite:
            error_log = f"{self.error_prefix}: trimmed_msa_path='{trimmed_msa_path}' is same as input MSA path."
            error_log == "\nIf trimmed_msa_path is set, the trimmed MSA (without target sequence gaps and non-std AAs) will be saved to this path."
            error_log += "\nWARNING: This operation will overwrite initial input MSA file."
            error_log += "\nTo continue, set argument 'allow_msa_overwrite' to True."
            raise ValueError(error_log)
        
    def _verify_rsa_values(self) -> None:
        """Warnings for non-assigned RSA residues."""
        norsa_std, norsa_non_std = 0, 0
        for residue in self.structure.chain_residues:
            if residue.rsa is None:
                if residue.amino_acid.is_standard():
                    norsa_std += 1
                else:
                    norsa_non_std += 1
        norsa = norsa_std + norsa_non_std
        if norsa > 0:
            warning_log = f"{norsa} / {len(self.structure.chain_residues)} residues with no assigned RSA values ({norsa_std} std and {norsa_non_std} non-std) in PDB target chain '{self.chain}'."
            warning_log += "\n   -> This can be caused by non-standard AAs or missing atoms."
            warning_log += "\n   -> For optimal RSA estimations, we highly recommend to 'repair' the PDB and standardize AAs."
            self.logger.warning(warning_log)
