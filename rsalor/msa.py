
# Imports ----------------------------------------------------------------------
import os.path
from typing import Union, List, Dict, Literal
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


# Main -------------------------------------------------------------------------
class MSA:
    """Class MSA: computes and collects all data from the MSA and structural information from the PDB.
    Main class of the RSALOR package.
    """


    # Constants ----------------------------------------------------------------

    # Logger
    LOG_STEP = "\033[92m" + "RSALOR" + "\033[0m"
    LOG_WARNING = "\033[93m" + "WARNING" + "\033[0m"
    LOR_ERROR = "\033[91m" + "ERROR" + "\033[0m"

    # State constants
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
            dssp_path: Union[None, str]=None,
            theta_regularization: float=0.01,
            n_regularization: float=0.0,
            count_target_sequence: bool=True,
            remove_redundant_sequences: bool=True,
            use_weights: bool=True,
            seqid: float=0.80,
            num_threads: int=1,
            weights_cache_path: Union[None, str]=None,
            trimmed_msa_path: Union[None, str]=None,
            allow_msa_overwrite: bool=False,
            verbose: bool=False,
            name: Union[None, str]=None,
        ):
        """MSA: collects and computes all data from the MSA and combines with structural information from the PDB.
        
        usage:
        msa = MSA('./msa/msa1.fasta', './pdb/pdb1.pdb', 'A')

        Arguments:
        msa_path (str):                                 path to MSA '.fasta' file
        pdb_path (Union[None, str]=None):               path to PDB '.pdb' file (leave empty to ignore structure)
        chain (Union[None, str]=None):                  chain of the PDB to consider
        dssp_path (Union[None, str]=None):              path to DSSP executable to compute RSA (leave empty if 'dssp' or 'mkdssp' is in system PATH)
        theta_regularization (float=0.01):              regularization term for LOR/LR at frequency level
        n_regularization (float=0.0):                   regularization term for LOR/LR at counts level
        count_target_sequence (bool=True):              count target (first) sequence of the MSA in frequencies
        remove_redundant_sequences (bool=True):         process MSA to remove redundent sequences
        use_weights (bool=True):                        compute weights
        seqid (float=0.80):                             sequence identity to consider two sequence as close (for weights)
        num_threads (int=1):                            number of threads (CPUs) for weights evaluation (in the C++ backend)
        weights_cache_path (Union[None, str]=None):     set to read (is file exists) write (is files does not exists) weights (leave empty to ignore)
        trimmed_msa_path (Union[None, str]=None):       set to save the trimmed + non-redundent MSA file (leave empty to ignore)
        allow_msa_overwrite (bool=False):               allow to overwrite initial MSA file with the trimmed + non-redundent MSA file
        verbose (bool=False):                           log execution steps
        name (Union[None, str]=None):                   overwrite name of the MSA instance (for logging)
        """

        # MSA path Guardians
        assert msa_path.endswith(".fasta"), f"{self.LOR_ERROR} in MSA(): msa_path='{msa_path}' should end with '.fasta'."
        assert os.path.exists(msa_path), f"{self.LOR_ERROR} in MSA(): msa_path='{msa_path}' files does not exist."

        # Fill basic properties
        self.msa_path: str = msa_path
        self.msa_filename: str = os.path.basename(self.msa_path)
        self.name: str = name
        if self.name is None:
            self.name = self.msa_filename.removesuffix(".fasta")
        self.pdb_path: str = pdb_path
        self.chain: str = chain
        self.dssp_path: str = dssp_path
        self.theta_regularization: float = theta_regularization
        self.n_regularization: float = n_regularization
        self.remove_redundant_sequences: bool = remove_redundant_sequences
        self.count_target_sequence: bool = count_target_sequence
        self.use_weights: bool = use_weights
        self.seqid: float = seqid
        self.num_threads: int = num_threads
        self.weights_cache_path: str = weights_cache_path
        self.trimmed_msa_path: Union[None, str] = trimmed_msa_path
        self.allow_msa_overwrite: bool = allow_msa_overwrite
        self.verbose: bool = verbose

        # Init structure (if pdb_path is specified)
        self._init_structure()

        # Read sequences
        self._read_sequences()

        # Align Structure and Sequence (if pdb_path is specified)
        self._align_structure_to_sequence()

        # Save trimmed MSA (if trimmed_msa_path is specified)
        if self.trimmed_msa_path is not None:
            if self.verbose:
                print(f"{self.LOG_STEP}: save trimmed MSA (without target sequence gaps and non-std AAs, without redundent sequences) to a file.")
                print(f" * trimmed_msa_path: '{trimmed_msa_path}'")
            self._verify_trimmed_seq_path()
            self.write(trimmed_msa_path)

        # Assign weights
        self._init_weights()

        # Counts and Frequencies
        self._init_counts()


    # Constructor dependencies -------------------------------------------------
    def _init_structure(self) -> None:
        """Parse PDB file and compute RSA (Relative Solvent Accessibility) values with DSSP."""

        # Case: pdb_path is None -> just log some warnings and continue
        if self.pdb_path is None:
            if self.chain is not None:
                warning_log = f"{self.LOG_WARNING} in {self}: pdb_path is not set, so structure and RSA are ignored."
                warning_log += f"   However chain is set to '{self.chain}'."
                warning_log += f"   Please specify pdb_path to consider structure and RSA."
                print(warning_log)
            if self.dssp_path is not None:
                warning_log = f"{self.LOG_WARNING} in {self}: pdb_path is not set, so structure and RSA are ignored."
                warning_log += f"   However dssp_path is set to '{self.dssp_path}'."
                warning_log += f"   Please specify pdb_path to consider structure and RSA."
                print(warning_log)
            self.structure = None
            return None
        
        # Set Structure
        if self.verbose:
            print(f"{self.LOG_STEP}: parse PDB structure '{os.path.basename(self.pdb_path)}' (chain '{self.chain}') and compute RSA with DSSP.")
        assert self.chain is not None, f"{self.LOR_ERROR} in {self}: pdb_path='{self.pdb_path}' is set, so please set also the PDB chain to consider."
        self.structure = Structure(self.pdb_path, self.chain, self.dssp_path, verbose=self.verbose)

    def _read_sequences(self) -> None:
        """Read sequences from MSA FASTA file."""
        
        # Read MSA
        if self.verbose:
            print(f"{self.LOG_STEP}: read sequences from MSA file '{self.msa_filename}'.")

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
            raise ValueError(f"{self.LOR_ERROR} in {self}: target sequence does not contain any standard amino acid residues.")
        if do_trimming:
            print(f"{self.LOG_WARNING} in {self}: target sequence contains some gaps or non-standard amino acids: MSA will be trimmed: {len(target_sequence)} -> {n_keep} (num trimmed positions: {n_remove}).")
        if n_gaps > 0:
            print(f"{self.LOG_WARNING} in {self}: target sequence contains {n_gaps} gaps -> those positions will be trimmed.")
        if len(non_standard) > 0:
            non_std_str = "".join(non_standard)
            if len(non_std_str) > 10:
                non_std_str = non_std_str[0:7] + "..."
            print(f"{self.LOG_WARNING} in {self}: target sequence contains {len(non_standard)} non-standard amino acids ('{non_std_str}') -> those positions will be trimmed.")

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
                sequence_str = sequence.sequence
                if sequence_str not in sequences_set:
                    self.sequences.append(sequence)
                    sequences_set.add(sequence_str)
                sequence = fasta_stream.get_next()
                n_tot_sequences += 1
            if self.verbose:
                print(f" * remove redundant sequences  : {n_tot_sequences} -> {len(self.sequences)}")
        fasta_stream.close()

        # Verify MSA consisency
        assert self.depth > 1, f"{self.LOR_ERROR} in {self}: MSA contains no or only 1 sequence."
        assert self.length > 0, f"{self.LOR_ERROR} in {self}: MSA target (first) sequence is of length 0."

        # Log
        if self.verbose:
            print(f" * MSA length (tgt seq length) : {len(self.target_sequence)}")
            print(f" * MSA depth (num sequences)   : {len(self.sequences)}")

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
        if self.verbose:
            print(f"{self.LOG_STEP}: align Structure (from PDB) and Sequence (from MSA).")
        
        # Init alignment
        self.str_seq_align = PairwiseAlignment(self.structure.sequence, self.target_sequence)
        
        # Map positions
        i_pdb, i_fasta_trimmed = 0, 0
        for aa_pdb, aa_fasta_trimmed in zip(self.str_seq_align.align1, self.str_seq_align.align2):
            if aa_pdb != self.GAP_CHAR and aa_fasta_trimmed != self.GAP_CHAR:
                residue = self.structure.residues[i_pdb]
                fasta_trimmed_id = str(i_fasta_trimmed+1)
                self.pdb_to_fasta_trimmed[residue.resid] = fasta_trimmed_id
                self.fasta_trimmed_to_pdb[fasta_trimmed_id] = residue.resid
                self.rsa_array[i_fasta_trimmed] = residue.rsa
            if aa_pdb != self.GAP_CHAR:
                i_pdb += 1
            if aa_fasta_trimmed != self.GAP_CHAR:
                i_fasta_trimmed += 1

        # Set RSA factor
        for i, rsa in enumerate(self.rsa_array):
            if rsa is not None:
                self.rsa_factor_array[i] = (1.0 - min(rsa, 100.0) / 100.0)

        # Log
        if self.verbose:
            n_assigned = len([rsa for rsa in self.rsa_array if rsa is not None])
            print(f" * {n_assigned} / {len(self.rsa_array)} assigned RSA values for positions in trimmed MSA")

    def _init_weights(self) -> None:
        """Initialize weights for all sequences of the MSA (using C++ backend or from a cache file)."""

        # Case: keep all weights to 1
        if not self.use_weights:
            # Put weight of first sequence to 0.0 manually to ignore it if required
            if not self.count_target_sequence:
                self.sequences[0].weight = 0.0
            # Set Neff
            self.Neff: float = np.sum([s.weight for s in self.sequences])
            return None

        # Read from cached file case
        if self.weights_cache_path is not None and os.path.isfile(self.weights_cache_path):
            if self.verbose:
                print(f"{self.LOG_STEP}: read weights from cached file.")
                print(f" * weights_cache_path: '{self.weights_cache_path}'")
            weights = read_weights(self.weights_cache_path)
            if len(weights) != len(self.sequences):
                error_log = f"{self.LOR_ERROR} in {self}: read_weights(weights_cache_path='{self.weights_cache_path}'): "
                error_log += f"\nnumber of parsed weights ({len(weights)}) does not match number of sequences ({len(self.sequences)}) in MSA."
                error_log += f"\n   * Please remove current weights_cache file and re-run weights or set weights_cache_path to None."
                raise ValueError(error_log)
        
        # Re-compute case weights case
        else:
            if self.verbose:
                print(f"{self.LOG_STEP}: compute weights using C++ backend.")
                dt = (0.0000000001 * self.length * self.depth**2) / self.num_threads
                dt_str = time_str(dt)
                print(f" * seqid (to compute clusters) : {self.seqid}")
                print(f" * expected computation-time   : {dt_str} (with {self.num_threads} CPUs)")

            # Case when processed+trimmed MSA in saved 
            if self.trimmed_msa_path is not None:
                weights = compute_weights(
                    self.trimmed_msa_path,
                    self.length,
                    self.depth,
                    self.seqid,
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
                        self.seqid,
                        self.count_target_sequence,
                        self.num_threads,
                        self.verbose
                    )

            # Verify coherence of computed weights
            if len(weights) != len(self.sequences):
                error_log = f"{self.LOR_ERROR} in {self}: compute_weights(): "
                error_log += f"number of computed weights ({len(weights)}) does not match number of sequences ({len(self.sequences)}) in MSA."
                raise ValueError(error_log)

        # Assign weights
        for i, wi in enumerate(weights):
            self.sequences[i].weight = wi

        # Set Neff
        self.Neff: float = np.sum(weights)
        if self.verbose:
            print(f" * Neff (sum of weights): {self.Neff:.2f}")

        # Save weights in cache file if required
        if self.weights_cache_path is not None and not os.path.isfile(self.weights_cache_path):
            if self.verbose:
                print(f"{self.LOG_STEP}: save computed weights to file '{self.weights_cache_path}'.")
                print(f" * weights_cache_path: '{self.weights_cache_path}'")
            write_weights(weights, self.weights_cache_path)
        
    def _init_counts(self) -> None:
        """Initialize residues counts and frequences from the MSA."""

        # Log
        if self.verbose:
            print(f"{self.LOG_STEP}: initialize residues counts and frequencies.")

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
        if self.verbose:
            print(f"{self.LOG_STEP}: compute regularized frequencies.")
            print(f" * theta_regularization : {theta_regularization}")
            print(f" * n_regularization     : {n_regularization}")

        # Regularization Guardians
        assert theta_regularization >= 0.0, f"{self.LOR_ERROR} in {self}: theta_regularization={theta_regularization} should be positive."
        assert n_regularization >= 0.0, f"{self.LOR_ERROR} in {self}: n_regularization={n_regularization} sould be positive."
        assert theta_regularization > 0.0 or n_regularization > 0.0, f"{self.LOR_ERROR} in {self}: both theta_regularization and n_regularization can not be zero to avoid divering values."

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
        assert metric in ALLOWED_METRICS, f"{self.LOR_ERROR} in {self}.eval_mutations(): metric='{metric}' should be in {ALLOWED_METRICS}."
        if metric == "LOR":
            E_matrix = self.LOR
        else:
            E_matrix = self.LR

        # Uniformize mutations to 'fasta_trimmed' reference
        ALLOWED_MUTATIONS_TYPES = ["fasta_trimmed", "fasta", "pdb"]
        assert mutations_reference in ALLOWED_MUTATIONS_TYPES, f"{self.LOR_ERROR} in {self}: mutations_reference='{mutations_reference}' sould be in {ALLOWED_MUTATIONS_TYPES}."
        if mutations_reference == "fasta" or mutations_reference == "pdb":
            residues_map = self.fasta_to_fasta_trimmed if mutations_reference == "fasta" else self.pdb_to_fasta_trimmed
            mutations_list_converted = []
            for mutation in mutations_list:
                wt, resid, mt = mutation[0], mutation[1:-1], mutation[-1]
                if resid not in residues_map:
                    error_log = f"{self.LOR_ERROR} in {self}.eval_mutations():"
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
            assert 1 <= mutation.position <= self.length, f"{self.LOR_ERROR} in {self}.eval_mutations(): position of mutation='{mutation}' is out of range of target sequence of the MSA."
            if not disable_wt_warning:
                aa_target = self.target_sequence[mutation.position-1]
                aa_mutation = mutation.wt_aa.one
                # Trigger incorrect wt aa warning
                if aa_mutation != aa_target:
                    mutation_description = f"'{mutation}'"
                    if mutations_reference != "fasta_trimmed":
                        mutation_description = f"{mutation_description} ('{mutations_list[i]}' in '{mutations_reference}' reference)"
                    print(f"{self.LOG_WARNING} in {self}.eval_mutations(): mutation {mutation_description}: wt-aa does not match target sequence aa '{aa_target}'.")
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
        if self.verbose:
            print(f"{self.LOG_STEP}: compute scores for all single-site mutations.")

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
    
    def save_scores(self, scores_path: str, round_digit: Union[None, int]=None, sep: str=";", log_results: bool=False) -> List[dict]:
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
        if self.verbose:
            print(f"{self.LOG_STEP}: save scores to a file.")
            print(f" * scores_path: '{scores_path}'")

        # Format in CSV
        scores_properties = list(scores[0].keys())
        scores_csv = CSV(scores_properties, name=self.name)
        scores_csv.set_sep(sep)
        scores_csv.add_entries(scores)

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
        assert os.path.isdir(os.path.dirname(msa_path)), f"{self.LOR_ERROR} in {self}.write(): directory of msa_path='{msa_path}' does not exists."
        assert msa_path.endswith(".fasta"), f"{self.LOR_ERROR} in {self}.write(): msa_path='{msa_path}' should end with '.fasta'."

        # Write
        with open(msa_path, "w") as fs:
            fs.write("".join([seq.to_fasta_string() for seq in self.sequences]))
        return self
    

    # Guardians Dependencies ---------------------------------------------------
    # Helpers to verify coherence of inputs and current state

    def _verify_sequence_length(self, sequence: Sequence, target_length: int, i: int) -> None:
        """For coherence of all sequences in the MSA."""
        if len(sequence) != target_length:
            seq_str = sequence.sequence
            if len(seq_str) > 40:
                seq_str = seq_str[0:37] + "..."
            error_log = f"{self.LOR_ERROR} in {self}._read_sequences(): msa_path='{self.msa_path}':"
            error_log += f"\n -> length of sequence [{i+1}] l={len(sequence)} ('{seq_str}') does not match length of target sequence l={target_length}."
            raise ValueError(error_log)

    def _verify_trimmed_seq_path(self) -> None:
        """For coherence of trimmed_msa_path and for safety to not overwrite initial input MSA."""
        trimmed_msa_path = str(os.path.abspath(self.trimmed_msa_path))
        assert os.path.isdir(os.path.dirname(trimmed_msa_path)), f"{self.LOR_ERROR} in {self}: directory of trimmed_msa_path='{trimmed_msa_path}' does not exists."
        assert trimmed_msa_path.endswith(".fasta"), f"{self.LOR_ERROR} in {self}: trimmed_msa_path='{trimmed_msa_path}' should end with '.fasta'."
        if os.path.normpath(self.msa_path) == os.path.normpath(trimmed_msa_path) and not self.allow_msa_overwrite:
            error_log = f"{self.LOR_ERROR} in {self}: trimmed_msa_path='{trimmed_msa_path}' is same as input MSA path."
            error_log == "\nIf trimmed_msa_path is set, the trimmed MSA (without target sequence gaps and non-std AAs) will be saved to this path."
            error_log += "\nWARNING: This operation will overwrite initial input MSA file."
            error_log += "\nTo continue, set argument 'allow_msa_overwrite' to True."
            raise ValueError(error_log)
