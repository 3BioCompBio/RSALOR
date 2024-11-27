
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


# Main -------------------------------------------------------------------------
class MSA:
    """
    Container class for MSA.
    usage: 
        msa = MSA(msa_path)
    """

    # Constants ----------------------------------------------------------------
    N_STATES = len(AminoAcid.ONE_2_ID) + 1
    GAP_ID = N_STATES - 1
    GAP_CHAR = "-"
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
        """"""

        # Guardians
        assert msa_path.endswith(".fasta"), f"ERROR in MSA(): msa_path='{msa_path}' should end with '.fasta'."
        assert os.path.exists(msa_path), f"ERROR in MSA('{msa_path}'): msa_path='{msa_path}' files does not exist."
        assert 0.0 < seqid < 1.0, f"ERROR in MSA('{msa_path}'): seqid={seqid} (for clustering to compute weights) should be in [0, 1] excluded."
        assert num_threads > 0, f"ERROR in MSA('{msa_path}'): num_threads={num_threads} should be stricktly positive."
        if trimmed_msa_path is not None:
            assert os.path.isdir(os.path.dirname(trimmed_msa_path)), f"ERROR in MSA('{msa_path}'): directory of trimmed_msa_path='{trimmed_msa_path}' does not exists."
            assert trimmed_msa_path.endswith(".fasta"), f"ERROR in MSA('{msa_path}'): trimmed_msa_path='{trimmed_msa_path}' should end with '.fasta'."
            if os.path.normpath(msa_path) == os.path.normpath(trimmed_msa_path) and not allow_msa_overwrite:
                error_log = f"ERROR in MSA('{msa_path}'): trimmed_msa_path='{trimmed_msa_path}' is same as input MSA path."
                error_log += "\nWARNING: This operation will overwrite initial input MSA file."
                error_log += "\nTo continue, set argument 'allow_msa_overwrite' to True."
                raise ValueError(error_log)

        # Fill basic properties
        self.msa_path: str = msa_path
        self.filename: str = os.path.basename(self.msa_path)
        self.name: str = name
        if self.name is None:
            self.name = self.filename.removesuffix(".fasta")
        self.use_weights: bool = use_weights
        self.seqid: float = seqid
        self.num_threads: int = num_threads
        self.weights_cache_path: str = weights_cache_path
        self.count_target_sequence: bool = count_target_sequence
        self.remove_redundant_sequences: bool = remove_redundant_sequences
        self.trimmed_msa_path: Union[None, str] = trimmed_msa_path
        self.allow_msa_overwrite: bool = allow_msa_overwrite
        self.verbose: bool = verbose

        # Init structure
        self.structure = None
        if pdb_path is None and chain is not None:
            print(f"WARNING in {self}: pdb_path is not set, so structure and RSA are ignored. However chain is set to '{chain}'. Please specify a '.pdb' file in order to consider structure.")
        if pdb_path is not None:
            if self.verbose:
                print("MSA: parse PDB structure and obtain RSA with DSSP software.")
            assert chain is not None, f"ERROR in {self}: if pdb_path='{pdb_path}' is set, please set also the PDB chain to consider."
            self.structure = Structure(pdb_path, chain, dssp_path, verbose=self.verbose)

        # Read sequences
        self.fasta_to_fasta_trimmed: Dict[str, str] = {}
        self.fasta_trimmed_to_fasta: Dict[str, str] = {}
        self._read_sequences()

        # Align Structure and Sequence
        self.str_seq_align: PairwiseAlignment
        self.pdb_to_fasta_trimmed: Dict[str, str] = {}
        self.fasta_trimmed_to_pdb: Dict[str, str] = {}
        self.rsa_array: List[Union[None, float]] = [None for _ in range(self.length)]
        self.rsa_factor_array: List[Union[None, float]] = [None for _ in range(self.length)]
        if self.structure is not None:
            self._align_structure_to_sequence()

        # Save trimmed MSA
        if self.trimmed_msa_path is not None:
            if self.verbose:
                print("MSA: save processed MSA to file.")
            self.write(trimmed_msa_path)

        # Assign weights
        if self.use_weights:
            self._init_weights()
        # Put weight of first sequence to 0.0 manually to ignore it if required
        if not self.count_target_sequence:
            self.sequences[0].weight = 0.0
        self.Neff: float = sum([s.weight for s in self.sequences])

        # Counts and Frequencies
        self._init_counts()
        self.update_regularization(theta_regularization, n_regularization)

    # Init dependencies --------------------------------------------------------
    def _read_sequences(self) -> None:
        """Read sequences from MSA '.fasta' file."""
        
        # Read MSA
        if self.verbose:
            print("MSA: read sequences from file.")

        # Inspect target sequence for gaps and non-standard AAs
        target_sequence = FastaReader.read_first_sequence(self.msa_path)
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
            raise ValueError(f"ERROR in {self}: target sequence does not contain any standard amino acid residues.")
        if do_trimming:
            print(f"WARNING in {self}: target sequence contains some gaps or non-standard amino acids: MSA will be trimmed: {len(target_sequence)} -> {n_keep} (n_trimmed={n_remove}).")
        if n_gaps > 0:
            print(f" -> WARNING in {self}: target sequence contains {n_gaps} gaps -> those positions will be trimmed.")
        if len(non_standard) > 0:
            non_std_str = "".join(non_standard)
            if len(non_std_str) > 10:
                non_std_str = non_std_str[0:7] + "..."
            print(f" -> WARNING in {self}: target sequence contains {len(non_standard)} non-standard amino acids ('{non_std_str}') -> those positions will be trimmed.")

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
                print(f" * sequence length: {len(self.sequences[0])}")
                print(f" * remove redundant sequences: {n_tot_sequences} -> {len(self.sequences)}")
        fasta_stream.close()

        # Verify MSA consisency
        assert self.depth > 1, f"ERROR in {self}: input file contains no or only 1 sequence."
        assert self.length > 0, f"ERROR in {self}: input file's target (first) sequence is of length 0."

    def _align_structure_to_sequence(self) -> None:
        """Align residues position between PDB sequence and target sequence of the MSA."""
        
        # Log
        if self.verbose:
            print("MSA: align Structure (from PDB) and Sequence (from MSA).")
        
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

    def _init_weights(self) -> None:
        """Initialize weights for all sequences of the MSA (using C++ backend or from a cache file)."""

        # Read from cached file case
        if self.weights_cache_path is not None and os.path.isfile(self.weights_cache_path):
            if self.verbose:
                print(f"MSA: read_weights() from cached file '{self.weights_cache_path}'.")
            weights = read_weights(self.weights_cache_path)
            if len(weights) != len(self.sequences):
                error_log = f"ERROR in {self}: read_weights(weights_cache_path='{self.weights_cache_path}'): "
                error_log += f"\nnumber of parsed weights ({len(weights)}) does not match number of sequences ({len(self.sequences)}) in MSA."
                error_log += "\n   * Please remove current weights_cache file and re-run weights or set weights_cache_path to None."
                raise ValueError(error_log)
        
        # Re-compute case weights case
        else:
            if self.verbose:
                print("MSA: compute weights using C++ backend.")
                dt = (0.0000000001 * self.length * self.depth**2) / self.num_threads
                if dt > 1.0: # Short times are useless to print and unreliable
                    dt_str = time_str(dt)
                    print(f" * expected computation-time: {dt_str} (with {self.num_threads} CPUs)")

            # Case when processed MSA in saved 
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
            # Case when processed MSA is not saved
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
                error_log = f"ERROR in {self}: compute_weights(): "
                error_log += f"number of computed weights ({len(weights)}) does not match number of sequences ({len(self.sequences)}) in MSA."
                raise ValueError(error_log)
            
            # Save weights in cache file if required
            if self.weights_cache_path is not None:
                if self.verbose:
                    print(f"MSA: save computed weights to file '{self.weights_cache_path}'.")
                write_weights(weights, self.weights_cache_path)

        # Set propreties
        for i, wi in enumerate(weights):
            self.sequences[i].weight = wi
        
    def _init_counts(self) -> None:
        """Initialize residues counts and frequences from the MSA."""

        # Log
        if self.verbose:
            print("MSA: initialize residues counts and frequencies.")

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

    def update_regularization(self, theta_regularization: float, n_regularization: float) -> "MSA":
        """Update regularization parameters and recompute regularized frequencies.

        Arguments:
        theta_regularization (float):  Regularization at the level of frequencies (add theta to all positional frequencies and normalize)
        n_regularization     (float):  Regularization at the level of counts (add n to all positional counts and normalize)
        """

        # Log
        if self.verbose:
            print("MSA: compute regularized frequencies.")

        # Regularization Guardians
        assert theta_regularization >= 0.0, f""
        assert n_regularization >= 0.0, f""
        assert theta_regularization > 0.0 or n_regularization > 0.0, f"ERROR in {self}: both theta_regularization and n_regularization can not be zero to avoid divering values."

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
    
    def get_residue_frequency(self, residue_id: int, amino_acid_one_char: str, regularized: bool=True):
        """Get amino acid (regularized) frequency at a given position:

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
            mutations_list: Union[str, List[str]],
            mutations_type: Literal["fasta_trimmed", "fasta", "pdb"]="fasta_trimmed",
            metric: Literal["LOR", "LR"]="LOR",
            disable_wt_warning: bool=False,
            use_rsa_factor: bool=False,
        ) -> List[float]:
        """Return list of Logg Odd Ratios for each of the mutations.
            * for a mutation: LOR('H13K') = log(freq(H, 13) / 1 - freq(H, 13)) - log(freq(K, 13) / 1 - freq(K, 13))
            * position of the mutation is given in the fasta convention (first residue position is 1)

        Arguments:
        mutations_list: (str or List[str]):     list of mutations as strings
        disable_wt_warning (bool):              set True to not throw WARNING is mutation wt-aa does not match aa in the target sequence
        """

        # Set metric
        assert metric in ["LOR", "LR"], f""
        if metric == "LOR":
            E_matrix = self.LOR
        else:
            E_matrix = self.LR

        # Conter to list of mutations if a single mutation is given
        if isinstance(mutations_list, str):
            mutations_list = [mutations_list]

        # Uniformize mutations to 'fasta_trimmed' type
        assert mutations_type in ["fasta_trimmed", "fasta", "pdb"], f""
        if mutations_type == "fasta" or mutations_type == "pdb":
            residues_map = self.fasta_to_fasta_trimmed if mutations_type == "fasta" else self.pdb_to_fasta_trimmed
            mutations_list_converted = []
            for mutation in mutations_list:
                wt, resid, mt = mutation[0], mutation[1:-1], mutation[-1]
                if resid not in residues_map:
                    error_log = f"ERROR in {self}.eval_mutations():"
                    error_log += f"\nmutation '{mutation}' can not be converted from '{mutations_type}' to 'fasta_trimmed'."
                    error_log += f"residue '{resid}' may be missing in the PDB or in the target sequence of the MSA."
                    raise ValueError(error_log)
                mutation_converted = wt + residues_map[resid] + mt
                mutations_list_converted.append(mutation_converted)
            mutations_list = mutations_list_converted
        mutations_list: List[Mutation] = [Mutation(mut) for mut in mutations_list]

        # Compute mutations
        dE_arr = []
        for mutation in mutations_list:
            if not disable_wt_warning:
                aa_target = self.target_sequence[mutation.position-1]
                aa_mutation = mutation.wt_aa.one
                if aa_mutation != aa_target:
                    print(f"WARNING in {self}.eval_mutations(): with mut='{mutation}': wt-aa does not match target sequence aa='{aa_target}'.")
            dE = E_matrix[mutation.position-1, mutation.wt_aa.id] - E_matrix[mutation.position-1, mutation.mt_aa.id]
            dE_arr.append(dE)

        # Modulate by RSA factor
        if use_rsa_factor:
            for i, (mutation, dE) in enumerate(zip(mutations_list, dE_arr)):
                rsa_factor = self.rsa_factor_array[mutation.position-1]
                if rsa_factor is None:
                    dE_arr[i] = None
                else:
                    dE_arr[i] = rsa_factor * dE

        return dE_arr
    
    def get_scores(self) -> List[dict]:
        """"""
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
        return scores
    
    def save_scores(self, scores_path: str, sep: str=";", log_results: bool=False, round_digit: Union[None, int]=None) -> List[dict]:
        
        # Guardians
        scores_path = os.path.abspath(scores_path)
        assert os.path.isdir(os.path.dirname(scores_path)), f""
        assert scores_path.endswith(".csv"), f""
        assert len(sep) == 1, f""

        # Compute scores
        scores = self.get_scores()

        # Round if required
        if round_digit is not None:
            for score in scores:
                for prop in ["gap_freq", "wt_freq", "mt_freq", "CI", "RSA", "LOR", "LR", "RSA*LOR", "RSA*LR"]:
                    val = score[prop]
                    if val is not None:
                        score[prop] = round(val, round_digit)

        # Format in CSV
        scores_properties = list(scores[0].keys())
        scores_csv = CSV(scores_properties, name=self.name)
        scores_csv.add_entries(scores)

        # Log
        if log_results:
            scores_csv.show(n_entries=40, max_colsize=23)

        # Save and return
        scores_csv.write(scores_path)
        return scores

    # IO Methods ---------------------------------------------------------------
    def write(self, msa_path: str) -> "MSA":
        """Save MSA to a '.fasta' MSA file."""

        # Guardians
        assert os.path.isdir(os.path.dirname(msa_path)), f"ERROR in {self}.write(): directory of msa_path='{msa_path}' does not exists."
        assert msa_path.endswith(".fasta"), f"ERROR in {self}.write(): msa_path='{msa_path}' should end with '.fasta'."

        # Write
        with open(msa_path, "w") as fs:
            fs.write("".join([seq.to_fasta_string() for seq in self.sequences]))
        return self
    
    # Dependencies -------------------------------------------------------------
    def _verify_sequence_length(self, sequence: Sequence, target_length: int, i: int) -> None:
        if len(sequence) != target_length:
            seq_str = sequence.sequence
            if len(seq_str) > 40:
                seq_str = seq_str[0:37] + "..."
            error_log = f"ERROR in {self}._read_sequences(): msa_path='{self.msa_path}':"
            error_log += f"\n -> length of sequence [{i+1}] l={len(sequence)} ('{seq_str}') does not match length of target sequence l={target_length}."
            raise ValueError(error_log)
