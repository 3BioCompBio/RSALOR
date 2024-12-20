
# RSALOR

[![PyPi Version](https://img.shields.io/pypi/v/rsalor.svg)](https://pypi.org/project/rsalor/)

`rsalor` is a Python package that computes the `RSA*LOR` score for each missence mutation in a protein. It combines multiple computational steps into a fast and user-friendly tool.

**Please cite**:
Hermans Pauline, Tsishyn Matsvei, Schwersensky Martin, Rooman Marianne and Pucci Fabrizio (2024). Exploring evolution to enhance mutational stability prediction. [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.28.596203v2.abstract).

## Installation and Usage

Installation with `pip`:
```bash
pip install rsalor
```

Make sure the first sequence in your MSA file is the target sequence to mutate.  
```python
# Import
from rsalor import MSA

# Initialize MSA
msa_path = "./test_data/6acv_A_29-94.fasta"
pdb_path = "./test_data/6acv_A_29-94.pdb"
chain = "A"
msa = MSA(msa_path, pdb_path, chain, num_threads=8, verbose=True)

# You can ignore structure and RSA by omitting the pdb_path argument
#msa = MSA(msa_path, num_threads=8, verbose=True)

# Get LOR and other scores for all mutations
scores = msa.get_scores() # [{'mutation_fasta': 'S1A', 'mutation_pdb': 'SA1A', 'RSA': 61.54, 'LOR': 5.05, ...}, ...]

# Or directly save scores to a CSV file
msa.save_scores("./test_data/6acv_A_29-94_scores.csv", sep=";")
```

## Requirements

- Python 3.9 or later
- Python packages `numpy` ans `biopython` (version 1.75 or later)
- A C++ compiler that supports C++11 (such as GCC)
- Optionally, OpenMP for multithreading support

An example of a working `conda` environment is provided in `./conda-env.yml`.

## Short description

The `rsalor` package combines structural data (Relative Solvent Accessibility, RSA) and evolutionary data (Log Odd Ratio, LOR from MSA) to evaluate missense mutations in proteins.

It parses a Multiple Sequence Alignment (MSA), removes redundant sequences, and assigns a weight to each sequence based on sequence identity clustering. The package then computes the weighted Log Odd Ratio (LOR) and Log Ratio (LR) for each single missense mutation. Additionally, it calculates the Relative Solvent Accessibility (RSA) for each residue and combines the LOR/LR and RSA scores, as described in the reference paper. The package resolves discrepancies between the MSA's target sequence and the protein structure (e.g., missing residues in structure) by aligning the PDB structure with the MSA target sequence.

## Additional arguments

- **`msa_path`** (`str`): Path to MSA `.fasta` file
- **`pdb_path`** (`Union[None, str]=None`, optional): Path to the PDB `.pdb` file. Leave empty to ignore structure and RSA calculation.
- **`chain`** (`Union[None, str]=None`, optional): Chain of the PDB to consider.
- **`rsa_solver`** (`'biopython'/'DSSP'/'MuSiC'`, default `biopython`):  Solver used to compute RSA. DSSP or MuSiC requires the corresponding software to be installed.
- **`rsa_solver_path`** (`Union[None, str]=None`, optional): Path to the DSSP/MuSiC executable. Leave empty if the software is in the system `PATH`.
- **`rsa_cache_path`** (`Union[None, str]=None`, optional):  Path to read/write the RSA values. If empty, no file will be generated.
- **`theta_regularization`** (`float=0.1`): Regularization term for LOR/LR at the frequency level.
- **`n_regularization`** (`float=0.0`): Regularization term for LOR/LR at the counts level.
- **`count_target_sequence`** (`bool=True`): Whether to include the target (first) sequence of the MSA in the frequencies.
- **`remove_redundant_sequences`** (`bool=True`): Whether to remove redundant sequences from the MSA. This speeds up the process for deep MSAs.
- **`use_weights`** (`bool=True`): Whether to compute sequence weights. Set to `False` to set all weights to 1, which may result in faster (for ver deep MSAs) but less relevant scores.
- **`seqid`** (`float=0.80`): The sequence identity threshold to consider two sequences as similar for weight evaluation.
- **`num_threads`** (`int=1`): The number of threads (CPUs) to use for weights evaluation in the C++ backend.
- **`weights_cache_path`** (`Union[None, str]=None`, optional): Path to read/write weights for each sequence in the MSA. If empty, no file will be generated.
- **`trimmed_msa_path`** (`Union[None, str]=None`, optional): Path to save the trimmed (removed positions that are gaps in the target sequence) and non-redundent sequences MSA file. Leave empty to ignore.
- **`allow_msa_overwrite`** (`bool=False`): Whether to allow overwriting the original MSA file with the trimmed and non-redundant MSA file `trimmed_msa_path`.
- **`verbose`** (`bool=False`): Log execution steps.
- **`disable_warnings`** (`bool=False`) Disable logging for warnings.
- **`name`** (`Union[None, str]=None`, optional): Name of the MSA object instance (for logging).

## Compile from source

For performance reasons, `rsalor` uses a C++ backend to weight sequences in the MSA. The C++ code needs to be compiled to use it directly from source. To compile the code, follow these steps:
```bash
git clone https://github.com/3BioCompBio/RSALOR # Clone the repository
cd RSALOR/rsalor/weights/            # Navigate to the C++ code directory
mkdir build                          # Create a build directory
cd build                             # Enter the build directory
cmake ..                             # Generate make files
make                                 # Compile the C++ code
mv ./lib_computeWeightsBackend* ../  # Move the compiled file to the correct directory
```