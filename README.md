
# RSALOR

Combine structural (Relative Solvent Accessibility) and Evolutionary (Log Odd Ratio from MSA) data to evaluate missence mutations in proteins.

**Please cite**:
Hermans Pauline, Tsishyn Matsvei, Schwersensky Martin, Rooman Marianne and Pucci Fabrizio (2024). Exploring evolution to enhance mutational stability prediction. [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.28.596203v2.abstract).

## Installation and dependencies

`rsalor` is implemented in Python and can be installed as a pip package. It uses a C++ backend for optimization.

Installation with `pip`:

```bash
pip install rsalor
```

Requirements:
- Python (version 3.9 or later)
- Python packages `numpy` ans `biopython` (version 1.75 or later)
- C++ compiler that supports C++11 such as the GNU compiler collection (GCC)
- Optionally, OpenMP for multithreading support

## Usage

Base usage:

```python
# Import package
from rsalor import MSA

# Initialize MSA
msa_path = "./test_data/6acv_A_29-94.fasta"
pdb_path = "./test_data/6acv_A_29-94.pdb"
chain = "A"
msa = MSA(msa_path, pdb_path, chain, num_threads=8, verbose=True)

# Get LOR and other scores (List[Dict[str, str | float]]) for all single mutations
scores = msa.get_scores() # scores = [{'mutation_fasta': 'S1A', 'mutation_pdb': 'SA1A', 'RSA': 61.54, 'LOR': 5.05}, ...]

# Alternatively, save scores to a CSV file
msa.save_scores("./test_data/6acv_A_29-94_scores.csv")

# If you do not need RSA values extracted from structure, just omit the 'pdb_path' input parameter
msa = MSA(msa_path, num_threads=8, verbose=True)
```


## Compile from source code

Alternatively to use `pip`, you can directly compile the code from source with:

```bash
cd rsalor/weights/                        # navigate to directory with the C++ code
mkdir build                               # create directory for build files
cd build                                  # go to created directory
cmake ..                                  # generate make files
make                                      # compile all C++ code
mv ./lib_computeWeightsBackend* ../       # move compiled file to correct direcoty
```