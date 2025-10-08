
# RSALOR

[![PyPi Version](https://img.shields.io/pypi/v/rsalor.svg)](https://pypi.org/project/rsalor/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<div style="text-align: center;">
<img src="Logo.png" alt="[RSALOR Logo]" height="300"/>
</div>

The `rsalor` package combines structural data (Relative Solvent Accessibility, RSA) and evolutionary data (Log Odd Ratio, LOR from MSA) to evaluate effects of missense mutations in proteins.
It computes the `RSA*LOR` score for each single-site missense mutation in a target protein by combining multiple computational steps into a fast and user-friendly tool.

**Please cite**:
- [Matsvei Tsishyn, Pauline Hermans, Fabrizio Pucci, Marianne Rooman (2025). Residue conservation and solvent accessibility are (almost) all you need for predicting mutational effects in proteins. Bioinformatics, btaf322](https://doi.org/10.1093/bioinformatics/btaf322).

- [Pauline Hermans, Matsvei Tsishyn, Martin Schwersensky, Marianne Rooman, Fabrizio Pucci (2024). Exploring evolution to uncover insights into protein mutational stability. Molecular Biology and Evolution, 42(1), msae267](https://doi.org/10.1093/molbev/msae267).


## Installation and Usage

Installation with `pip`:
```bash
pip install rsalor
```

Make sure the first sequence in your MSA file is the target sequence to mutate.  
From directory `./test_data/` execute the following Python code:
```python
# Import
from rsalor import MSA

# Log basic usage instructions and arguments of the package
MSA.help()

# Initialize MSA
msa_path = "./6acv_A_29-94.fasta"
pdb_path = "./6acv_A_29-94.pdb"
chain = "A"
msa = MSA(msa_path, pdb_path, chain, num_threads=8, verbose=True)

# You can ignore structure and RSA by omitting the pdb_path argument
#msa = MSA(msa_path, num_threads=8, verbose=True)

# Get RSA*LOR and other scores for all mutations
scores = msa.get_scores() # [{'mutation_fasta': 'S1A', 'mutation_pdb': 'SA1A', 'RSA': 61.54, 'LOR': 5.05, ...}, ...]

# Or directly save scores to a CSV file
msa.save_scores("./6acv_A_29-94_scores.csv", sep=";")
```

Alternatively, you can run the `rsalor` package with a Command Line Interface (CLI).
To compute scores for all single-site missense mutations on an example target sequence, from the directory `./test_data/`, run:
```bash
rsalor ./6acv_A_29-94.fasta ./6acv_A_29-94.pdb A -o ./6acv_A_29-rsalor.csv
```

To show CLI usage and optional arguments, run:
```bash
rsalor -h
```

## Requirements

- Python 3.9 or later
- Python packages `numpy` and `biopython` (version 1.75 or later)
- A C++ compiler that supports C++11 (such as GCC)

## Short description

The `rsalor` package combines structural data (Relative Solvent Accessibility, RSA) and evolutionary data (Log Odd Ratio, LOR from MSA) to evaluate effects of missense mutations in proteins.

It parses a Multiple Sequence Alignment (MSA), removes redundant sequences, and assigns a weight to each sequence based on sequence identity clustering. The package then computes the weighted Log Odd Ratio (LOR) and Log Ratio (LR) for each single missense mutation. Additionally, it calculates the Relative Solvent Accessibility (RSA) for each residue and combines the LOR/LR and RSA scores, as described in the reference paper. The package resolves discrepancies between the MSA's target sequence and the protein structure (e.g., missing residues in structure) by aligning the PDB structure with the MSA target sequence.

The sign of RSALOR / LOR is defined such that the result of mutations from a highly represented amino acid to a less represented amino acid is positive, which generally corresponds to a decrease in protein stability or fitness. In other words, large positive values predict highly destabilizing / disruptive mutations, while values close to zero or negative predict positive or neutral mutations.

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
