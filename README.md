
# RSALOR


## Installation and dependencies

biopython, numpy
python 3.9 or later
gcc compiler

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
