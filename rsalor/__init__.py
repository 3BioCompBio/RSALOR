"""
RSALOR
======

Combines structural data (Relative Solvent Accessibility, RSA) and
evolutionary data (Log Odds Ratio, LOR from MSA) to evaluate missense
mutations in proteins.

Example of usage in Python:

>>> from rsalor import MSA
>>> msa = MSA('./msa1.fasta', './pdb1.pdb', chain='A')
>>> scores = msa.get_scores()
>>> msa.save_scores('./msa1_rsalor-scores.csv')

Example of command line usage::

    $ rsalor ./msa1.fasta ./pdb1.pdb A -o ./msa1_rsalor-scores.csv
"""

from rsalor.msa import MSA
