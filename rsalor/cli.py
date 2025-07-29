
# Imports ----------------------------------------------------------------------
import os.path
import argparse
from rsalor import MSA


# CLI: dependencies ------------------------------------------------------------

# Custom Help Formatter (for 'rsalor -h' help logs menu)
class CustomHelpFormatter(argparse.HelpFormatter):

    # Increase max_help_position
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, max_help_position=40)

    # Only show metavar once
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        parts = action.option_strings[:]
        if action.nargs != 0:
            parts[-1] += f" {self._format_args(action, action.dest)}"
        return ", ".join(parts)


# CLI: main function -----------------------------------------------------------
def main():
    

    # Define Argument Parser ---------------------------------------------------

    # Init parser
    parser = argparse.ArgumentParser(
        description="Run RSALOR for all single-site missence mutations on a target protein sequence.",
        usage=f"rsalor <msa_path> <pdb_path> <chain> [options]\nhelp:  rsalor -h",
        formatter_class=CustomHelpFormatter,
    )

    parser.add_argument(
        "msa_path", type=str,
        help="path to MSA '.fasta' or '.a2m' file",
    )

    parser.add_argument(
        "pdb_path", type=str,
        help="path to PDB '.pdb' file",
    )

    parser.add_argument(
        "chain", type=str,
        help="chain in the PDB to consider",
    )

    parser.add_argument(
        "-o", "--output_path", type=str, default=None, metavar="<str>",
        help="path to output '.csv' file (default='./[msa_filename]_rsalor.csv')",
    )

    parser.add_argument(
        "-t", "--theta_regularization", type=float, default=0.01, metavar="<float>",
        help="regularization term at amino acid frequencies level (default=0.01)",
    )

    parser.add_argument(
        "-n", "--n_regularization", type=float, default=0.00, metavar="<float>",
        help="regularization term at amino acid counts level (default=0.00)",
    )

    parser.add_argument(
        "-i", "--ignore_target_sequence", dest="ignore_target_sequence", action="store_true",
        help="ignore target (first) sequence of the MSA in frequencies",
    )
    parser.set_defaults(ignore_target_sequence=False)

    parser.add_argument(
        "-r", "--keep_redundant_sequences", dest="keep_redundant_sequences", action="store_true",
        help="skip MSA pre-procesing step to remove redundent sequences",
    )
    parser.set_defaults(keep_redundant_sequences=False)

    parser.add_argument(
        "-w", "--seqid_weights", type=float, default=0.80, metavar="<float>",
        help="seqid threshold for weighting (default=0.80, set 0.0 to ignore)",
    )

    parser.add_argument(
        "-m", "--min_seqid", type=float, default=0.35, metavar="<float>",
        help="discard sequences which seqid with target sequence is below (default=0.35)",
    )

    parser.add_argument(
        "-N", "--num_threads", type=int, default=1, metavar="<int>",
        help="number of threads (CPUs) for weights step (default=1)",
    )

    parser.add_argument(
        "-S", "--silent", dest="silent", action="store_true",
        help="run in silent mode (default=False)",
    )
    parser.set_defaults(silent=False)

    parser.add_argument(
        "-W", "--disable_warnings", dest="disable_warnings", action="store_true",
        help="disable logging of warnings (default=False)",
    )
    parser.set_defaults(disable_warnings=False)

    parser.add_argument(
        "--sep", type=str, default=";", metavar="<str>",
        help="separator in the output '.csv' file",
    )

    parser.add_argument(
        "--weights_cache_path", type=str, default=None, metavar="<str>",
        help="set to read (if file exists) or write (if files does not exists) weights",
    )

    parser.add_argument(
        "--rsa_cache_path", type=str, default=None, metavar="<str>",
        help="set to read (if file exists) or write (if files does not exists) RSA values",
    )

    args = parser.parse_args()

    # Set default output_path
    if args.output_path is None:
        output_dir = "./"
        msa_name:str = os.path.basename(args.msa_path)
        for extention in MSA.ACCEPTED_EXTENTIONS:
            if msa_name.endswith(f".{extention}"):
                msa_name = msa_name.removesuffix(f".{extention}")
                break
        args.output_path = os.path.join(output_dir, f"{msa_name}_rsalor.csv")

    # Verify output_path
    else:
        if not args.output_path.endswith(".csv"):
            raise ValueError(f"ERROR in rsalor: output_path='{args.output_path}' should end with '.csv'.")
        
    # Set None values for 'degenerated' arguments
    if float(args.seqid_weights) == 0.0 or float(args.seqid_weights) == 1.0:
        args.seqid_weights = None
    if float(args.min_seqid) == 0.0:
        args.min_seqid = None


    # Execute RSALOR -----------------------------------------------------------

    # Run RSALOR
    msa = MSA(
        args.msa_path,
        args.pdb_path,
        args.chain,
        theta_regularization=args.theta_regularization,
        n_regularization=args.n_regularization,
        count_target_sequence=not args.ignore_target_sequence,
        remove_redundant_sequences=not args.keep_redundant_sequences,
        seqid_weights=args.seqid_weights,
        min_seqid=args.min_seqid,
        num_threads=args.num_threads,
        #rsa_solver=args.rsa_solver,
        #rsa_solver_path=args.rsa_solver_path,
        #trimmed_msa_path=args.trimmed_msa_path,
        #allow_msa_overwrite=args.allow_msa_overwrite,
        weights_cache_path=args.weights_cache_path,
        rsa_cache_path=args.rsa_cache_path,
        verbose=not args.silent,
        disable_warnings=args.disable_warnings,
        #name=args.name,
    )

    # Save scores
    msa.save_scores(
        args.output_path,
        round_digit=6,
        sep=args.sep,
    )
