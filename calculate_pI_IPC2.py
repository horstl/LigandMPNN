
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import math

# Amino acid order for indexing
_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# Your provided pKa values
_PKA_POS = [
    (_AA_ORDER.index("H"),  6.0),
    (_AA_ORDER.index("K"), 10.5),
    (_AA_ORDER.index("R"), 12.5),
]

_PKA_NEG = [
    (_AA_ORDER.index("D"),  3.9),
    (_AA_ORDER.index("E"),  4.07),
    (_AA_ORDER.index("C"),  8.18),
    (_AA_ORDER.index("Y"), 10.1),
]

_PKA_NTERM = 8.0
_PKA_CTERM = 3.65


def count_amino_acids(seq):
    counts = [0] * len(_AA_ORDER)
    for aa in seq:
        if aa in _AA_ORDER:
            counts[_AA_ORDER.index(aa)] += 1
    return counts


def net_charge(seq, pH):
    counts = count_amino_acids(seq)

    charge = 0.0

    # Positive groups (protonated when charged)
    for idx, pKa in _PKA_POS:
        n = counts[idx]
        charge += n * (1.0 / (1.0 + 10**(pH - pKa)))

    # N-terminus
    charge += 1.0 / (1.0 + 10**(pH - _PKA_NTERM))

    # Negative groups (deprotonated when charged)
    for idx, pKa in _PKA_NEG:
        n = counts[idx]
        charge -= n * (1.0 / (1.0 + 10**(pKa - pH)))

    # C-terminus
    charge -= 1.0 / (1.0 + 10**(_PKA_CTERM - pH))

    return charge


def calculate_pI_ipc2(seq, tol=1e-4, max_iter=100):
    low, high = 0.0, 14.0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        charge = net_charge(seq, mid)

        if abs(charge) < tol:
            return mid

        if charge > 0:
            low = mid
        else:
            high = mid

    return mid  # fallback


def calculate_pI_from_fasta(fasta_file):
    results = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)

        # Biopython pI
        biopy_pi = ProteinAnalysis(seq).isoelectric_point()

        # IPC2-style pI
        ipc2_pi = calculate_pI_ipc2(seq)

        results.append((record.id, biopy_pi, ipc2_pi))

    return results


if __name__ == "__main__":
    fasta_path = "outputs/pi_steered/seqs/HuBBE20FAD__0004.fa"

    results = calculate_pI_from_fasta(fasta_path)

    print("ID\tBiopython_pI\tIPC2_pI")
    for seq_id, pi_bio, pi_ipc in results:
        print(f"{seq_id}\t{pi_bio:.2f}\t{pi_ipc:.2f}")
