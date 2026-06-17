import argparse
import copy
import json
import os.path
import random
import sys
import math 
import freesasa

import numpy as np
import torch
from data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN
from prody import writePDB
from sc_utils import Packer, pack_side_chains

# ============================================================
# <<< pI PATCH 1 of 3 >>>  
# ============================================================
import torch.nn.functional as F
# Canonical AA order used by LigandMPNN (20 standard AAs + X)
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"   # indices 0-19; index 20 = X (unknown)
 
# pKa look-up for ionisable side chains from
#  https://ipc2.mimuw.edu.pl/theory.html / https://academic.oup.com/nar/article/49/W1/W285/6255695
'''
_PKA_POS = [                         # groups positively charged at low pH
    (_AA_ORDER.index("H"),  5.492),   # His
    (_AA_ORDER.index("K"), 9.247),   # Lys
    (_AA_ORDER.index("R"), 10.223),   # Arg
]
_PKA_NEG = [                         # groups negatively charged at high pH
    (_AA_ORDER.index("D"),  3.799),   # Asp
    (_AA_ORDER.index("E"),  4.497),   # Glu
    (_AA_ORDER.index("C"),  7.89),   # Cys
    (_AA_ORDER.index("Y"), 11.491),   # Tyr
]
_PKA_NTERM = 5.779
_PKA_CTERM = 6.065
'''
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
_SIGMOID_T = 1 / math.log(10)    # sharpness of ionisation sigmoid,  Henderson-Hasselbalch formula
 
# Background distribution of natural amino acids (Uniprot frequencies, order = _AA_ORDER)
_AA_FREQ_VALUES = [
    0.08, 0.02, 0.05, 0.06, 0.04,
    0.07, 0.02, 0.06, 0.07, 0.09,
    0.02, 0.04, 0.05, 0.04, 0.05,
    0.07, 0.06, 0.01, 0.03, 0.06
]
 

_BIOPYTHON_PKA = {
    # (positive groups: charge +1 below pKa)
    "K":  10.5,
    "R":  12.5,
    "H":   6.0,
    # (negative groups: charge -1 above pKa)  
    "D":   3.9,
    "E":   4.07,
    "C":   8.18,
    "Y":  10.1,
    # termini
    "NT":  8.0,
    "CT":  3.65,
}
def _biopython_charge(probs: torch.Tensor, pH: torch.Tensor) -> torch.Tensor:
    """
    Exact BioPython charge model — matches isoelectric_point() output.
    probs: [L, 20], pH: scalar
    Uses BioPython convention: cr = 10**(pKa-pH) / (1 + 10**(pKa-pH)) for positive groups
    """
    log10 = math.log(10)
    q = torch.zeros(1, device=probs.device, dtype=probs.dtype)
    
    # Positive groups (protonated = charged at low pH)
    for aa, pka in [("K", 10.5), ("R", 12.5), ("H", 6.0)]:
        idx = _AA_ORDER.index(aa)
        # BioPython: cr = 10^(pKa-pH) / (1 + 10^(pKa-pH)) = sigmoid((pKa-pH)*log10)
        cr = torch.sigmoid((pka - pH) * log10)
        q = q + (probs[:, idx] * cr).sum()
    
    # Negative groups (deprotonated = charged at high pH)
    for aa, pka in [("D", 3.9), ("E", 4.07), ("C", 8.18), ("Y", 10.1)]:
        idx = _AA_ORDER.index(aa)
        cr = torch.sigmoid((pH - pka) * log10)
        q = q - (probs[:, idx] * cr).sum()
    
    # Termini (once per chain)
    q = q + torch.sigmoid((8.0  - pH) * log10)   # N-term
    q = q - torch.sigmoid((pH  - 3.65) * log10)  # C-term
    
    return q
 
 
def _estimate_pI(probs20: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """
    Differentiable binary search for the isoelectric point.
    Uses smooth tanh-branching so gradients flow back through the bisection
    without needing create_graph or higher-order autograd.
    """
    dev, dt = probs20.device, probs20.dtype
    lo = torch.tensor(0.0,  device=dev, dtype=dt)
    hi = torch.tensor(14.0, device=dev, dtype=dt)
    for _ in range(n_iter):
        mid  = (lo + hi) / 2.0
        q    = _biopython_charge(probs20, mid)
        # tanh(q*1e3) ≈ sign(q): +1 when q>0 (pI above mid), -1 when q<0
        sign = torch.tanh(q * 1e3)
        lo   = lo  + (mid - lo) * ( sign + 1) / 2   # lo → mid when q > 0
        hi   = hi  - (hi - mid) * (-sign + 1) / 2   # hi → mid when q < 0
    return (lo + hi) / 2.0

def _charge_from_counts(counts, pH):
    log10 = math.log(10)
    q = torch.zeros(1, device=counts.device, dtype=counts.dtype)
    for aa, pka in [("K",10.5),("R",12.5),("H",6.0)]:
        idx = _AA_ORDER.index(aa)
        q = q + counts[idx] * torch.sigmoid((pka - pH) * log10)
    for aa, pka in [("D",3.9),("E",4.07),("C",8.18),("Y",10.1)]:
        idx = _AA_ORDER.index(aa)
        q = q - counts[idx] * torch.sigmoid((pH - pka) * log10)
    q = q + torch.sigmoid((8.0  - pH) * log10)
    q = q - torch.sigmoid((pH - 3.65) * log10)
    return q


def compute_pI_logit_bias(
    feature_dict: dict,
    surface_weights: torch.Tensor,
    target_pI:    float,
    weight:       float = 1.0,
    n_steps:      int   = 400,
    lr:           float = 0.2,
    device:       str   = "cpu",
    base_logits_full=None,
) -> torch.Tensor:
    """
    Optimise a per-residue AA logit bias so that the expected AA composition
    (measured via _charge_from_counts, identical to BioPython) hits target_pI.

    Key design decisions
    --------------------
    1. Loss is on expected INTEGER counts (probs.sum(0)), not soft per-residue
       distributions — this is what BioPython actually measures.
    2. base_logits reflects the MODEL's actual preferences (passed in from
       feature_dict["bias"] existing values), so the bias is computed relative
       to what the model would sample without intervention.
    3. No bias_l2 penalty — it was suppressing the bias below the level needed
       to compete with the model's strong prior for this protein.
    4. KL regularisation is kept very weak (0.001) and uses the model prior,
       not the uniform prior, so it only penalises biologically unreasonable
       deviations.
    5. His is capped at 8% (was 5%, too tight — His is a valid surface residue).
    """
    existing_bias = feature_dict.get("bias", None)
    if existing_bias is not None:
        B, L, _ = existing_bias.shape
    else:
        L = int(feature_dict["mask"].shape[1])

    w = surface_weights.to(device)          # [L] rSASA weights in [0,1]
    w_loss = 0.2 + 0.8 * w                  # floor at 0.2 so buried residues contribute
    w_sum  = w_loss.sum().clamp(min=1e-6)

    # base_logits: use the model's existing bias as the reference point so the
    # optimiser learns the *delta* needed relative to the model's preferences.
    # If the existing bias encodes strong K/R preference (native pI 8.93),
    # the optimiser must work against it explicitly.
    if base_logits_full is not None:
        base_logits = base_logits_full.clone().detach().to(device)  # [L, 20]
    elif existing_bias is not None:
        # Use existing bias[:, :, :20] as the reference logit frame
        base_logits = existing_bias[0, :, :20].clone().detach().to(device)  # [L, 20]
    else:
        base_logits = torch.zeros(L, 20, device=device)

    bias_all = torch.zeros(L, 20, device=device, requires_grad=True)
    opt = torch.optim.Adam([bias_all], lr=lr)

    pH_target = torch.tensor(target_pI, device=device, dtype=torch.float32)

    for step in range(n_steps):
        opt.zero_grad()

        # Probabilities in the biased model's reference frame
        probs      = F.softmax(base_logits + bias_all, dim=-1)          # [L, 20]
        #prior_probs = F.softmax(base_logits, dim=-1).detach()            # [L, 20]
        p_nat = torch.tensor(_AA_FREQ_VALUES, device=device, dtype=torch.float32)
        p_nat = p_nat / p_nat.sum()                              # normalise to sum=1
        #prior_probs = p_nat.unsqueeze(0).expand(L, -1)   
        prior_probs = F.softmax(base_logits, dim=-1).detach() 

        # ── 1. pI loss on expected counts (= BioPython-equivalent) ──────────
        # rSASA-weighted expected counts: surface residues contribute more
        weighted_counts = (w_loss.unsqueeze(-1) * probs).sum(0)          # [20]
        # Normalise to true expected counts (scale back to L residues)
        expected_counts = weighted_counts * (L / w_sum)                  # [20]

        # Bisection on counts — same maths as BioPython isoelectric_point()
        lo_c = torch.tensor(0.0,  device=device, dtype=torch.float32)
        hi_c = torch.tensor(14.0, device=device, dtype=torch.float32)
        for _ in range(30):
            mid_c = (lo_c + hi_c) / 2
            q_c   = _charge_from_counts(expected_counts, mid_c)
            s_c   = torch.tanh(q_c * 1e3)
            lo_c  = lo_c + (mid_c - lo_c) * (s_c + 1) / 2
            hi_c  = hi_c - (hi_c - mid_c) * (-s_c + 1) / 2
        pI_est_train = (lo_c + hi_c) / 2
        pI_loss = (pI_est_train - pH_target) ** 2

        # ── 2. KL toward model prior (very weak — just prevents wild sequences) 
        kl_loss = (w_loss * torch.sum(
            probs * torch.log((probs + 1e-8) / (prior_probs + 1e-8)), dim=-1
        )).sum() / w_sum

        # ── 3. Ionisable floor: keep at least 25% DEKRH on surface positions ─
        ionisable_idx = [_AA_ORDER.index(aa) for aa in "DEKRH"]
        ionisable_frac = (w_loss * probs[:, ionisable_idx].sum(-1)).sum() / w_sum
        ionisable_penalty = F.relu(0.25 - ionisable_frac) ** 2


        # ── 4. His cap at 3%  / Cys cap at 2% - natural distribution — prevents His from being used as a free buffer ─
        his_frac    = (w_loss * probs[:, _AA_ORDER.index("H")]).sum() / w_sum
        his_penalty = F.relu(his_frac - 0.03) ** 2
        cys_frac = (w_loss * probs[:, _AA_ORDER.index("C")]).sum() / w_sum
        cys_penalty = F.relu(cys_frac - 0.02) ** 2   # cap Cys at 2% (natural ~2%)

        loss = pI_loss + 0.05 * kl_loss + 20.0 * ionisable_penalty + 50.0 * his_penalty + 50.00 * cys_penalty
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"  step {step:3d}: pI_counts={pI_est_train.item():.4f}  "
                  f"loss={loss.item():.6f}  bias_max={bias_all.abs().max().item():.4f}")

    # ── Reporting ────────────────────────────────────────────────────────────
    with torch.no_grad():
        probs_final     = F.softmax(base_logits + bias_all, dim=-1)
        weighted_counts = (w_loss.unsqueeze(-1) * probs_final).sum(0)
        expected_counts = weighted_counts * (L / w_sum)

        # BioPython cross-check
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        seq_approx = "".join(aa * max(1, int(expected_counts[i].round()))
                             for i, aa in enumerate(_AA_ORDER))
        pa = ProteinAnalysis(seq_approx)
        bp_pI = pa.isoelectric_point()

        # Our bisection on same counts
        lo_r = torch.tensor(0.0, device=device, dtype=torch.float32)
        hi_r = torch.tensor(14.0, device=device, dtype=torch.float32)
        for _ in range(30):
            mid_r = (lo_r + hi_r) / 2
            q_r   = _charge_from_counts(expected_counts, mid_r)
            s_r   = torch.tanh(q_r * 1e3)
            lo_r  = lo_r + (mid_r - lo_r) * (s_r + 1) / 2
            hi_r  = hi_r - (hi_r - mid_r) * (-s_r + 1) / 2
        pI_report = ((lo_r + hi_r) / 2).item()

        print("\n[DIAG] Expected AA counts (rSASA-weighted, scaled to L):")
        for i, aa in enumerate(_AA_ORDER):
            if expected_counts[i] > expected_counts.mean() * 1.1:
                tag = " ▲"
            elif expected_counts[i] < expected_counts.mean() * 0.9:
                tag = " ▼"
            else:
                tag = ""
            print(f"  {aa}: {expected_counts[i]:.1f}{tag}")
        print(f"[DIAG] BioPython pI of expected composition : {bp_pI:.4f}")
        print(f"[DIAG] Our count-based pI estimate          : {pI_report:.4f}")
        print(f"[DIAG] Target                               : {target_pI:.4f}")
        print(f"[pI bias] bias_max = {bias_all.abs().max().item():.4f} logit units  "
              f"(effective at sampling = {bias_all.abs().max().item() * weight:.4f})")

    full_bias = torch.zeros(1, L, 21, device=device)
    full_bias[0, :, :20] = bias_all.detach() * weight
    return full_bias
 
 
# max SASA values (Tien et al. 2013)
MAX_SASA = {
    'A': 121, 'C': 148, 'D': 187, 'E': 214, 'F': 228,
    'G': 97,  'H': 216, 'I': 195, 'K': 230, 'L': 191,
    'M': 203, 'N': 187, 'P': 154, 'Q': 214, 'R': 265,
    'S': 143, 'T': 163, 'V': 165, 'W': 264, 'Y': 255
}

def compute_rsasa_weights(pdb_file, device="cpu"):
    structure = freesasa.Structure(pdb_file)
    result = freesasa.calc(structure)

    areas = result.residueAreas()

    rsasa = []
    for chain in areas:
        for resnum in areas[chain]:
            res = areas[chain][resnum]

            aa = res.residueType.strip()
            if aa not in MAX_SASA:
                rsasa.append(0.0)
                continue

            sasa = res.total
            rsasa_val = sasa / MAX_SASA[aa]

            rsasa.append(min(rsasa_val, 1.0))

    # convert to tensor before sigmoid (np.ndarray is not accepted by torch.sigmoid)
    rsasa = torch.tensor(np.array(rsasa), dtype=torch.float32, device=device)

    # smooth weighting: below 25% exposure → mostly ignored (0.1 weight), above → increasingly important, if turned off 
    # it would not properply calculate the pI
    weights = 0.1 + 0.9 * torch.sigmoid((rsasa - 0.25) * 8)

    return weights
 
def main(args) -> None:
    """
    Inference function
    """
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    folder_for_outputs = args.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if not os.path.exists(base_folder + "seqs"):
        os.makedirs(base_folder + "seqs", exist_ok=True)
    if not os.path.exists(base_folder + "backbones"):
        os.makedirs(base_folder + "backbones", exist_ok=True)
    if not os.path.exists(base_folder + "packed"):
        os.makedirs(base_folder + "packed", exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + "stats"):
            os.makedirs(base_folder + "stats", exist_ok=True)
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.pack_side_chains:
        model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=device,
            num_mix=3,
        )

        checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device)
        model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        model_sc.to(device)
        model_sc.eval()

    if args.pdb_path_multi:
        with open(args.pdb_path_multi, "r") as fh:
            pdb_paths = list(json.load(fh))
    else:
        pdb_paths = [args.pdb_path]

    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, "r") as fh:
            fixed_residues_multi = json.load(fh)
            fixed_residues_multi = {key:value.split() for key,value in fixed_residues_multi.items()}
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdb_paths:
            fixed_residues_multi[pdb] = fixed_residues

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, "r") as fh:
            redesigned_residues_multi = json.load(fh)
            redesigned_residues_multi = {key:value.split() for key,value in redesigned_residues_multi.items()}
    else:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_residues_multi = {}
        for pdb in pdb_paths:
            redesigned_residues_multi[pdb] = redesigned_residues

    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_str_to_int[AA]] = a2[i]

    if args.bias_AA_per_residue_multi:
        with open(args.bias_AA_per_residue_multi, "r") as fh:
            bias_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue, "r") as fh:
                bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
            bias_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    if args.omit_AA_per_residue_multi:
        with open(args.omit_AA_per_residue_multi, "r") as fh:
            omit_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
    else:
        if args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue, "r") as fh:
                omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
            omit_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=device,
    )

    if len(args.parse_these_chains_only) != 0:
        parse_these_chains_only_list = args.parse_these_chains_only.split(",")
    else:
        parse_these_chains_only_list = []

    # <<< pI PATCH 2 of 3 >>>
    # Read the new pI flags.
    # ------------------------------------------------------------------ #
    target_pI      = args.target_pI        # float or None
    pI_weight      = args.pI_weight
    sasa_threshold = args.sasa_threshold
    # ------------------------------------------------------------------ #

    # loop over PDB paths
    for pdb in pdb_paths:
        if args.verbose:
            print("Designing protein from this path:", pdb)
        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]
        parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
            args.pack_side_chains and not args.repack_everything
        )
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=parse_these_chains_only_list,
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
            bias_dict = bias_AA_per_residue_multi[pdb]
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
            omit_dict = omit_AA_per_residue_multi[pdb]
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                args.global_transmembrane_label + 0 * fixed_positions
            )
        if len(args.chains_to_design) != 0:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]

        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        # specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in args.symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        # specify linking weights
        if args.symmetry_weights:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in args.symmetry_weights.split("|")
            ]
        else:
            symmetry_weights = [[]]

        if args.homo_oligomer:
            if args.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:] for item in encoded_residues if item[:lc] == reference_chain
            ]
            remapped_symmetry_residues = []
            symmetry_weights = []
            for res in residue_indices:
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1 / len(chain_letters_set))
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if args.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif args.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    for i, atom_type in enumerate(atom_types):
                        print(
                            f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                        )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []

            # ------------------------------------------------------------------ #
            # <<< pI PATCH 3 of 3 >>>
            # Compute pI-steering bias and inject into feature_dict["bias"]
            # ------------------------------------------------------------------ #
            if target_pI is not None:
                device_str = str(next(model.parameters()).device)

                # bug fix: use `pdb` (current loop variable), not args.pdb_path
                surface_weights = compute_rsasa_weights(pdb, device=device_str)

                if len(surface_weights) != L:
                    print(f"WARNING: SASA computation returned {len(surface_weights)} "
                          f"weights but expected {L} residues. Falling back to uniform weights.")
                    surface_weights = torch.ones(L, dtype=torch.float32, device=device_str)

                base_logits_full = None

                with torch.enable_grad():
                    # Step 1: one unbiased forward pass to get the model's actual
                    # per-position sampling distribution as base logits.
                    with torch.no_grad():
                        tmp_feature_dict = {k: v.clone() if torch.is_tensor(v) else v
                                            for k, v in feature_dict.items()}
                        tmp_feature_dict["bias"] = torch.zeros_like(feature_dict["bias"])
                        tmp_feature_dict["temperature"] = 1.0
                        # randn is required by model.sample() for the decoding order
                        tmp_feature_dict["randn"] = torch.randn(
                            [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                            device=device,
                        )
                        tmp_out = model.sample(tmp_feature_dict)
                        # sampling_probs is [B, L, 21] — full per-position distribution.
                        # log_probs is [B, L] scalar (log-prob of sampled token only) — wrong.
                        base_logits_full = torch.log(
                            tmp_out["sampling_probs"][0, :, :20].clamp(min=1e-8)
                        )  # [L, 20]

                    # Step 2: optimise the pI bias relative to these base logits.
                    pi_bias = compute_pI_logit_bias(
                        feature_dict     = feature_dict,
                        surface_weights  = surface_weights,
                        target_pI        = target_pI,
                        weight           = pI_weight,
                        base_logits_full = base_logits_full,
                        device           = device_str,
                    )

                # detach so model.sample() sees a plain tensor with no grad_fn
                feature_dict["bias"] = (feature_dict["bias"] + pi_bias).detach()
            # ------------------------------------------------------------------ #

            # Sampling loop — inside no_grad like the rest of inference
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                output_dict = model.sample(feature_dict)

                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if args.model_type == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

            native_seq = "".join(
                [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [args.fasta_seq_separation]
            seq_out_str = "".join(seq_out_str)[:-1]

            output_fasta = base_folder + "/seqs/" + name + args.file_ending + ".fa"
            output_backbones = base_folder + "/backbones/"
            output_packed = base_folder + "/packed/"
            output_stats_path = base_folder + "stats/" + name + args.file_ending + ".pt"

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            out_dict["temperature"] = args.temperature
            if args.save_stats:
                torch.save(out_dict, output_stats_path)

            if args.pack_side_chains:
                if args.verbose:
                    print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=args.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = args.batch_size
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(args.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(args.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            model_sc,
                            args.sc_num_denoising_steps,
                            args.sc_num_samples,
                            args.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)

            with open(output_fasta, "w") as f:
                f.write(
                    ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, ligand_cutoff_distance={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                        name,
                        args.temperature,
                        seed,
                        torch.sum(rec_mask).cpu().numpy(),
                        torch.sum(combined_mask[:1]).cpu().numpy(),
                        bool(args.ligand_mpnn_use_atom_context),
                        float(args.ligand_mpnn_cutoff_for_score),
                        args.batch_size,
                        args.number_of_batches,
                        checkpoint_path,
                        seq_out_str,
                    )
                )
                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not args.zero_indexed:
                        ix_suffix += 1
                    seq_rec_print = np.format_float_positional(
                        rec_stack[ix].cpu().numpy(), unique=False, precision=4
                    )
                    loss_np = np.format_float_positional(
                        np.exp(-loss_stack[ix].detach().cpu().numpy()), unique=False, precision=4
                    )
                    loss_XY_np = np.format_float_positional(
                        loss_XY_stack[ix].detach().cpu().numpy(),
                        unique=False,
                        precision=4,
                    )
                    seq = "".join(
                        [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                    )

                    # write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                        None,
                    ].repeat(4, 1)
                    bfactor_prody = (
                        loss_per_residue_stack[ix].detach().cpu().numpy()[None, :].repeat(4, 1)
                    )
                    backbone.setResnames(seq_prody)
                    backbone.setBetas(
                        np.exp(-bfactor_prody)
                        * (bfactor_prody > 0.01).astype(np.float32)
                    )
                    if other_atoms:
                        writePDB(
                            output_backbones
                            + name
                            + "_"
                            + str(ix_suffix)
                            + args.file_ending
                            + ".pdb",
                            backbone + other_atoms,
                        )
                    else:
                        writePDB(
                            output_backbones
                            + name
                            + "_"
                            + str(ix_suffix)
                            + args.file_ending
                            + ".pdb",
                            backbone,
                        )

                    # write full PDB files
                    if args.pack_side_chains:
                        for c_pack in range(args.number_of_packs_per_design):
                            X_stack = X_stack_list[c_pack]
                            X_m_stack = X_m_stack_list[c_pack]
                            b_factor_stack = b_factor_stack_list[c_pack]
                            write_full_PDB(
                                output_packed
                                + name
                                + args.packed_suffix
                                + "_"
                                + str(ix_suffix)
                                + "_"
                                + str(c_pack + 1)
                                + args.file_ending
                                + ".pdb",
                                X_stack[ix].cpu().numpy(),
                                X_m_stack[ix].cpu().numpy(),
                                b_factor_stack[ix].cpu().numpy(),
                                feature_dict["R_idx_original"][0].cpu().numpy(),
                                protein_dict["chain_letters"],
                                S_stack[ix].cpu().numpy(),
                                other_atoms=other_atoms,
                                icodes=icodes,
                                force_hetatm=args.force_hetatm,
                            )
                    # -----

                    # write fasta lines
                    seq_np = np.array(list(seq))
                    seq_out_str = []
                    for mask in protein_dict["mask_c"]:
                        seq_out_str += list(seq_np[mask.cpu().numpy()])
                        seq_out_str += [args.fasta_seq_separation]
                    seq_out_str = "".join(seq_out_str)[:-1]
                    if ix == S_stack.shape[0] - 1:
                        # final 2 lines
                        f.write(
                            ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
                                name,
                                ix_suffix,
                                args.temperature,
                                seed,
                                loss_np,
                                loss_XY_np,
                                seq_rec_print,
                                seq_out_str,
                            )
                        )
                    else:
                        f.write(
                            ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
                                name,
                                ix_suffix,
                                args.temperature,
                                seed,
                                loss_np,
                                loss_XY_np,
                                seq_rec_print,
                                seq_out_str,
                            )
                        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",
    )
    # protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    # ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    # per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    # global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    # soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    argparser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="/home/hole11/software/LigandMPNN_pI/model_params/proteinmpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="/home/hole11/software/LigandMPNN_pI/model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="/home/hole11/software/LigandMPNN_pI/model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="/home/hole11/software/LigandMPNN_pI/model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="/home/hole11/software/LigandMPNN_pI/model_params/solublempnn_v_48_020.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--fasta_seq_separation",
        type=str,
        default=":",
        help="Symbol to use between sequences from different chains",
    )
    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument(
        "--pdb_path", type=str, default="", help="Path to the input PDB."
    )
    argparser.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    argparser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Provide fixed residues, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--bias_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'",
    )
    argparser.add_argument(
        "--bias_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}",
    )
    argparser.add_argument(
        "--bias_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}",
    )

    argparser.add_argument(
        "--omit_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'ACG'",
    )
    argparser.add_argument(
        "--omit_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': 'APQ', 'A13': 'QST'}",
    )
    argparser.add_argument(
        "--omit_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': 'QSPC', 'A13': 'AGE'}}",
    )

    argparser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    argparser.add_argument(
        "--symmetry_weights",
        type=str,
        default="",
        help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'",
    )
    argparser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    argparser.add_argument(
        "--out_folder",
        type=str,
        help="Path to a folder to output sequences, e.g. /home/out/",
    )
    argparser.add_argument(
        "--file_ending", type=str, default="", help="adding_string_to_the_end"
    )
    argparser.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="1 - to start output PDB numbering with 0",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for torch, numpy, and python random.",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequence to generate per one pass.",
    )
    argparser.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="Number of times to design sequence using a chosen batch size.",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature to sample sequences.",
    )
    argparser.add_argument(
        "--save_stats", type=int, default=0, help="Save output statistics"
    )

    argparser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use atom context, 0 - do not use atom context.",
    )
    argparser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )
    argparser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",
    )
    argparser.add_argument(
        "--chains_to_design",
        type=str,
        default="",
        help="Specify which chains to redesign, all others will be kept fixed, 'A,B,C,F'",
    )

    argparser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Provide chains letters for parsing backbones, 'A,B,C,F'",
    )

    argparser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    argparser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    argparser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    argparser.add_argument(
        "--pack_side_chains",
        type=int,
        default=0,
        help="1 - to run side chain packer, 0 - do not run it",
    )

    argparser.add_argument(
        "--checkpoint_path_sc",
        type=str,
        default="/home/hole11/software/LigandMPNN_pI/model_params/ligandmpnn_sc_v_32_002_16.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--number_of_packs_per_design",
        type=int,
        default=4,
        help="Number of independent side chain packing samples to return per design",
    )

    argparser.add_argument(
        "--sc_num_denoising_steps",
        type=int,
        default=3,
        help="Number of denoising/recycling steps to make for side chain packing",
    )

    argparser.add_argument(
        "--sc_num_samples",
        type=int,
        default=16,
        help="Number of samples to draw from a mixture distribution and then take a sample with the highest likelihood.",
    )

    argparser.add_argument(
        "--repack_everything",
        type=int,
        default=0,
        help="1 - repacks side chains of all residues including the fixed ones; 0 - keeps the side chains fixed for fixed residues",
    )

    argparser.add_argument(
        "--force_hetatm",
        type=int,
        default=0,
        help="To force ligand atoms to be written as HETATM to PDB file after packing.",
    )

    argparser.add_argument(
        "--packed_suffix",
        type=str,
        default="_packed",
        help="Suffix for packed PDB paths",
    )

    argparser.add_argument(
        "--pack_with_ligand_context",
        type=int,
        default=1,
        help="1-pack side chains using ligand context, 0 - do not use it.",
    )
    # ------------------------------------------------------------------ #
    # <<< pI PATCH >>> #
    # ------------------------------------------------------------------ #
    argparser.add_argument(
        "--target_pI",
        type=float,
        default=None,
        help=(
            "Target isoelectric point for the designed sequence. "
            "When set, a per-residue AA bias is computed before sampling "
            "to steer the expected pI of surface residues toward this value. "
            "Example: --target_pI 6.0"
        ),
    )
    argparser.add_argument(
        "--pI_weight",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the pI steering bias (logit units). "
            "Values 0.5–3.0 are practical; higher = stronger steering "
            "but less sequence diversity. Default: 1.0"
        ),
    )
    argparser.add_argument(
        "--sasa_threshold",
        type=float,
        default=0.25,
        help=(
            "Relative SASA threshold above which a residue is treated as "
            "surface-exposed for pI steering (0–1). Default: 0.25"
        ),
    )
    # ------------------------------------------------------------------ #
#    return argparser

    args = argparser.parse_args()
    main(args)
