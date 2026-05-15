from collections import namedtuple
from moleculekit import __share_dir
import json
import os

_sel = os.path.join(__share_dir, "atomselect", "atomselect.json")
with open(_sel, "r") as f:
    _sel = json.load(f)
WATER_RESIDUE_NAMES = set(_sel["water_resnames"])

_Residue = namedtuple(
    "Residue", ["full_name", "resname", "single_letter", "resname_variants"]
)


PROTEIN_RESIDUES = (
    _Residue(
        full_name="Alanine", resname="ALA", single_letter="A", resname_variants=[]
    ),
    _Residue(
        full_name="Arginine", resname="ARG", single_letter="R", resname_variants=["AR0"]
    ),
    _Residue(
        full_name="Asparagine", resname="ASN", single_letter="N", resname_variants=[]
    ),
    _Residue(
        full_name="Aspartate",
        resname="ASP",
        single_letter="D",
        resname_variants=["ASH"],
    ),
    _Residue(
        full_name="Cysteine",
        resname="CYS",
        single_letter="C",
        resname_variants=["CYM", "CYX"],
    ),
    _Residue(
        full_name="Glutamine", resname="GLN", single_letter="Q", resname_variants=[]
    ),
    _Residue(
        full_name="Glutamate",
        resname="GLU",
        single_letter="E",
        resname_variants=["GLH"],
    ),
    _Residue(
        full_name="Glycine", resname="GLY", single_letter="G", resname_variants=[]
    ),
    _Residue(
        full_name="Histidine",
        resname="HIS",
        single_letter="H",
        resname_variants=["HID", "HIE", "HIP", "HSD", "HSE", "HSP"],
    ),
    _Residue(
        full_name="Isoleucine", resname="ILE", single_letter="I", resname_variants=[]
    ),
    _Residue(
        full_name="Leucine", resname="LEU", single_letter="L", resname_variants=[]
    ),
    _Residue(
        full_name="Lysine",
        resname="LYS",
        single_letter="K",
        resname_variants=["LSN", "LYN"],
    ),
    _Residue(
        full_name="Methionine", resname="MET", single_letter="M", resname_variants=[]
    ),
    _Residue(
        full_name="Phenylalanine", resname="PHE", single_letter="F", resname_variants=[]
    ),
    _Residue(
        full_name="Proline", resname="PRO", single_letter="P", resname_variants=[]
    ),
    _Residue(full_name="Serine", resname="SER", single_letter="S", resname_variants=[]),
    _Residue(
        full_name="Threonine", resname="THR", single_letter="T", resname_variants=[]
    ),
    _Residue(
        full_name="Tryptophan", resname="TRP", single_letter="W", resname_variants=[]
    ),
    _Residue(
        full_name="Tyrosine", resname="TYR", single_letter="Y", resname_variants=[]
    ),
    _Residue(full_name="Valine", resname="VAL", single_letter="V", resname_variants=[]),
    _Residue(
        full_name="Selenocysteine",
        resname="SEC",
        single_letter="U",
        resname_variants=[],
    ),
    _Residue(
        full_name="Pyrrolysine", resname="PYL", single_letter="O", resname_variants=[]
    ),
)
MODIFIED_PROTEIN_RESIDUES = (
    _Residue(
        full_name="Selenomethionine",
        resname="MSE",
        single_letter="M",
        resname_variants=[],
    ),
    _Residue(
        full_name="N-methyl-lysine",
        resname="MLZ",
        single_letter="K",
        resname_variants=[],
    ),
    _Residue(
        full_name="N-dimethyl-lysine",
        resname="MLY",
        single_letter="K",
        resname_variants=[],
    ),
    _Residue(
        full_name="N-trimethyl-lysine",
        resname="M3L",
        single_letter="K",
        resname_variants=[],
    ),
    _Residue(
        full_name="Phosphoserine",
        resname="SEP",
        single_letter="S",
        resname_variants=[],
    ),
    _Residue(
        full_name="Phosphothreonine",
        resname="TPO",
        single_letter="T",
        resname_variants=[],
    ),
    _Residue(
        full_name="Phosphotyrosine",
        resname="PTR",
        single_letter="Y",
        resname_variants=[],
    ),
)
NUCLEIC_RESIDUES = (
    _Residue(
        full_name="Guanine",
        resname="G",
        single_letter="G",
        resname_variants=["G5", "G3", "DG", "DG5", "DG3"],
    ),
    _Residue(
        full_name="Cytosine",
        resname="C",
        single_letter="C",
        resname_variants=["C5", "C3", "DC", "DC5", "DC3"],
    ),
    _Residue(
        full_name="Uracil",
        resname="U",
        single_letter="U",
        resname_variants=["U5", "U3"],
    ),
    _Residue(
        full_name="Adenine",
        resname="A",
        single_letter="A",
        resname_variants=["A5", "A3", "DA", "DA5", "DA3"],
    ),
    _Residue(
        full_name="Thymine",
        resname="T",
        single_letter="T",
        resname_variants=["T5", "T3", "DT", "DT5", "DT3"],
    ),
)

PROTEIN_RESIDUE_NAMES = set(rr.resname for rr in PROTEIN_RESIDUES)
NUCLEIC_RESIDUE_NAMES = set(rr.resname for rr in NUCLEIC_RESIDUES)
MODIFIED_PROTEIN_RESIDUE_NAMES = set(rr.resname for rr in MODIFIED_PROTEIN_RESIDUES)

SINGLE_LETTER_RESIDUE_NAME_TABLE = {}
ORIGINAL_RESIDUE_NAME_TABLE = {}
for rr in PROTEIN_RESIDUES + NUCLEIC_RESIDUES:
    ORIGINAL_RESIDUE_NAME_TABLE[rr.resname] = rr.resname
    SINGLE_LETTER_RESIDUE_NAME_TABLE[rr.resname] = rr.single_letter
    for variant in rr.resname_variants:
        ORIGINAL_RESIDUE_NAME_TABLE[variant] = rr.resname
        SINGLE_LETTER_RESIDUE_NAME_TABLE[variant] = rr.single_letter


SINGLE_LETTER_MODIFIED_RESIDUE_NAME_TABLE = {
    rr.resname: rr.single_letter for rr in MODIFIED_PROTEIN_RESIDUES
}


# SMILES for each residue (canonical + force-field variants).
# Canonical 20 + SEC/PYL + modified-protein + RNA/DNA bases are OpenEye
# SMILES_CANONICAL from the RCSB chemical component dictionary.
# Variants are hand-derived from the parent residue:
#   - Protein protonation variants: AMBER/CHARMM naming.
#   - Nucleic 5'/3' terminals: AMBER convention (5'-terminal = no 5'-phosphate;
#     3'-terminal = same as canonical, which already exposes a free 3'-OH).
RESIDUE_SMILES = {
    "ALA": "C[C@@H](C(=O)O)N",
    "ARG": "C(C[C@@H](C(=O)O)N)CNC(=[NH2+])N",
    "AR0": "C(C[C@@H](C(=O)O)N)CN=C(N)N",
    "ASN": "C([C@@H](C(=O)O)N)C(=O)N",
    "ASP": "C([C@@H](C(=O)O)N)C(=O)[O-]",
    "ASH": "C([C@@H](C(=O)O)N)C(=O)O",
    "CYS": "C([C@@H](C(=O)O)N)S",
    "CYM": "C([C@@H](C(=O)O)N)[S-]",
    "CYX": "C([C@@H](C(=O)O)N)[S-]",
    "GLN": "C(CC(=O)N)[C@@H](C(=O)O)N",
    "GLU": "C(CC(=O)[O-])[C@@H](C(=O)O)N",
    "GLH": "C(CC(=O)O)[C@@H](C(=O)O)N",
    "GLY": "C(C(=O)O)N",
    "HIS": "c1c(nc[nH]1)C[C@@H](C(=O)O)N",
    "HID": "c1c([nH]cn1)C[C@@H](C(=O)O)N",
    "HIE": "c1c(nc[nH]1)C[C@@H](C(=O)O)N",
    "HIP": "c1c([nH+]c[nH]1)C[C@@H](C(=O)O)N",
    "HSD": "c1c([nH]cn1)C[C@@H](C(=O)O)N",
    "HSE": "c1c(nc[nH]1)C[C@@H](C(=O)O)N",
    "HSP": "c1c([nH+]c[nH]1)C[C@@H](C(=O)O)N",
    "ILE": "CC[C@H](C)[C@@H](C(=O)O)N",
    "LEU": "CC(C)C[C@@H](C(=O)O)N",
    "LYS": "C(CC[NH3+])C[C@@H](C(=O)O)N",
    "LSN": "C(CCN)C[C@@H](C(=O)O)N",
    "LYN": "C(CCN)C[C@@H](C(=O)O)N",
    "MET": "CSCC[C@@H](C(=O)O)N",
    "PHE": "c1ccc(cc1)C[C@@H](C(=O)O)N",
    "PRO": "C1C[C@H](NC1)C(=O)O",
    "SER": "C([C@@H](C(=O)O)N)O",
    "THR": "C[C@H]([C@@H](C(=O)O)N)O",
    "TRP": "c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N",
    "TYR": "c1cc(ccc1C[C@@H](C(=O)O)N)O",
    "VAL": "CC(C)[C@@H](C(=O)O)N",
    "SEC": "C([C@@H](C(=O)O)N)[SeH]",
    "PYL": "C[C@@H]1CC=N[C@H]1C(=O)NCCCC[C@@H](C(=O)O)N",
    "MSE": "C[Se]CC[C@@H](C(=O)O)N",
    "MLZ": "CNCCCC[C@@H](C(=O)O)N",
    "MLY": "CN(C)CCCC[C@@H](C(=O)O)N",
    "M3L": "C[N+](C)(C)CCCC[C@@H](C(=O)O)N",
    "SEP": "C([C@@H](C(=O)O)N)OP(=O)(O)O",
    "TPO": "C[C@H]([C@@H](C(=O)O)N)OP(=O)(O)O",
    "PTR": "c1cc(ccc1C[C@@H](C(=O)O)N)OP(=O)(O)O",
    # RNA bases (canonical = 5'-phosphate + free 3'-OH)
    "G": "c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N=C(NC2=O)N",
    "G5": "c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N=C(NC2=O)N",
    "G3": "c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N=C(NC2=O)N",
    "C": "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    "C5": "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)CO)O)O",
    "C3": "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    "U": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    "U5": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)CO)O)O",
    "U3": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    "A": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N",
    "A5": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N",
    "A3": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N",
    # DNA bases (RCSB canonical; "T" reuses the DT skeleton since RCSB has no T chemcomp)
    "T": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)O)O",
    "T5": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)CO)O",
    "T3": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)O)O",
    "DG": "c1nc2c(n1[C@H]3C[C@@H]([C@H](O3)COP(=O)(O)O)O)N=C(NC2=O)N",
    "DG5": "c1nc2c(n1[C@H]3C[C@@H]([C@H](O3)CO)O)N=C(NC2=O)N",
    "DG3": "c1nc2c(n1[C@H]3C[C@@H]([C@H](O3)COP(=O)(O)O)O)N=C(NC2=O)N",
    "DC": "C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)COP(=O)(O)O)O",
    "DC5": "C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)CO)O",
    "DC3": "C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)COP(=O)(O)O)O",
    "DA": "c1nc(c2c(n1)n(cn2)[C@H]3C[C@@H]([C@H](O3)COP(=O)(O)O)O)N",
    "DA5": "c1nc(c2c(n1)n(cn2)[C@H]3C[C@@H]([C@H](O3)CO)O)N",
    "DA3": "c1nc(c2c(n1)n(cn2)[C@H]3C[C@@H]([C@H](O3)COP(=O)(O)O)O)N",
    "DT": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)O)O",
    "DT5": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)CO)O",
    "DT3": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)O)O",
}
