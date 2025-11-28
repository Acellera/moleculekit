from collections import namedtuple


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
