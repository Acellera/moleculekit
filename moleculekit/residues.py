from collections import namedtuple


_Residue = namedtuple("Residue", ["resname", "single_letter", "resname_variants"])


PROTEIN_RESIDUES = {
    "Alanine": _Residue(resname="ALA", single_letter="A", resname_variants=[]),
    "Arginine": _Residue(resname="ARG", single_letter="R", resname_variants=["AR0"]),
    "Asparagine": _Residue(resname="ASN", single_letter="N", resname_variants=[]),
    "Aspartate": _Residue(resname="ASP", single_letter="D", resname_variants=["ASH"]),
    "Cysteine": _Residue(
        resname="CYS", single_letter="C", resname_variants=["CYM", "CYX"]
    ),
    "Glutamine": _Residue(resname="GLN", single_letter="Q", resname_variants=[]),
    "Glutamate": _Residue(resname="GLU", single_letter="E", resname_variants=["GLH"]),
    "Glycine": _Residue(resname="GLY", single_letter="G", resname_variants=[]),
    "Histidine": _Residue(
        resname="HIS",
        single_letter="H",
        resname_variants=["HID", "HIE", "HIP", "HSD", "HSE", "HSP"],
    ),
    "Isoleucine": _Residue(resname="ILE", single_letter="I", resname_variants=[]),
    "Leucine": _Residue(resname="LEU", single_letter="L", resname_variants=[]),
    "Lysine": _Residue(
        resname="LYS", single_letter="K", resname_variants=["LSN", "LYN"]
    ),
    "Methionine": _Residue(resname="MET", single_letter="M", resname_variants=[]),
    "Phenylalanine": _Residue(resname="PHE", single_letter="F", resname_variants=[]),
    "Proline": _Residue(resname="PRO", single_letter="P", resname_variants=[]),
    "Serine": _Residue(resname="SER", single_letter="S", resname_variants=[]),
    "Threonine": _Residue(resname="THR", single_letter="T", resname_variants=[]),
    "Tryptophan": _Residue(resname="TRP", single_letter="W", resname_variants=[]),
    "Tyrosine": _Residue(resname="TYR", single_letter="Y", resname_variants=[]),
    "Valine": _Residue(resname="VAL", single_letter="V", resname_variants=[]),
    "Selenocysteine": _Residue(resname="SEC", single_letter="U", resname_variants=[]),
    "Pyrrolysine": _Residue(resname="PYL", single_letter="O", resname_variants=[]),
}
MODIFIED_PROTEIN_RESIDUES = {
    "Selenomethionine": _Residue(resname="MSE", single_letter="M", resname_variants=[]),
    "N-methyl-lysine": _Residue(resname="MLZ", single_letter="K", resname_variants=[]),
    "N-dimethyl-lysine": _Residue(
        resname="MLY", single_letter="K", resname_variants=[]
    ),
}
NUCLEIC_RESIDUES = {
    "Guanine": _Residue(
        resname="G",
        single_letter="G",
        resname_variants=["G5", "G3", "DG", "DG5", "DG3"],
    ),
    "Cytosine": _Residue(
        resname="C",
        single_letter="C",
        resname_variants=["C5", "C3", "DC", "DC5", "DC3"],
    ),
    "Uracil": _Residue(resname="U", single_letter="U", resname_variants=["U5", "U3"]),
    "Adenine": _Residue(
        resname="A",
        single_letter="A",
        resname_variants=["A5", "A3", "DA", "DA5", "DA3"],
    ),
    "Thymine": _Residue(
        resname="T",
        single_letter="T",
        resname_variants=["T5", "T3", "DT", "DT5", "DT3"],
    ),
}

SINGLE_LETTER_RESIDUE_NAME_TABLE = {}
ORIGINAL_RESIDUE_NAME_TABLE = {}
for rr in (PROTEIN_RESIDUES | NUCLEIC_RESIDUES).values():
    ORIGINAL_RESIDUE_NAME_TABLE[rr.resname] = rr.resname
    SINGLE_LETTER_RESIDUE_NAME_TABLE[rr.resname] = rr.single_letter
    for variant in rr.resname_variants:
        ORIGINAL_RESIDUE_NAME_TABLE[variant] = rr.resname
        SINGLE_LETTER_RESIDUE_NAME_TABLE[variant] = rr.single_letter


SINGLE_LETTER_MODIFIED_RESIDUE_NAME_TABLE = {
    rr.resname: rr.single_letter for rr in MODIFIED_PROTEIN_RESIDUES.values()
}
