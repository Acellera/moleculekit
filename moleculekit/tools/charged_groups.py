# (c) 2015-2026 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""Which atoms of a structure carry a charge.

The knowledge of which atoms carry a residue's charge lives in
:data:`moleculekit.residues.CHARGED_RESIDUE_ATOMS`. This module layers that
table with the charges a table cannot express: nucleic phosphates, free
termini, formal charges assigned at templating, and monatomic ions.

Anything it cannot classify is returned in a second list rather than dropped.
A charge that goes unreported is invisible, and an invisible charge changes a
protonation decision with nothing anywhere saying so.
"""

from typing import TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import logging

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)

# Non-bridging phosphate oxygens, in both PDB v3 and AMBER naming, in
# preference order for the group's representative atom. OP2 is what
# interactions.get_nucleic_charged selects today.
_PHOSPHATE_CENTERS = ("OP2", "O2P", "OP1", "O1P")
# The same names, as the set of atoms that carry the phosphate charge. Aliased
# rather than written out a second time: every name eligible to carry the
# charge must also be eligible to represent the group, or _phosphate_group
# would find charge atoms and no center and hand out a ChargedGroup whose
# center is None. Aliasing makes that invariant structural instead of a
# coincidence between two literals that a future edit can break silently.
_PHOSPHATE_OXYGENS = _PHOSPHATE_CENTERS

# Human-readable identity per (resname, charge) of every non-zero entry in
# CHARGED_RESIDUE_ATOMS. Every one of them needs a row here: the table lookup
# is unconditional, so a charged resname added to the table without a label
# raises KeyError inside the scan rather than degrading. A test derives the
# required keys from the table so a future addition fails there instead.
_CHARGE_LABELS = {
    ("ASP", -1): "carboxylate",
    ("GLU", -1): "carboxylate",
    ("CYM", -1): "thiolate",
    ("TYM", -1): "phenolate",
    ("LYS", 1): "ammonium",
    ("ARG", 1): "guanidinium",
    ("HIP", 1): "imidazolium",
    ("HSP", 1): "imidazolium",
}


# eq=False because the dataclass holds a numpy array: the generated __eq__
# compares field by field, so g1 == g2 returns an array and any truth test on
# it raises. Identity comparison is what a list of groups actually needs.
#
# The Attributes section below groups every field with its neighbours instead
# of giving each one its own entry. A single-name entry makes Sphinx emit a
# second description of a field autodoc already documents from the annotation:
# one build warning and two rendered entries per field.
@dataclass(eq=False)
class ChargedGroup:
    """One charge-carrying group of atoms.

    Attributes
    ----------
    atoms, charge, center, label : np.ndarray, int, int, str
        The charge itself. ``atoms`` holds the indices of every heavy atom
        carrying it, and distances to the group are measured to those atoms.
        ``charge`` is the formal charge. ``center`` is the index of one
        representative atom, never ``None``, because every source picks its
        representative from the same names it accepts as charge carriers.
        ``label`` is a human-readable identity for a report, e.g.
        ``"carboxylate"``.
    resname, resid, insertion, chain : str, int, str, str
        Identity of the residue the group belongs to.
    source, sign_only : str, bool
        ``source`` names the rule that produced the group: ``"table"``,
        ``"phosphate"``, ``"terminus"``, ``"formalcharge"`` or ``"ion"``.
        ``sign_only`` is True when ``charge`` records only a sign and not a
        real magnitude, which is the case for every ion: no ionic-charge
        magnitude table exists in the library, so a zinc and a molybdate are
        written +1 and -1. A report must not print that numeral as a
        magnitude, and a positive ``sign_only`` group is exactly a monatomic
        cation, which is what lets
        :func:`reviewProtonation <moleculekit.tools.protonation_review.reviewProtonation>`
        find the metals it flags on. Both consumers branch on this flag rather
        than matching ``label`` or ``source`` text, so renaming either cannot
        silently switch a rule off.
    """

    atoms: np.ndarray
    charge: int
    center: int
    label: str
    resname: str
    resid: int
    insertion: str
    chain: str
    source: str
    sign_only: bool = False


def _residue_identity(mol, idx):
    rep = idx[0]
    return (
        str(mol.resname[rep]),
        int(mol.resid[rep]),
        str(mol.insertion[rep]),
        str(mol.chain[rep]),
    )


def _atoms_named(mol, idx, names):
    return idx[np.isin(mol.name[idx], names)]


def _first_named(mol, idx, names):
    """First atom in idx matching a name, trying names in the given order.

    Unlike :func:`_atoms_named`, which returns matches in ascending atom-index
    order, this walks the caller's preference list itself. That distinction
    matters whenever the preferred name for a representative atom does not
    happen to be the one written earliest in the file, e.g. a residue whose
    OP1 precedes its OP2.
    """
    residue_names = mol.name[idx]
    for name in names:
        match = idx[residue_names == name]
        if len(match):
            return int(match[0])
    return None


def _table_group(mol, idx):
    """Charge from CHARGED_RESIDUE_ATOMS. Returns a list of 0 or 1 groups."""
    from moleculekit.residues import CHARGED_RESIDUE_ATOMS

    resname, resid, insertion, chain = _residue_identity(mol, idx)
    entry = CHARGED_RESIDUE_ATOMS.get(resname)
    if entry is None or entry.charge == 0:
        return []

    atoms = _atoms_named(mol, idx, entry.atoms)
    center = _atoms_named(mol, idx, (entry.center,))
    if len(atoms) == 0 or len(center) == 0:
        # A residue whose charge-carrying atoms are not in the file at all
        # (unmodelled sidechain density) carries no measurable charge here.
        # chargedGroups reports it as unclassified instead, via
        # _charge_atoms_are_absent.
        return []
    label = _CHARGE_LABELS[(resname, entry.charge)]
    return [
        ChargedGroup(
            atoms=atoms,
            charge=entry.charge,
            center=int(center[0]),
            label=label,
            resname=resname,
            resid=resid,
            insertion=insertion,
            chain=chain,
            source="table",
        )
    ]


def _charge_atoms_are_absent(mol, idx):
    """True for a charged table residue whose charge-carrying atoms are missing.

    Unmodelled sidechain density truncates a residue: an ASP arriving with no
    OD1 and no OD2 still carries -1 according to the table, but the structure
    cannot say where. :func:`_table_group` therefore produces no group, and
    without this predicate the residue would also stay out of
    ``unclassified``, because :func:`_is_known_residue` calls an ASP known.
    """
    from moleculekit.residues import CHARGED_RESIDUE_ATOMS

    entry = CHARGED_RESIDUE_ATOMS.get(str(mol.resname[idx[0]]))
    if entry is None or entry.charge == 0:
        return False
    return len(_atoms_named(mol, idx, entry.atoms)) == 0 or len(
        _atoms_named(mol, idx, (entry.center,))
    ) == 0


def _phosphate_group(mol, idx):
    """-1 per nucleic phosphate. A rule rather than 40 table rows."""
    from moleculekit.residues import NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS

    resname, resid, insertion, chain = _residue_identity(mol, idx)
    if resname not in NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS:
        return []
    atoms = _atoms_named(mol, idx, _PHOSPHATE_OXYGENS)
    if len(atoms) == 0:
        # A 5'-terminal nucleotide with no phosphate.
        return []
    center = _first_named(mol, idx, _PHOSPHATE_CENTERS)
    return [
        ChargedGroup(
            atoms=atoms,
            charge=-1,
            center=center,
            label="phosphate",
            resname=resname,
            resid=resid,
            insertion=insertion,
            chain=chain,
            source="phosphate",
        )
    ]


# N-terminal ammonium hydrogen names. PDB2PQR writes H1/H2/H3; the HT* triplet
# is the CHARMM convention. Plain "H" is the ordinary backbone amide hydrogen
# name and is included too: when the terminal residue was produced by
# mutateResidue, PDB2PQR keeps that original amide hydrogen's name for one of
# the three ammonium protons instead of renumbering it, so a residue can carry
# "H", "H2" and "H3" rather than "H1", "H2" and "H3". Adding "H" here is safe
# for every other residue because an interior backbone N carries exactly one
# hydrogen named "H", never three. All three present means a charged
# terminus; two means a neutral one, which carries no charge and yields no
# group.
_NTERM_HYDROGENS = ("H", "H1", "H2", "H3", "HT1", "HT2", "HT3")


def _terminus_groups(mol, idx, hydrogen_count):
    """Free-terminus charges.

    A C-terminal carboxylate is marked by an ``OXT``. An N-terminal ammonium is
    marked by three hydrogens on the backbone ``N``, found either by name or,
    when a real bond graph is present, by counting bonded hydrogens.

    Both paths are needed. A molecule returned by
    :func:`systemPrepare <moleculekit.tools.preparation.systemPrepare>` has
    hydrogens but almost no bonds, so the bond count alone finds nothing
    there. A molecule read from a bonded file has a real graph but may use a
    naming convention neither list anticipates, so the name match alone is
    not enough either.

    Additive with :func:`_table_group`: an N-terminal lysine carries both a
    sidechain ammonium and a terminal one.
    """
    from moleculekit.residues import PROTEIN_RESIDUE_NAMES_WITH_VARIANTS

    resname, resid, insertion, chain = _residue_identity(mol, idx)
    # A free terminus is a polymer concept. Without this gate a ligand carrying
    # an atom named N plus SMILES-style sequential hydrogens H1/H2/H3 would be
    # reported as a +1 ammonium it does not have.
    #
    # Capping groups are NOT admitted here: NME in the tleap convention is
    # N, H, C, H1, H2, H3, so four of its names are in
    # _NTERM_HYDROGENS, the >= 3 rule fires, and a neutral C-terminal amide cap
    # is reported as a +1 ammonium into the charge scan of every titratable
    # residue near it. The bond-count path disagrees (NME's N carries one
    # bonded H) but the two paths are OR'd, so it cannot overrule the names.
    # Excluding caps loses nothing real: ACE has no N at all, and NME, NMA,
    # NHE and NH2 are neutral amides, so no capping group can carry a genuine
    # terminal charge.
    if resname not in PROTEIN_RESIDUE_NAMES_WITH_VARIANTS:
        return []
    groups = []

    oxt = _atoms_named(mol, idx, ("OXT",))
    if len(oxt):
        atoms = _atoms_named(mol, idx, ("O", "OXT"))
        groups.append(
            ChargedGroup(
                atoms=atoms,
                charge=-1,
                center=int(oxt[0]),
                label="carboxylate",
                resname=resname,
                resid=resid,
                insertion=insertion,
                chain=chain,
                source="terminus",
            )
        )

    backbone_n = _atoms_named(mol, idx, ("N",))
    named_h = _atoms_named(mol, idx, _NTERM_HYDROGENS)
    charged_nterm = len(named_h) >= 3 or (
        len(backbone_n) and hydrogen_count[backbone_n[0]] >= 3
    )
    if len(backbone_n) and charged_nterm:
        groups.append(
            ChargedGroup(
                atoms=backbone_n,
                charge=1,
                center=int(backbone_n[0]),
                label="ammonium",
                resname=resname,
                resid=resid,
                insertion=insertion,
                chain=chain,
                source="terminus",
            )
        )
    return groups


def _formalcharge_groups(mol, idx):
    """Groups from assigned formal charges, one per residue per charge sign.

    Split by sign rather than netted, so a zwitterion is not collapsed into one
    misleading number.

    This source is meaningful only after formal charges have been assigned
    at templating: a residue the caller templated before preparation, or a
    non-canonical or renamed terminus. A plain canonical titratable residue
    carries formalcharge 0 and gets its charge from the residue-name table
    instead. A freshly read, untemplated structure carries zeros throughout,
    so it gets nothing here and the residue in ``unclassified`` instead,
    deliberately, to make the stage visible rather than return a confident
    zero.

    The distance is measured to the atoms actually carrying the charge. A
    charge delocalized over a symmetric group but assigned to one atom (an
    amidinium nitrogen in an RCSB SMILES) measures to that atom, not to the
    group's centroid. That is a property of the formalcharge field itself.
    """
    resname, resid, insertion, chain = _residue_identity(mol, idx)
    charges = mol.formalcharge[idx]
    groups = []
    for sign in (1, -1):
        atoms = idx[np.sign(charges) == sign]
        if len(atoms) == 0:
            continue
        total = int(np.sum(mol.formalcharge[atoms]))
        groups.append(
            ChargedGroup(
                atoms=atoms,
                charge=total,
                center=int(atoms[0]),
                label="formal charge",
                resname=resname,
                resid=resid,
                insertion=insertion,
                chain=chain,
                source="formalcharge",
            )
        )
    return groups


def _is_known_residue(resname):
    """True for a residue whose chemistry the library already knows.

    Used to decide what NOT to report as unclassified. The test cannot be
    "absent from CHARGED_RESIDUE_ATOMS": that table holds only the titratable
    residues, so every ordinary amino acid (ALA, GLY, SER, LEU, ...) is absent
    from it. Using it here would report about 150 of prepared 3PTB's 223
    residues as unclassified and bury the two entries that matter.

    Lipids are deliberately NOT part of this union. A phospholipid head group
    (POPC's choline, POPS's and POPA's phosphate) carries a real charge that no
    source in this module models, so a charged lipid is reportable: nothing in
    the library knows its chemistry, and the report collapsing unclassified
    entries by resname is what keeps a bilayer from flooding the list.

    An ion code that reaches here is genuinely unclassified: ION_RESIDUE_NAMES
    is deliberately NOT part of this union, so a code in none of
    METAL_ION_RESIDUE_NAMES, CATIONIC_ION_RESIDUE_NAMES and
    ANIONIC_ION_RESIDUE_NAMES is reported rather than assumed neutral.
    """
    from moleculekit.residues import (
        PROTEIN_RESIDUE_NAMES_WITH_VARIANTS,
        NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS,
        WATER_RESIDUE_NAMES,
        CAP_RESIDUE_NAMES,
    )

    return (
        resname in PROTEIN_RESIDUE_NAMES_WITH_VARIANTS
        or resname in NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS
        or resname in WATER_RESIDUE_NAMES
        or resname in CAP_RESIDUE_NAMES
    )


def _ion_group(mol, idx):
    """Monatomic ion charges, by residue name.

    A code in none of METAL_ION_RESIDUE_NAMES, CATIONIC_ION_RESIDUE_NAMES and
    ANIONIC_ION_RESIDUE_NAMES yields nothing and lands in ``unclassified``.
    Sign and identity only: no ionic-charge magnitude table exists in the
    library.

    CATIONIC_ION_RESIDUE_NAMES is consulted alongside METAL_ION_RESIDUE_NAMES
    and treated identically. Those two sets are separate because the metal set
    is defined as the element-symbol naming convention and autoSegment consumes
    it to classify residues, but a cation is a cation to this module: the
    CHARMM ion names (CAL, CES, POT, SOD) and the oxidation-state PDB codes
    (CU1, MN3) name real monatomic cations, and leaving them unclassified
    stopped reviewProtonation's metal rule from ever firing on them.
    """
    from moleculekit.residues import (
        METAL_ION_RESIDUE_NAMES,
        CATIONIC_ION_RESIDUE_NAMES,
        ANIONIC_ION_RESIDUE_NAMES,
    )

    resname, resid, insertion, chain = _residue_identity(mol, idx)

    if resname in ANIONIC_ION_RESIDUE_NAMES:
        charge, label = -1, "ion anion"
    elif (
        resname in METAL_ION_RESIDUE_NAMES or resname in CATIONIC_ION_RESIDUE_NAMES
    ) and len(idx) == 1:
        # The single-atom guard residues.py documents: several of those codes
        # are also real polyatomic ligands (CO is carbon monoxide as well as
        # cobalt), and a metal inside a cofactor keeps its real bonds.
        charge, label = 1, "metal cation"
    else:
        return []

    return [
        ChargedGroup(
            atoms=idx,
            charge=charge,
            center=int(idx[0]),
            label=label,
            resname=resname,
            resid=resid,
            insertion=insertion,
            chain=chain,
            source="ion",
            sign_only=True,
        )
    ]


def _hydrogen_counts(mol):
    """Per-atom count of bonded hydrogens.

    Zero everywhere when the molecule has no bonds, which is the normal case
    for a molecule straight out of systemPrepare. The name-based path in
    :func:`_terminus_groups` is what covers that case; this count is the
    fallback for molecules that do carry a real bond graph.
    """
    counts = np.zeros(mol.numAtoms, dtype=int)
    if mol.bonds is None or len(mol.bonds) == 0:
        return counts
    is_h = mol.element == "H"
    for a, b in mol.bonds:
        a, b = int(a), int(b)
        if is_h[b]:
            counts[a] += 1
        if is_h[a]:
            counts[b] += 1
    return counts


def chargedGroups(mol: "Molecule", sel="all"):
    """Find every charge-carrying group of atoms in a structure.

    Layers five sources. The residue-name table, nucleic phosphates and free
    termini are additive, so one residue can yield several groups: an
    N-terminal lysine carries both a sidechain ammonium and a terminal one.
    Assigned formal charges cover anything the table does not know (ligands,
    non-canonical residues, anything templated), and monatomic ions are
    classified by residue name.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure to scan. Terminal ammonium detection needs hydrogens and
        bonds; without them no terminal ammonium is reported.
    sel : str or np.ndarray
        Atom selection to scan. A selection string, a boolean mask, or an
        integer index array.

    Returns
    -------
    groups : list of :class:`ChargedGroup`
        One entry per charge-carrying group.
    unclassified : list of tuple
        ``(resname, resid, insertion, chain)`` for each residue in ``sel``
        whose charge could not be determined: one that matched no source and
        carries no assigned formal charge, and one the residue-name table says
        is charged but whose charge-carrying atoms are missing from the file,
        so no distance to that charge can be measured.

    Example
    -------
    >>> groups, unclassified = chargedGroups(mol)  # doctest: +SKIP
    """
    sel_mask = mol.atomselect(sel)
    sel_idx = np.where(sel_mask)[0]
    if len(sel_idx) == 0:
        return [], []

    _, residue_idx = mol.getResidues(sel=sel_mask, return_idx=True)
    residue_idx = [sel_idx[i] for i in residue_idx]

    hydrogen_count = _hydrogen_counts(mol)

    groups = []
    unclassified = []
    for idx in residue_idx:
        found = _table_group(mol, idx)
        found += _phosphate_group(mol, idx)
        found += _terminus_groups(mol, idx, hydrogen_count)

        # Assigned formal charges are consulted for every atom no earlier
        # source already claimed. Gating on the resname instead drops real
        # charges: a CYS carrying an assigned -1 on SG has a neutral table
        # entry, so a resname gate reports nothing at all. Claim-based gating
        # also still prevents double counting, because a charged table entry
        # has already claimed its own atoms.
        claimed = set()
        for g in found:
            claimed.update(int(a) for a in g.atoms)
        unclaimed = np.array(
            [int(i) for i in idx if int(i) not in claimed], dtype=int
        )
        if len(unclaimed):
            found += _formalcharge_groups(mol, unclaimed)
        found += _ion_group(mol, idx)

        resname = str(mol.resname[idx[0]])
        if found:
            groups.extend(found)
        elif not _is_known_residue(resname) or _charge_atoms_are_absent(mol, idx):
            unclassified.append(_residue_identity(mol, idx))

    return groups, unclassified
