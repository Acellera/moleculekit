# (c) 2015-2026 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""GLYCAM-06 naming tables and glycan analysis helpers.

This module holds the static tables needed to translate a PDB carbohydrate
residue name (e.g. ``NAG``, ``BMA``, ``SIA``) plus its glycosidic linkage
positions into the corresponding 3-character GLYCAM-06 residue name (e.g.
``4YB``), and the inverse mapping recovering linkage positions from a
GLYCAM residue name.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from moleculekit.residues import (
    ION_RESIDUE_NAMES,
    METAL_ION_RESIDUE_NAMES,
    PROTEIN_RESIDUE_NAMES_WITH_VARIANTS,
    WATER_RESIDUE_NAMES,
)

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SugarTemplate:
    """GLYCAM identity of one PDB carbohydrate residue name.

    Attributes
    ----------
    letter : str
        GLYCAM one-letter sugar code. Case encodes the sugar's configuration:
        upper-case is D, lower-case is L (e.g. ``"M"`` for D-mannose, ``"f"``
        for L-fucose).
    anomer : str
        Anomeric configuration at the ring-closing carbon: ``"A"`` for alpha
        or ``"B"`` for beta.
    anomeric_carbon : str
        Atom name of the anomeric carbon, ``"C1"`` for most sugars or
        ``"C2"`` for sialic acid and fructose (both ketoses, with the
        anomeric center one carbon further along the chain than an
        aldose's).
    anomeric_oxygen : str
        Atom name of the anomeric hydroxyl oxygen, ``"O1"`` for most sugars
        or ``"O2"`` for sialic acid and fructose.
    ring_oxygen : str
        Atom name of the ring-closing oxygen, ``"O5"`` for most sugars
        (including fructose, a furanose) or ``"O6"`` for sialic acid (a
        pyranose whose two extra ring carbons push its ring oxygen one
        position further along).
    atom_renames : dict
        Mapping from PDB atom name to the GLYCAM atom name it must be
        renamed to before building (e.g. the N-acetyl or N-glycolyl
        substituent atoms), empty for sugars with no such substituent.
    """

    letter: str
    anomer: str
    anomeric_carbon: str
    anomeric_oxygen: str
    ring_oxygen: str
    atom_renames: dict


# N-acetyl substituent (GlcNAc/GalNAc): PDB C7/O7/C8 -> GLYCAM C2N/O2N/CME.
_ACETYL_RENAMES = {"C7": "C2N", "O7": "O2N", "C8": "CME"}
# N-acetyl substituent on sialic acid, offset by the two extra ring carbons:
# PDB C10/O10/C11 -> GLYCAM C5N/O5N/CME.
_SIALIC_RENAMES = {"C10": "C5N", "O10": "O5N", "C11": "CME"}

# GLYCAM identity of every PDB carbohydrate residue name this module supports.
# Keyed by the 3-character PDB Chemical Component Dictionary resname.
GLYCAM_SUGARS = {
    "NAG": SugarTemplate("Y", "B", "C1", "O1", "O5", _ACETYL_RENAMES),
    "NDG": SugarTemplate("Y", "A", "C1", "O1", "O5", _ACETYL_RENAMES),
    "BMA": SugarTemplate("M", "B", "C1", "O1", "O5", {}),
    "MAN": SugarTemplate("M", "A", "C1", "O1", "O5", {}),
    "GAL": SugarTemplate("L", "B", "C1", "O1", "O5", {}),
    "GLA": SugarTemplate("L", "A", "C1", "O1", "O5", {}),
    "BGC": SugarTemplate("G", "B", "C1", "O1", "O5", {}),
    "GLC": SugarTemplate("G", "A", "C1", "O1", "O5", {}),
    "FUC": SugarTemplate("f", "A", "C1", "O1", "O5", {}),
    "FUL": SugarTemplate("f", "B", "C1", "O1", "O5", {}),
    "XYP": SugarTemplate("X", "B", "C1", "O1", "O5", {}),
    "XYS": SugarTemplate("X", "A", "C1", "O1", "O5", {}),
    "SIA": SugarTemplate("S", "A", "C2", "O2", "O6", _SIALIC_RENAMES),
    "NGA": SugarTemplate("V", "B", "C1", "O1", "O5", _ACETYL_RENAMES),
    "A2G": SugarTemplate("V", "A", "C1", "O1", "O5", _ACETYL_RENAMES),
    # Beta-D-fructofuranose. Ketohexose: anomeric carbon C2, exocyclic C1
    # and C6 hydroxymethyls, ring oxygen O5 (a furanose, not the O6 of
    # sialic acid's pyranose ring despite sharing the C2/O2 anomeric
    # atoms). Anomer characters D (alpha) / U (beta), not the usual A/B,
    # established by comparing tleap-built 0CD/0CU against CCD FRU's
    # signed volume at C2. atom_renames is empty: GLYCAM 0CU's heavy
    # atoms (C1-C6, O1, O3, O4, O5, O6) already match CCD FRU's naming,
    # which only adds the free anomeric hydroxyl O2 that the ROH split
    # handles.
    "FRU": SugarTemplate("C", "U", "C2", "O2", "O5", {}),
    # Uronic acids: their C6 is a carboxylate (O6A/O6B) rather than a
    # hydroxymethyl, but that is invisible to this table since it only
    # tracks the anomeric carbon and ring oxygen, both ordinary C1/O5.
    # GLYCAM 06j ships units for exactly three uronic acids, each in one
    # configuration: GlcA (Z, D only), GalA (O, D only), IdoA (u, L only).
    # It has no mannuronic or guluronic acid unit, so BEM and LGU are
    # deliberately absent here (see test_glycam_resname_unsupported_uronic).
    "BDP": SugarTemplate("Z", "B", "C1", "O1", "O5", {}),  # beta-D-GlcA
    "GCU": SugarTemplate("Z", "A", "C1", "O1", "O5", {}),  # alpha-D-GlcA
    "GTR": SugarTemplate("O", "B", "C1", "O1", "O5", {}),  # beta-D-GalA
    "ADA": SugarTemplate("O", "A", "C1", "O1", "O5", {}),  # alpha-D-GalA
    "IDR": SugarTemplate("u", "A", "C1", "O1", "O5", {}),  # alpha-L-IdoA
}


def _derive_letter_ring_atoms() -> dict:
    """Derive the anomeric-carbon / ring-oxygen pair for each GLYCAM sugar
    letter from :data:`GLYCAM_SUGARS`, so the sugar table stays the single
    source of truth for it.

    Asserts that every CCD code sharing a letter (e.g. ``NAG``/``NDG``,
    both letter ``Y``) agrees on these two atoms, rather than silently
    keeping whichever entry happened to be seen last while building the
    table.

    Returns
    -------
    table : dict
        Maps a GLYCAM one-letter sugar code to its
        ``(anomeric_carbon, ring_oxygen)`` atom-name pair.

    Raises
    ------
    AssertionError
        If two sugars sharing a GLYCAM letter disagree on their anomeric
        carbon or ring oxygen atom name.
    """
    table: dict = {}
    for resname, tmpl in GLYCAM_SUGARS.items():
        pair = (tmpl.anomeric_carbon, tmpl.ring_oxygen)
        if tmpl.letter in table and table[tmpl.letter] != pair:
            raise AssertionError(
                f"GLYCAM letter {tmpl.letter!r} disagrees on anomeric "
                f"carbon / ring oxygen between sugars sharing it: "
                f"{table[tmpl.letter]} vs {pair} from {resname!r}."
            )
        table[tmpl.letter] = pair
    return table


# GLYCAM one-letter sugar code -> (anomeric_carbon, ring_oxygen), derived
# from GLYCAM_SUGARS above at import time. Keyed by letter rather than CCD
# resname because glycamUnitMask and glycanBondsFromNames only ever see a
# GLYCAM 3-character unit name (never the original CCD resname) and so
# cannot look this up on a SugarTemplate directly. A letter absent from
# this table (i.e. not yet covered by GLYCAM_SUGARS) defaults to
# ("C1", "O5") at the call sites.
_LETTER_RING_ATOMS = _derive_letter_ring_atoms()

# Protein residues glycans may attach to: resname -> (GLYCAM unit, anchor atom,
# hydrogen displaced by the glycosidic bond; None = pick the anchor-atom H
# nearest the sugar anomeric carbon and rename the survivor to HD21)
GLYCAN_ANCHORS = {
    "ASN": ("NLN", "ND2", None),
    "SER": ("OLS", "OG", "HG"),
    "THR": ("OLT", "OG1", "HG1"),
    "HYP": ("OLP", "OD1", "HD1"),
}

# GLYCAM anchor residue name -> its side-chain anchor atom (the atom the
# first sugar's anomeric oxygen bonds to).
GLYCAM_ANCHOR_UNITS = {"NLN": "ND2", "OLS": "OG", "OLT": "OG1", "OLP": "OD1"}

# Upper bound, in Angstrom, for a glycosidic C-O / C-N bond. Used only by
# glycanBondsFromNames, which has no bond list to consult and must infer
# connectivity from geometry instead.
GLYCAN_LINK_CUTOFF = 1.8


def _skip_glycan_bond_residue(resname: str, natoms: int) -> bool:
    """Decide whether a residue must be ignored as a glycosidic bond partner.

    Follows the same water / ion skip applied to inter-residue bonds by
    :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`:
    a bond touching water or an ion is a crystallographic or coordination
    contact, never a covalent glycosidic linkage. The monatomic-metal check
    is additionally gated on the residue having exactly one atom, which
    matches :func:`moleculekit.tools.autosegment._classify_residues` rather
    than ``detectNonStandardResidues`` (whose metal-resname check has no
    atom-count gate).

    Parameters
    ----------
    resname : str
        Residue name of the candidate bond partner.
    natoms : int
        Number of atoms in that residue, used to gate the monatomic metal
        ion check (an organometallic cofactor keeps its resname's metal
        symbol but has more than one atom).

    Returns
    -------
    skip : bool
        True if the residue must be excluded from glycosidic bond analysis.
    """
    return (
        resname in WATER_RESIDUE_NAMES
        or resname in ION_RESIDUE_NAMES
        or (resname in METAL_ION_RESIDUE_NAMES and natoms == 1)
    )


# Linkage-position combination -> the GLYCAM name-building character encoding
# it. A single-digit position is its own character; multi-position
# combinations (branch points) use a letter, per the GLYCAM-06 naming scheme.
_LINKAGE_CHARS = {
    frozenset(): "0",
    frozenset({2, 3}): "Z",
    frozenset({2, 4}): "Y",
    frozenset({2, 6}): "X",
    frozenset({3, 4}): "W",
    frozenset({3, 6}): "V",
    frozenset({4, 6}): "U",
    frozenset({2, 3, 4}): "T",
    frozenset({2, 3, 6}): "S",
    frozenset({2, 4, 6}): "R",
    frozenset({3, 4, 6}): "Q",
    frozenset({2, 3, 4, 6}): "P",
}
for _p in range(1, 10):
    _LINKAGE_CHARS[frozenset({_p})] = str(_p)
# Inverse of _LINKAGE_CHARS, keyed by the single character.
_LINKAGE_POSITIONS = {v: tuple(sorted(k)) for k, v in _LINKAGE_CHARS.items()}

# All 3-char units in GLYCAM_06j-1.prep whose sugar letter is covered by
# GLYCAM_SUGARS. Extracted from AmberTools 24.8; regenerate with:
#   grep -E '^[0-9A-Za-z]{3,4} +INT' $AMBERHOME/dat/leap/prep/GLYCAM_06j-1.prep
# then keep a unit whose first character is one of _LINKAGE_CHARS's values,
# whose second character (the sugar letter) is a letter used in
# GLYCAM_SUGARS, and whose third character is one of that letter's two
# anomer characters: ("D", "U") for "C" (fructose's furanose ring; its
# ("A", "B") pyranose forms are out of scope), ("A", "B") for every other
# letter. This keeps both anomer forms of a letter even where GLYCAM_SUGARS
# only has a CCD entry for one of them (e.g. sialic acid, letter "S", is
# alpha-only in GLYCAM_SUGARS but both 0SA and 0SB are genuine units).
GLYCAM_UNIT_NAMES = frozenset(
    """0CD 0CU 0fA 0fB 0GA 0GB 0LA 0LB 0MA 0MB 0OA 0OB 0SA 0SB 0uA 0uB 0VA 0VB
    0XA 0XB 0YA 0YB 0ZA 0ZB
    1CD 1CU 1fA 1fB 1GA 1GB 1LA 1LB 1MA 1MB 1OA 1OB 1uA 1uB 1VA 1VB 1XA 1XB
    1YA 1YB 1ZA 1ZB
    2CD 2CU 2fA 2fB 2GA 2GB 2LA 2LB 2MA 2MB 2OA 2OB 2uA 2uB 2XA 2XB 2ZA 2ZB
    3CD 3CU 3fA 3fB 3GA 3GB 3LA 3LB 3MA 3MB 3OA 3OB 3uA 3uB 3VA 3VB 3XA 3XB
    3YA 3YB 3ZA 3ZB
    4CD 4CU 4fA 4fB 4GA 4GB 4LA 4LB 4MA 4MB 4OA 4OB 4SA 4SB 4uA 4uB 4VA 4VB
    4XA 4XB 4YA 4YB 4ZA 4ZB
    6CD 6CU 6GA 6GB 6LA 6LB 6MA 6MB 6VA 6VB 6YA 6YB 7SA 7SB 8SA 8SB
    9SA 9SB PGA PGB PLA PLB PMA PMB
    QCD QCU QGA QGB QLA QLB QMA QMB QVA QVB QYA QYB
    RGA RGB RLA RLB RMA RMB SGA SGB SLA SLB SMA SMB
    TfA TfB TGA TGB TLA TLB TMA TMB TOA TOB TuA TuB TXA TXB TZA TZB
    UCD UCU UGA UGB ULA ULB UMA UMB UVA UVB UYA UYB
    VCD VCU VGA VGB VLA VLB VMA VMB VVA VVB VYA VYB
    WCD WCU WfA WfB WGA WGB WLA WLB WMA WMB WOA WOB WuA WuB WVA WVB WXA WXB
    WYA WYB WZA WZB
    XGA XGB XLA XLB XMA XMB YfA YfB YGA YGB YLA YLB YMA YMB YOA YOB YuA YuB
    YXA YXB YZA YZB
    ZfA ZfB ZGA ZGB ZLA ZLB ZMA ZMB ZOA ZOB ZuA ZuB ZXA ZXB ZZA ZZB""".split()
)


def _candidate_residue_groups(mol: "Molecule", codes) -> list:
    """Group ``mol``'s atoms into per-residue ``(atom_mask, resname)`` pairs
    for every residue whose resname is a key of ``codes``.

    Shared iteration helper for :func:`glycamUnitMask` and
    :func:`pdbSugarMask`: both need to inspect one candidate residue at a
    time (to check its atom composition), keyed by a different lookup
    table (:data:`GLYCAM_UNIT_NAMES` / :data:`GLYCAM_SUGARS`).

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to scan.
    codes : Container[str]
        Resnames to treat as candidates, e.g. :data:`GLYCAM_UNIT_NAMES` or
        :data:`GLYCAM_SUGARS`.

    Returns
    -------
    groups : list
        One ``(atom_mask, resname)`` tuple per matching residue, where
        ``atom_mask`` is a boolean array of shape ``(mol.numAtoms,)``.
    """
    from moleculekit.util import sequenceID

    candidate = np.isin(mol.resname, sorted(codes))
    if not candidate.any():
        return []
    uq = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))
    return [
        (uq == rid, str(mol.resname[uq == rid][0])) for rid in np.unique(uq[candidate])
    ]


def _has_glycam_ring_atoms(
    mol: "Molecule", rmask: np.ndarray, anomeric_carbon: str, ring_oxygen: str
) -> bool:
    """True if a residue carries the named anomeric carbon and ring oxygen
    atoms, positioned close enough together to be the real ring bond
    between them, rather than a same-named atom pair that coincides only
    because an unrelated ligand's own numbering scheme happens to reuse
    those two atom names.

    Atom names alone are not a reliable sugar-composition test: pyromellitic
    acid (PDB Chemical Component Dictionary code ``PMA``, which also
    happens to be a valid GLYCAM mannose-unit code) genuinely has atoms
    named ``C1`` and ``O5``, just several bonds apart on an aromatic ring
    rather than directly bonded. A real GLYCAM unit's anomeric carbon is
    always directly bonded to its own ring oxygen (a pyranose/furanose
    ring bond, around 1.4 A), so this additionally gates on proximity,
    reusing :data:`GLYCAN_LINK_CUTOFF` - already used elsewhere in this
    module for exactly this kind of "should be a real bond" geometric
    check.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule containing the residue.
    rmask : np.ndarray
        Boolean atom mask selecting the candidate residue.
    anomeric_carbon : str
        Expected anomeric carbon atom name (``"C1"`` or ``"C2"``).
    ring_oxygen : str
        Expected ring oxygen atom name (``"O5"`` or ``"O6"``).

    Returns
    -------
    present : bool
        True if both atoms exist in the residue and are within
        :data:`GLYCAN_LINK_CUTOFF` of each other.
    """
    c_idx = np.where(rmask & (mol.name == anomeric_carbon))[0]
    o_idx = np.where(rmask & (mol.name == ring_oxygen))[0]
    if len(c_idx) == 0 or len(o_idx) == 0:
        return False
    dist = np.linalg.norm(mol.coords[c_idx[0], :, 0] - mol.coords[o_idx[0], :, 0])
    return bool(dist < GLYCAN_LINK_CUTOFF)


def glycamUnitMask(mol: "Molecule") -> np.ndarray:
    """Boolean atom mask selecting genuine GLYCAM-06j sugar unit residues.

    Matching purely on a GLYCAM unit's 3-character resname
    (:data:`GLYCAM_UNIT_NAMES`) is unsafe: many of those codes are also
    real PDB Chemical Component Dictionary ligand codes with nothing to
    do with GLYCAM (``TLA`` is L-(+)-tartaric acid, ``TMA`` is
    tetramethylammonium, ``PGA`` is 2-phosphoglycolic acid, ``PMA`` is
    pyromellitic acid, and others), and two more (``1MA``, ``2MA``) are
    also AMBER modrna08 modified-ribonucleotide names. This gates the
    resname match on the sugar-like composition every GLYCAM unit
    template actually has: the anomeric carbon directly bonded to the
    ring oxygen GLYCAM always numbers alongside it, both looked up from
    the code's own sugar letter (the middle character of the 3-character
    code) in :data:`_LETTER_RING_ATOMS` (``C1``/``O5`` for most sugars,
    ``C2``/``O6`` for sialic acid, ``C2``/``O5`` for fructose) - see
    :func:`_has_glycam_ring_atoms`.

    For the two codes also claimed by modrna08 (``1MA``, ``2MA``), a
    residue is additionally required to carry no nitrogen atom. Both are
    mannose (``M``) linkage codes, so a genuine GLYCAM instance of either
    never has one, while every real modrna08 ribonucleotide does (its
    purine/pyrimidine base). This heuristic is deliberately narrow: it is
    NOT a general "sugars have no nitrogen" rule (GlcNAc/GalNAc/sialic-acid
    units all carry one from their N-acetyl group); it works only because
    the current collision set happens to be limited to nitrogen-free
    mannose codes. If GLYCAM or modrna08 ever add a colliding code on a
    different letter, this must be revisited - see
    ``test_glycam_modrna_collision_set_is_1ma_2ma`` in htmd's
    ``tests/test_amber_builder.py``, which pins today's exact collision
    set and is meant to fail first if that ever changes.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to scan. Assumed already renamed to GLYCAM unit
        names where applicable (i.e. run after :func:`glycamResname` /
        ``systemPrepare``, not on raw PDB Chemical Component Dictionary
        names - see :func:`pdbSugarMask` for that case).

    Returns
    -------
    mask : np.ndarray
        Boolean array, shape ``(mol.numAtoms,)``, True for every atom of
        a residue confirmed to be a genuine GLYCAM sugar unit.
    """
    from moleculekit.residues import MODIFIED_NUCLEIC_RESIDUE_NAMES

    mask = np.zeros(mol.numAtoms, dtype=bool)
    for rmask, code in _candidate_residue_groups(mol, GLYCAM_UNIT_NAMES):
        anomeric_carbon, ring_oxygen = _LETTER_RING_ATOMS.get(code[1], ("C1", "O5"))
        if not _has_glycam_ring_atoms(mol, rmask, anomeric_carbon, ring_oxygen):
            continue
        if code in MODIFIED_NUCLEIC_RESIDUE_NAMES and np.any(mol.element[rmask] == "N"):
            continue
        mask |= rmask
    return mask


def pdbSugarMask(mol: "Molecule") -> np.ndarray:
    """Boolean atom mask selecting genuine un-renamed PDB sugar residues.

    Companion to :func:`glycamUnitMask` for the opposite naming stage: a
    residue still carrying its original PDB Chemical Component Dictionary
    carbohydrate resname (a key of :data:`GLYCAM_SUGARS`, e.g.
    ``NAG``/``BMA``/``SIA``) rather than a GLYCAM-06j unit name. Used to
    catch a glycan that reaches a builder without first being renamed by
    ``systemPrepare``. Gated on the same composition-plus-geometry check
    as :func:`glycamUnitMask` (see :func:`_has_glycam_ring_atoms`), so an
    unrelated ligand that merely happens to reuse one of these codes is
    not misdetected.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to scan.

    Returns
    -------
    mask : np.ndarray
        Boolean array, shape ``(mol.numAtoms,)``, True for every atom of
        a residue confirmed to be a genuine un-renamed PDB sugar.
    """
    mask = np.zeros(mol.numAtoms, dtype=bool)
    for rmask, code in _candidate_residue_groups(mol, GLYCAM_SUGARS):
        tmpl = GLYCAM_SUGARS[code]
        if _has_glycam_ring_atoms(mol, rmask, tmpl.anomeric_carbon, tmpl.ring_oxygen):
            mask |= rmask
    return mask


def glycamResname(resname: str, linked_positions) -> str:
    """Construct the 3-character GLYCAM-06 unit name for a sugar residue.

    Combines the sugar's GLYCAM one-letter code and anomer with a character
    encoding which ring positions carry outgoing glycosidic linkages, and
    validates the result against the set of units GLYCAM-06j actually ships.

    Parameters
    ----------
    resname : str
        PDB Chemical Component Dictionary residue name of the sugar (e.g.
        ``"NAG"``, ``"BMA"``, ``"SIA"``). Must be a key of
        :data:`GLYCAM_SUGARS`.
    linked_positions : Iterable[int]
        Ring carbon positions (e.g. ``2``, ``3``, ``4``, ``6``) at which
        another sugar is glycosidically linked onto this one. Pass an empty
        iterable for a terminal (non-reducing end) sugar with no outgoing
        linkages.

    Returns
    -------
    name : str
        The 3-character GLYCAM-06 residue name, e.g. ``"4YB"``.

    Raises
    ------
    RuntimeError
        If ``resname`` is not in :data:`GLYCAM_SUGARS`, if ``linked_positions``
        is a combination :data:`_LINKAGE_CHARS` has no naming character for
        yet, or if the constructed name is absent from
        :data:`GLYCAM_UNIT_NAMES` (GLYCAM-06j does not ship that particular
        sugar/linkage combination).
    """
    if resname not in GLYCAM_SUGARS:
        raise RuntimeError(
            f"Sugar residue {resname} is not in the supported GLYCAM sugar "
            f"table ({sorted(GLYCAM_SUGARS)}). Add it to "
            f"moleculekit.tools.glycans.GLYCAM_SUGARS if GLYCAM supports it."
        )
    tmpl = GLYCAM_SUGARS[resname]
    key = frozenset(int(p) for p in linked_positions)
    if key not in _LINKAGE_CHARS:
        raise RuntimeError(
            f"This table has no naming character for linkage positions "
            f"{sorted(key)} of sugar {resname} (add it to "
            f"moleculekit.tools.glycans._LINKAGE_CHARS if GLYCAM ships a "
            f"unit for that combination)."
        )
    name = f"{_LINKAGE_CHARS[key]}{tmpl.letter}{tmpl.anomer}"
    if name not in GLYCAM_UNIT_NAMES:
        raise RuntimeError(
            f"Constructed GLYCAM unit {name} for sugar {resname} linked at "
            f"positions {sorted(key)} does not exist in the GLYCAM-06j "
            f"forcefield."
        )
    return name


def linkedPositionsFromGlycamResname(resname: str) -> tuple:
    """Recover the linked ring positions encoded in a GLYCAM-06 residue name.

    Inverse of the linkage-position character used by :func:`glycamResname`:
    reads the first character of ``resname`` and looks it up in the
    linkage-character table.

    Parameters
    ----------
    resname : str
        A 3-character GLYCAM-06 residue name, e.g. ``"4YB"`` or ``"UYB"``.

    Returns
    -------
    positions : tuple
        Sorted tuple of ring carbon positions with an outgoing glycosidic
        linkage, e.g. ``(4, 6)``. Empty tuple for a terminal sugar.
    """
    return _LINKAGE_POSITIONS[resname[0]]


@dataclass
class GlycanResidueInfo:
    """Per-residue result of :func:`analyzeGlycanResidues`.

    Attributes
    ----------
    linked_positions : tuple
        Ring carbon positions at which another sugar is glycosidically
        linked onto this residue, sorted ascending. Empty for a sugar with
        no outgoing linkage (a non-reducing terminal residue).
    anchor_res : int or None
        Index into the caller's ``residue_atom_idx`` list of the protein
        residue this sugar's anomeric carbon is bonded to. ``None`` when
        this sugar's anomeric carbon is bonded to another sugar instead of a
        protein anchor, or when it has a free reducing end.
    anchor_atom : str or None
        Name of the protein anchor atom the anomeric carbon bonds to (e.g.
        ``"ND2"``), or ``None`` when ``anchor_res`` is ``None``.
    free_reducing_end : bool
        True when nothing is bonded to this residue's anomeric carbon,
        regardless of whether the anomeric hydroxyl atom itself is resolved
        in the input structure.
    """

    linked_positions: tuple = ()
    anchor_res: int | None = None
    anchor_atom: str | None = None
    free_reducing_end: bool = False


def _scan_glycan_bond_graph(
    mol: "Molecule",
    bonds: np.ndarray,
    atom_to_res: np.ndarray,
    resname_of: list,
    natoms_of: list,
    info: dict,
) -> tuple:
    """Walk every inter-residue bond once to gather raw per-sugar linkage data.

    For each residue recognized as a sugar (a key of ``info``), records which
    of its numbered ring oxygens are bonded outward to another residue
    (``linked``), and which residue(s), if any, its own anomeric carbon is
    bonded to (``anomeric_bonded``). Interpreting that raw data into
    :class:`GlycanResidueInfo` (including anchor validation and the
    unmapped-sugar checks) is left to :func:`_classify_glycan_residue`.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule containing the candidate glycan(s).
    bonds : np.ndarray
        ``(n, 2)`` array of atom index pairs.
    atom_to_res : np.ndarray
        Maps each atom index to its residue index (``-1`` for atoms not
        covered by ``resname_of`` / ``natoms_of``).
    resname_of : list
        Residue name for each residue index.
    natoms_of : list
        Atom count for each residue index.
    info : dict
        Residue index -> :class:`GlycanResidueInfo`, one entry per sugar
        residue; only used here to test membership.

    Returns
    -------
    linked : dict
        Residue index -> set of ring positions with an outgoing linkage.
    anomeric_bonded : dict
        Residue index -> list of ``(partner_residue, partner_atom)`` found
        on that residue's own anomeric carbon.

    Raises
    ------
    RuntimeError
        If a sugar's numbered ring oxygen is bonded to a residue that is
        itself a carbohydrate but absent from :data:`GLYCAM_SUGARS`, or to a
        protein residue (glycans never attach via a ring oxygen of a
        protein).
    """
    linked = {ri: set() for ri in info}
    anomeric_bonded = {ri: [] for ri in info}

    for a, b in bonds:
        a, b = int(a), int(b)
        ra, rb = int(atom_to_res[a]), int(atom_to_res[b])
        if ra == rb or ra < 0 or rb < 0:
            continue
        if _skip_glycan_bond_residue(
            resname_of[ra], natoms_of[ra]
        ) or _skip_glycan_bond_residue(resname_of[rb], natoms_of[rb]):
            continue
        for x, rx, y, ry in ((a, ra, b, rb), (b, rb, a, ra)):
            if rx not in info:
                continue
            tmpl = GLYCAM_SUGARS[resname_of[rx]]
            name_x = str(mol.name[x])
            if name_x == tmpl.anomeric_carbon:
                anomeric_bonded[rx].append((ry, y))
            elif name_x.startswith("O") and name_x[1:].isdigit():
                position = int(name_x[1:])
                if ry in info:
                    linked[rx].add(position)
                    continue
                if str(mol.element[y]) == "H":
                    continue  # a hydroxyl hydrogen never occupies a linkage
                if resname_of[ry] in PROTEIN_RESIDUE_NAMES_WITH_VARIANTS:
                    raise RuntimeError(
                        f"O{position} of sugar {resname_of[rx]} is bonded "
                        f"to {resname_of[ry]} atom {mol.name[y]}, which is "
                        "not how GLYCAM attaches a glycan (a glycosidic "
                        "bond runs from a sugar's own anomeric carbon into "
                        "this position, never from a protein residue)."
                    )
                raise RuntimeError(
                    f"Residue {resname_of[ry]} is bonded into a glycan "
                    f"(its atom {mol.name[y]} links to O{position} of "
                    f"{resname_of[rx]}) but is not in the supported GLYCAM "
                    f"sugar table ({sorted(GLYCAM_SUGARS)}). Add it to "
                    "moleculekit.tools.glycans.GLYCAM_SUGARS if GLYCAM "
                    "supports it."
                )
    return linked, anomeric_bonded


def _classify_glycan_residue(
    ri: int,
    resname_of: list,
    residue_atom_idx: list,
    mol: "Molecule",
    linked: dict,
    anomeric_bonded: dict,
    info: dict,
) -> GlycanResidueInfo:
    """Turn one sugar's raw scan results into a :class:`GlycanResidueInfo`.

    Parameters
    ----------
    ri : int
        Residue index of the sugar being classified.
    resname_of : list
        Residue name for each residue index.
    residue_atom_idx : list
        Atom-index array for each residue index.
    mol : moleculekit.molecule.Molecule
        The molecule containing the candidate glycan(s).
    linked : dict
        Residue index -> set of ring positions with an outgoing linkage, as
        returned by :func:`_scan_glycan_bond_graph`.
    anomeric_bonded : dict
        Residue index -> list of ``(partner_residue, partner_atom)`` found
        on that residue's own anomeric carbon, as returned by
        :func:`_scan_glycan_bond_graph`.
    info : dict
        Residue index -> :class:`GlycanResidueInfo` for every sugar residue;
        used to tell a sugar-to-sugar bond apart from a sugar-to-anchor one.

    Returns
    -------
    gi : GlycanResidueInfo
        The classification result for residue ``ri``.

    Raises
    ------
    RuntimeError
        If ``ri`` has no ``O{p}`` atom for one of its own linked positions;
        if its anomeric carbon is bonded to more than one partner; if that
        partner is a protein residue GLYCAM does not support as a
        glycosylation anchor, or the wrong atom of a supported one; or if
        the partner is a carbohydrate absent from :data:`GLYCAM_SUGARS`.
    """
    gi = GlycanResidueInfo()
    resname = resname_of[ri]
    tmpl = GLYCAM_SUGARS[resname]
    names_here = {str(n) for n in mol.name[residue_atom_idx[ri]]}
    for position in linked[ri]:
        if f"O{position}" not in names_here:
            raise RuntimeError(
                f"Sugar {resname} (residue index {ri}) is linked at "
                f"position {position} but has no O{position} atom in the "
                "input structure."
            )
    gi.linked_positions = tuple(sorted(linked[ri]))

    partners = anomeric_bonded[ri]
    if len(partners) > 1:
        raise RuntimeError(
            f"Anomeric carbon {tmpl.anomeric_carbon} of sugar {resname} "
            f"(residue index {ri}) is bonded to {len(partners)} partners; "
            "a sugar's anomeric carbon can carry at most one glycosidic "
            "bond."
        )
    if len(partners) == 0:
        gi.free_reducing_end = True
        return gi

    pres, patom = partners[0]
    if pres in info:
        # Child of another sugar; the parent's own O{p} scan already
        # recorded the link on the parent's side.
        return gi
    presname = resname_of[pres]
    if presname in GLYCAN_ANCHORS:
        expected_atom = GLYCAN_ANCHORS[presname][1]
        if str(mol.name[patom]) != expected_atom:
            raise RuntimeError(
                f"Sugar {resname} (residue index {ri}) is bonded to "
                f"{presname} atom {mol.name[patom]}, but GLYCAM attaches "
                f"glycans to {presname} only via its {expected_atom} atom."
            )
        gi.anchor_res = pres
        gi.anchor_atom = expected_atom
    elif presname in PROTEIN_RESIDUE_NAMES_WITH_VARIANTS:
        raise RuntimeError(
            f"Sugar {resname} (residue index {ri}) is glycosidically "
            f"bonded to {presname} atom {mol.name[patom]}, but GLYCAM does "
            f"not support glycosylation of {presname} (supported anchor "
            f"residues: {sorted(GLYCAN_ANCHORS)})."
        )
    else:
        raise RuntimeError(
            f"Residue {presname} is bonded into a glycan (the anomeric "
            f"carbon of sugar {resname} links to its atom {mol.name[patom]}) "
            f"but is not in the supported GLYCAM sugar table "
            f"({sorted(GLYCAM_SUGARS)}). Add it to "
            "moleculekit.tools.glycans.GLYCAM_SUGARS if GLYCAM supports it."
        )
    return gi


def analyzeGlycanResidues(
    mol: "Molecule", bonds: np.ndarray, residue_atom_idx: list
) -> dict[int, GlycanResidueInfo]:
    """Recover glycan-tree structure from a molecule's bond graph.

    Walks every inter-residue bond once to work out, for each sugar residue
    recognized in :data:`GLYCAM_SUGARS`, which ring positions carry an
    outgoing glycosidic linkage to another sugar, and whether the sugar's
    own anomeric carbon is bonded onward to a parent sugar, to a protein
    anchor residue (see :data:`GLYCAN_ANCHORS`), or to nothing at all (a
    free reducing end). This is the bond-graph counterpart of
    :func:`glycanBondsFromNames`, meant to be called while the molecule
    still carries its original (e.g. CONECT-derived) bonds.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule containing the candidate glycan(s).
    bonds : np.ndarray
        ``(n, 2)`` array of atom index pairs, e.g. ``mol.bonds``.
    residue_atom_idx : list
        One atom-index array per residue, ordered so that the index of an
        entry in this list is the residue index used as a key in the
        returned dictionary (e.g. as built by
        :func:`moleculekit.util.sequenceID` or
        ``mol.getResidues(return_idx=True)``).

    Returns
    -------
    info : dict
        Maps residue index (into ``residue_atom_idx``) to a
        :class:`GlycanResidueInfo`, for every residue recognized as a sugar.
        Non-sugar residues are absent from the result.

    Raises
    ------
    RuntimeError
        If a sugar's anomeric carbon, or one of a sugar's numbered ring
        oxygens, is bonded to a residue that is itself a carbohydrate but
        absent from :data:`GLYCAM_SUGARS`; if a sugar's anomeric carbon is
        bonded to a protein residue GLYCAM does not support as a
        glycosylation anchor, or to the wrong atom of a supported one; or
        if a sugar's anomeric carbon is bonded to more than one partner.
    """
    bonds = np.asarray(bonds, dtype=np.int64)
    atom_to_res = np.full(mol.numAtoms, -1, dtype=np.int64)
    for ri, idx in enumerate(residue_atom_idx):
        atom_to_res[idx] = ri
    resname_of = [str(mol.resname[idx[0]]) for idx in residue_atom_idx]
    natoms_of = [len(idx) for idx in residue_atom_idx]
    info = {
        ri: GlycanResidueInfo()
        for ri, rn in enumerate(resname_of)
        if rn in GLYCAM_SUGARS
    }

    linked, anomeric_bonded = _scan_glycan_bond_graph(
        mol, bonds, atom_to_res, resname_of, natoms_of, info
    )
    for ri in info:
        info[ri] = _classify_glycan_residue(
            ri, resname_of, residue_atom_idx, mol, linked, anomeric_bonded, info
        )
    return info


def glycanBondsFromNames(mol: "Molecule") -> list:
    """Derive glycan connectivity from GLYCAM residue names and geometry.

    Meant for the htmd builder, after the molecule has been renamed to
    GLYCAM unit names (see :func:`glycamResname`) and had its glycosidic
    bonds removed (tleap infers that connectivity from residue templates
    instead), so this function purposefully consults only residue names and
    coordinates and never ``mol.bonds``. GLYCAM sugar units are recognized
    via :func:`glycamUnitMask`, which gates the resname match on the
    sugar's actual composition since several GLYCAM 3-character codes
    collide with unrelated real PDB Chemical Component Dictionary ligand
    codes. For every genuine GLYCAM sugar unit, each ring position encoded
    in its own name (see :func:`linkedPositionsFromGlycamResname`) is
    resolved to the nearest anomeric carbon of another residue within
    :data:`GLYCAN_LINK_CUTOFF`. Protein anchor residues
    (:data:`GLYCAM_ANCHOR_UNITS`) and the free-hydroxyl cap ``ROH`` are
    resolved the same way, connecting into a sugar's anomeric carbon.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        A molecule already renamed to GLYCAM-06 unit names, with its
        glycosidic bonds removed.

    Returns
    -------
    pairs : list
        List of ``(atom_index_1, atom_index_2)`` tuples, one per glycosidic
        or anchor bond that should be added back to the molecule.

    Raises
    ------
    RuntimeError
        If a GLYCAM unit's name encodes a linkage position whose ``O{p}``
        atom is absent from the residue, or if a linked position or anchor
        atom does not have exactly one anomeric-carbon partner within
        :data:`GLYCAN_LINK_CUTOFF`.
    """
    from moleculekit.util import sequenceID

    uq = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))
    unit_names = glycamUnitMask(mol)

    # Anomeric carbon of every GLYCAM sugar unit, looked up from its own
    # sugar letter (the middle character of the 3-character code) in
    # _LETTER_RING_ATOMS: C2 for sialic acid and fructose, C1 otherwise.
    anomeric = np.zeros(mol.numAtoms, dtype=bool)
    for rid in np.unique(uq[unit_names]):
        rmask = uq == rid
        code = str(mol.resname[rmask][0])
        cname = _LETTER_RING_ATOMS.get(code[1], ("C1", "O5"))[0]
        anomeric |= rmask & (mol.name == cname)
    ano_idx = np.where(anomeric)[0]
    ano_res = uq[ano_idx]
    ano_xyz = mol.coords[ano_idx, :, 0]

    def _partner(oidx: int, rid: int) -> int:
        d = np.linalg.norm(ano_xyz - mol.coords[oidx, :, 0], axis=1)
        near = np.where((d < GLYCAN_LINK_CUTOFF) & (ano_res != rid))[0]
        if len(near) != 1:
            raise RuntimeError(
                f"Expected exactly one anomeric partner within "
                f"{GLYCAN_LINK_CUTOFF} A of atom {mol.name[oidx]} of "
                f"residue {mol.resname[oidx]} {mol.resid[oidx]} (chain "
                f"{mol.chain[oidx]}), found {len(near)}."
            )
        return int(ano_idx[near[0]])

    pairs = []
    for rid in np.unique(uq[unit_names]):
        rmask = uq == rid
        code = str(mol.resname[rmask][0])
        for position in _LINKAGE_POSITIONS[code[0]]:
            omask = rmask & (mol.name == f"O{position}")
            if not omask.any():
                raise RuntimeError(
                    f"GLYCAM unit {code} (residue {mol.resid[rmask][0]} "
                    f"chain {mol.chain[rmask][0]}) is named as linked at "
                    f"position {position} but has no O{position} atom."
                )
            oidx = int(np.where(omask)[0][0])
            pairs.append((oidx, _partner(oidx, int(rid))))

    # Protein anchors and the free-hydroxyl cap bond INTO a sugar's own
    # anomeric carbon rather than into a numbered ring position.
    for resname, aname in {**GLYCAM_ANCHOR_UNITS, "ROH": "O1"}.items():
        for aidx in np.where((mol.resname == resname) & (mol.name == aname))[0]:
            pairs.append((int(aidx), _partner(int(aidx), int(uq[aidx]))))
    return pairs


def applyGlycamNaming(mol: "Molecule") -> "Molecule":
    """Rename every glycan in ``mol`` to GLYCAM-06 naming, in place.

    This is the force-field-naming counterpart of
    :func:`~moleculekit.tools.preparation.systemPrepare`'s glycan handling:
    ``systemPrepare`` protonates glycans correctly but leaves them under
    their original PDB Chemical Component Dictionary names (``NAG``,
    ``BMA``, ...) because most of those 3-character GLYCAM unit codes also
    collide with unrelated real PDB ligand codes (``TLA`` is tartrate) and
    so must not leak into a general-purpose, standalone-usable protonation
    tool. A builder that has already committed to the GLYCAM force field
    calls this function to apply that naming just before building.

    Three things happen, in this order (mirrors
    :func:`~moleculekit.tools.preparation._apply_glycan_modifications`,
    minus the anchor-hydrogen removal, which is a protonation consequence
    ``systemPrepare`` already applied and is not repeated here):

    1. Atom renames (:attr:`SugarTemplate.atom_renames`, the N-acetyl /
       N-glycolyl substituent atoms) followed by the residue renames
       themselves (the sugar to its GLYCAM unit name via
       :func:`glycamResname`, the anchor to its GLYCAM anchor resname via
       :data:`GLYCAN_ANCHORS`). Pure value changes, no reindexing.
    2. The free-reducing-end split, last, because it is the only step that
       inserts/removes atoms. A sugar with no outgoing glycosidic bond at
       its own anomeric carbon has its anomeric hydroxyl oxygen (and
       hydrogen, if present) split off into its own ``ROH`` residue,
       GLYCAM's free-hydroxyl cap, sharing the sugar's segment/chain with a
       fresh resid. ``systemPrepare`` already guarantees this atom pair is
       present (chemistry); a geometric placeholder is constructed here too
       for a molecule that reaches this function without having gone
       through ``systemPrepare`` first, with a warning naming the residue
       so the substitution is never silent.

    Connectivity is recovered from ``mol.bonds`` via
    :func:`analyzeGlycanResidues`, so it needs the glycosidic and anchor
    bonds still present (as ``systemPrepare`` output has, see
    ``moleculekit.tools.preparation._capture_bonds`` /
    ``_restore_bonds``); it does not need a
    :class:`~moleculekit.tools.nonstandard_residues.GlycanSpec` list.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to rename, still carrying the original PDB Chemical
        Component Dictionary sugar and anchor resnames, with its glycosidic
        and anchor bonds intact. Mutated in place.

    Returns
    -------
    mol : moleculekit.molecule.Molecule
        The same molecule, renamed in place, returned for chaining. A no-op
        (returned unchanged) when ``mol`` has no glycan.

    Raises
    ------
    RuntimeError
        If a recognized sugar residue is present but carries no bonds at
        all (the glycan tree structure -- linkage positions, anchor,
        free reducing end -- cannot be recovered without them).
    """
    from moleculekit.molecule import Molecule, UniqueResidueID

    if not pdbSugarMask(mol).any():
        return mol

    fields = ("resname", "resid", "insertion", "segid", "chain")
    _, idx_lists = mol.getResidues(fields=fields, return_idx=True)

    has_bond = np.zeros(mol.numAtoms, dtype=bool)
    bonds = np.asarray(mol.bonds, dtype=np.int64)
    if len(bonds):
        has_bond[bonds[:, 0]] = True
        has_bond[bonds[:, 1]] = True

    residues = [UniqueResidueID.fromMolecule(mol, idx=idx) for idx in idx_lists]
    for residue, atom_idx in zip(residues, idx_lists):
        if residue.resname in GLYCAM_SUGARS and not has_bond[atom_idx].any():
            raise RuntimeError(
                f"Sugar residue {residue.resname} {residue.chain}:"
                f"{residue.resid}{residue.insertion} (segid {residue.segid}) "
                "carries no bonds, so its glycan tree structure (linkage "
                "positions, anchor, free reducing end) cannot be recovered. "
                "Ensure the molecule's glycosidic bonds are present (e.g. "
                "from CONECT records or systemPrepare output) before "
                "calling this function."
            )

    glycan_info = analyzeGlycanResidues(mol, mol.bonds, idx_lists)
    if not glycan_info:
        return mol

    def _res_mask(rid: UniqueResidueID) -> np.ndarray:
        return (
            (mol.segid == str(rid.segid))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
        )

    # 1. Atom renames, then residue renames (sugar + its anchor). Pure
    # value changes: no reindexing, so masks stay valid within this loop.
    for ri, gi in glycan_info.items():
        residue = residues[ri]
        tmpl = GLYCAM_SUGARS[residue.resname]
        smask = _res_mask(residue)
        for old, new in tmpl.atom_renames.items():
            mol.name[smask & (mol.name == old)] = new
        mol.resname[smask] = glycamResname(residue.resname, gi.linked_positions)
        if gi.anchor_res is not None:
            anchor_residue = residues[gi.anchor_res]
            anchor_new_resname = GLYCAN_ANCHORS[anchor_residue.resname][0]
            mol.resname[_res_mask(anchor_residue)] = anchor_new_resname

    # 2. Free reducing-end ROH split, last (inserts/removes atoms).
    for ri, gi in glycan_info.items():
        if not gi.free_reducing_end:
            continue
        residue = residues[ri]
        tmpl = GLYCAM_SUGARS[residue.resname]
        smask = _res_mask(residue)
        seg = str(mol.segid[smask][0])
        chain = str(mol.chain[smask][0])
        new_resid = int(mol.resid[mol.segid == seg].max()) + 1

        o1_idx = np.where(smask & (mol.name == tmpl.anomeric_oxygen))[0]
        if len(o1_idx):
            # Already resolved (systemPrepare guarantees this for its own
            # output): carry the real atom(s) over rather than discarding
            # deposited coordinates. A companion hydrogen is found by
            # proximity (a real O-H bond is always well under 1.2 A;
            # nothing else in a sugar sits that close to the anomeric
            # oxygen).
            o1_idx = int(o1_idx[0])
            hcand = np.where(smask & (mol.element == "H"))[0]
            sel = [o1_idx]
            if len(hcand):
                d = np.linalg.norm(
                    mol.coords[hcand, :, 0] - mol.coords[o1_idx, :, 0], axis=1
                )
                near = hcand[d < 1.2]
                if len(near):
                    sel.append(int(near[0]))
            o1sel = np.zeros(mol.numAtoms, dtype=bool)
            o1sel[sel] = True
            rohmol = mol.copy(sel=o1sel)
            # GLYCAM's ROH cap always names its own atom O1, regardless of
            # which position the source sugar's anomeric oxygen occupied
            # (O2 for a sialic acid, O1 for everything else).
            rohmol.name[rohmol.element == "O"] = "O1"
            rohmol.name[rohmol.element == "H"] = "HO1"
            mol.remove(o1sel, _logger=False)
        else:
            # A molecule that reaches this function without having gone
            # through systemPrepare first can still be missing its anomeric
            # oxygen entirely. Build a rough position (1.43 A from the
            # anomeric carbon, opposite its heavy-atom centroid) for a
            # downstream minimization to relax, and warn so the
            # substitution is never silent.
            logger.warning(
                f"Glycan sugar {residue.resname} (renamed to "
                f"{glycamResname(residue.resname, gi.linked_positions)}) "
                f"{residue.chain}:{residue.resid}{residue.insertion} (segid "
                f"{residue.segid}) has no anomeric oxygen; constructing its "
                "position geometrically instead of using a deposited one."
            )
            c1_idx = int(np.where(smask & (mol.name == tmpl.anomeric_carbon))[0][0])
            c1_xyz = mol.coords[c1_idx, :, 0]
            neigh = [
                int(n) for n in mol.getNeighbors(c1_idx) if str(mol.element[n]) != "H"
            ]
            if neigh:
                direction = c1_xyz - mol.coords[neigh, :, 0].mean(axis=0)
            else:
                direction = np.array([1.0, 0.0, 0.0])
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

            rohmol = Molecule().empty(1)
            rohmol.record[:] = "HETATM"
            rohmol.name[:] = "O1"
            rohmol.element[:] = "O"
            rohmol.coords = (
                (c1_xyz + direction * 1.43)
                .reshape(1, 3, 1)
                .astype(Molecule._dtypes["coords"])
            )

        rohmol.resname[:] = "ROH"
        rohmol.resid[:] = new_resid
        rohmol.insertion[:] = ""
        rohmol.chain[:] = chain
        rohmol.segid[:] = seg

        insert_at = int(np.where(_res_mask(residue))[0][0])
        mol.insert(rohmol, insert_at)

    return mol
