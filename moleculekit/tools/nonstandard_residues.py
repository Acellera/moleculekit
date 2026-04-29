"""Discovery helper for non-standard residues that need user-driven AMBER
parameterization before building.

A "non-standard residue" is any residue whose ``resname`` is not in
moleculekit's canonical amino-acid / nucleic / water / ion sets.
:func:`detectNonStandardResidues` walks the molecule, mutates ``mol`` to
rename / re-protonate any canonical amino-acid residue that is
covalently bonded to a non-canonical residue, and returns one spec per
non-standard residue (plus one per renamed canonical residue):

  - :class:`NCAASpec` - chain-resident NCAA with no sidechain crosslink
    (e.g. selenomethionine, norleucine, a D-amino acid).
  - :class:`CrosslinkedNCAASpec` - chain-resident NCAA whose sidechain
    *is* covalently bonded to one or more other residues (a stapled-
    peptide residue, a glycosylated NCAA).
  - :class:`ScaffoldSpec` - free non-canonical residue with two or more
    non-peptide bonds to other residues (the central scaffold of a
    bicyclic peptide, a multi-anchor covalent inhibitor).
  - :class:`CovalentLigandSpec` - free non-canonical residue with exactly
    one non-peptide bond to another residue (single-anchor covalent
    inhibitor, NAG-Asn glycan stem, single-Cys heme).
  - :class:`LigandSpec` - free non-canonical residue with no covalent
    bonds (small-molecule binding-pocket ligand, fatty acid).
  - :class:`CanonicalRenamedSpec` - one per canonical residue the
    detector renamed (e.g. CYS bonded to a scaffold becomes ``CY1``).

The actual parameterization (running antechamber, splitting per-residue,
emitting custombonds for ``amber.build``) lives in
``htmd.builder.scaffolded_peptide.parameterizeFromSpecs``, which consumes
this list together with the mutated ``mol``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Union

import numpy as np

from moleculekit.molecule import UniqueResidueID
from moleculekit.residues import (
    PROTEIN_RESIDUE_NAMES,
    NUCLEIC_RESIDUE_NAMES,
    MODIFIED_PROTEIN_RESIDUE_NAMES,
    PROTEIN_RESIDUES,
    NUCLEIC_RESIDUES,
    WATER_RESIDUE_NAMES,
)
from moleculekit.tools._anchor_variants import lookup_anchor_variant
from moleculekit import __share_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-residue spec dataclasses.
#
# Residue identities reuse :class:`UniqueResidueID` from
# :mod:`moleculekit.molecule` so callers can round-trip them against any
# in-memory ``Molecule`` (via ``selectAtoms``).
# ---------------------------------------------------------------------------


@dataclass
class NCAASpec:
    """A non-canonical amino acid embedded in a polypeptide chain via
    standard peptide (N-C) bonds to canonical amino acids, with no other
    inter-residue covalent bonds. Examples: selenomethionine (MSE),
    norleucine (NLE), D-amino acids, backbone-modified residues. The
    parameterizer treats it as a free residue with ACE/NME caps."""

    resname: str
    residue: UniqueResidueID
    is_n_term: bool
    is_c_term: bool


@dataclass
class CrosslinkedNCAASpec:
    """A non-canonical amino acid embedded in a polypeptide chain (peptide-
    bonded into the backbone) that *also* has one or more non-peptide
    sidechain bonds to other residues - a stapled-peptide residue, an NCAA
    whose sidechain is glycosylated, etc. The parameterizer combines this
    residue with its crosslink partners in a single antechamber compute."""

    resname: str
    residue: UniqueResidueID
    is_n_term: bool
    is_c_term: bool


@dataclass
class ScaffoldSpec:
    """A non-canonical residue that is not peptide-bonded into a chain and
    has two or more non-peptide bonds going out to other residues. Examples:
    the central scaffold of a bicyclic / tricyclic peptide, a multi-anchor
    covalent inhibitor."""

    resname: str
    residue: UniqueResidueID


@dataclass
class CovalentLigandSpec:
    """A non-canonical residue that is not peptide-bonded into a chain and
    has exactly one non-peptide bond going out to another residue. Examples:
    a single-anchor covalent inhibitor, a NAG-Asn glycan stem, a single-Cys
    heme."""

    resname: str
    residue: UniqueResidueID


@dataclass
class LigandSpec:
    """A non-canonical residue with no covalent bonds to any other residue
    (a free, non-covalently bound ligand). Examples: small-molecule drug
    ligands in binding pockets, fatty acids, lipid head-groups. The
    parameterizer treats it standalone with no caps."""

    resname: str
    residue: UniqueResidueID


@dataclass
class CanonicalRenamedSpec:
    """A canonical amino-acid residue that the detector renamed because at
    least one of its sidechain heavy atoms is covalently bonded to a non-
    canonical residue. The detector mutates ``mol`` so this residue's
    resname is already ``new_resname`` and the displaced hydrogens (per
    :data:`ANCHOR_VARIANTS`) have been removed - i.e.
    ``residue.resname == new_resname`` always holds."""

    residue: UniqueResidueID
    original_resname: str
    new_resname: str


PerResidueSpec = Union[
    NCAASpec,
    CrosslinkedNCAASpec,
    ScaffoldSpec,
    CovalentLigandSpec,
    LigandSpec,
    CanonicalRenamedSpec,
]


# ---------------------------------------------------------------------------
# Canonical-resname sets and small helpers.
# ---------------------------------------------------------------------------


with open(os.path.join(__share_dir, "atomselect", "atomselect.json")) as _f:
    _ION_RESNAMES = set(json.load(_f).get("ion_resnames", []))


# Standard peptide-terminus caps. AMBER ff14SB / ff19SB ship parameters for
# these so they don't need user-driven parameterization.
_CAP_RESNAMES = {"ACE", "NME", "NHE", "NH2"}


def _canonical_resnames():
    names = set(PROTEIN_RESIDUE_NAMES)
    names |= set(NUCLEIC_RESIDUE_NAMES)
    names |= set(MODIFIED_PROTEIN_RESIDUE_NAMES)
    for rr in PROTEIN_RESIDUES + NUCLEIC_RESIDUES:
        names.update(rr.resname_variants)
    names |= WATER_RESIDUE_NAMES
    names |= _ION_RESNAMES
    names |= _CAP_RESNAMES
    return names


_CANONICAL_RESNAMES = _canonical_resnames()
# Canonical-AA resnames (including ff14SB variants like CYX, HID/HIE/HIP, LYN).
# Used as a chain-residency hint: CIF / PDB inputs frequently carry only the
# special inter-residue bonds (via ``_struct_conn`` or ``CONECT``) and omit
# canonical peptide bonds, so we can't rely on explicit N-C bonds alone to
# flag chain residency for canonical amino acids.
_PROTEIN_RESNAMES = set(PROTEIN_RESIDUE_NAMES)
for _rr in PROTEIN_RESIDUES:
    _PROTEIN_RESNAMES.update(_rr.resname_variants)


_RESIDUE_FIELDS = ("resname", "resid", "insertion", "segid", "chain")


def _residue_groups(mol):
    """Return ``(a2r, groups)`` where ``a2r`` maps each atom index to a unique
    residue id and ``groups`` is a list of per-residue dicts ordered by
    residue index."""
    a2r, idx_lists = mol.getResidues(fields=_RESIDUE_FIELDS, return_idx=True)
    groups = []
    for atom_idx in idx_lists:
        first = int(atom_idx[0])
        groups.append(
            {
                "atom_idx": np.asarray(atom_idx, dtype=np.int64),
                "resname": str(mol.resname[first]),
                "resid": int(mol.resid[first]),
                "insertion": str(mol.insertion[first]),
                "segid": str(mol.segid[first]),
                "chain": str(mol.chain[first]),
            }
        )
    return a2r, groups


def _ensure_bonds(mol):
    """Return ``mol.bonds`` if populated, otherwise fall back to distance-
    based bond guessing. Does not mutate ``mol``."""
    if len(mol.bonds):
        return mol.bonds
    logger.warning(
        "Molecule has no bonds; falling back to distance-based bond guessing "
        "for non-standard residue detection."
    )
    bonds = mol._guessBonds(rdkit=False)
    return np.asarray(bonds, dtype=np.int64)


def _residue_id(g):
    return UniqueResidueID(
        resname=g["resname"],
        chain=g["chain"],
        resid=g["resid"],
        insertion=g["insertion"],
        segid=g["segid"],
    )


def _pick_bucket_resname(variant_or_resname, prefix_counter):
    """Build a 3-char custom resname so tLeap loads our antechamber-derived
    prepi for this bucket instead of falling back to the built-in
    ff14SB / GLYCAM template. Format: 2-char variant prefix + 1-char
    counter (``CY1``, ``CY2``, ``NL1``, ...). ``prefix_counter`` is a
    shared dict mapping each prefix to its next counter value across the
    whole detect call. PDB2PQR (used by ``systemPrepare``) truncates
    anything longer than 3 chars, so we cap the name at 3."""
    prefix = (variant_or_resname or "X")[:2]
    n = prefix_counter.get(prefix, 0) + 1
    prefix_counter[prefix] = n
    if n < 10:
        digit = str(n)
    elif n < 36:
        digit = chr(ord("a") + n - 10)
    else:
        raise RuntimeError(
            f"Too many distinct anchor buckets sharing prefix {prefix!r}; "
            f"unable to fit a unique 3-char resname."
        )
    return f"{prefix}{digit}"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def detectNonStandardResidues(mol):
    """Walk ``mol`` and return one spec per non-standard residue.

    Mutates ``mol`` in place: every canonical amino-acid residue whose
    sidechain is covalently bonded to a non-canonical residue is renamed
    to a custom 3-char resname (e.g. ``CY1`` for the CYS-SG-bonded-to-
    LFI bucket, ``NL1`` for ASN-ND2-bonded-to-NAG, ...) and the
    sidechain hydrogens listed in :data:`ANCHOR_VARIANTS` (``HG`` for
    Cys SG, ``HD22`` for Asn ND2, ...) are removed. The custom resname
    keeps the residue out of tLeap's built-in libraries so the
    parameterizer's antechamber-derived prepi is the one that loads.

    Residues that share the same ``(canonical_resname, anchor_atom_name,
    partner_resname)`` key are assigned the *same* resname so the
    parameterizer emits one prepi shared across them (e.g. all
    ASN-ND2-bonded-to-NAG residues become ``NL1``). Residues whose
    anchor has no entry in :data:`ANCHOR_VARIANTS` (e.g. SER OG) use
    their original resname's first two chars as the prefix.

    To preserve the input molecule, call ``mol.copy()`` first.

    Parameters
    ----------
    mol : :class:`moleculekit.molecule.Molecule`
        Input molecule. Should already carry covalent bonds (read from a
        PDB ``CONECT`` block or a CIF ``_struct_conn`` block, or set up
        via ``mol.templateResidueFromSmiles``); if ``mol.bonds`` is empty,
        the detector falls back to distance-based bond guessing via
        ``mol._guessBonds()``.

    Returns
    -------
    list[PerResidueSpec]
        A flat list mixing :class:`NCAASpec`,
        :class:`CrosslinkedNCAASpec`, :class:`ScaffoldSpec`,
        :class:`CovalentLigandSpec`, :class:`LigandSpec`, and
        :class:`CanonicalRenamedSpec` entries.
    """
    bonds = _ensure_bonds(mol)
    a2r, groups = _residue_groups(mol)
    n_res = len(groups)

    # Walk every inter-residue bond once. For each non-canonical residue
    # we track its non-peptide partners (with the atom name on this side);
    # for every residue we track whether it has a peptide bond on its N
    # side and / or its C side, which feeds the chain-residency check and
    # the is_n_term / is_c_term flags on (Crosslinked)NCAASpec.
    nonpep_partners = [[] for _ in range(n_res)]  # res -> [(other_res, this_atom_name)]
    has_peptide_bond = [False] * n_res
    peptide_attached_n = [False] * n_res
    peptide_attached_c = [False] * n_res

    for a1, a2 in bonds:
        a1, a2 = int(a1), int(a2)
        r1, r2 = int(a2r[a1]), int(a2r[a2])
        if r1 == r2 or r1 < 0 or r2 < 0:
            continue
        n1, n2 = str(mol.name[a1]), str(mol.name[a2])
        if {n1, n2} == {"N", "C"}:
            has_peptide_bond[r1] = True
            has_peptide_bond[r2] = True
            if n1 == "N":
                peptide_attached_n[r1] = True
                peptide_attached_c[r2] = True
            else:
                peptide_attached_c[r1] = True
                peptide_attached_n[r2] = True
        else:
            nonpep_partners[r1].append((r2, n1))
            nonpep_partners[r2].append((r1, n2))

    chain_resident = [
        groups[r]["resname"] in _PROTEIN_RESNAMES or has_peptide_bond[r]
        for r in range(n_res)
    ]

    # Plan the canonical renames + displaced-H drops without applying them
    # yet. Bucket key is (canonical_resname, anchor_atom_name,
    # partner_resname); residues sharing a bucket get the same 3-char
    # custom resname so the parameterizer emits one shared prepi.
    canonical_renames = {}  # residue_idx -> (original_resname, new_resname)
    drop_atom_idxs = set()
    bucket_to_resname = {}
    prefix_counter = {}
    for r_idx in range(n_res):
        partners = nonpep_partners[r_idx]
        if not partners:
            continue
        g = groups[r_idx]
        if g["resname"] not in _CANONICAL_RESNAMES:
            continue
        # Pick a non-canonical partner. Canonical-canonical non-peptide
        # bonds (e.g. disulfides) are handled by the existing amber.build
        # disulfide path and don't need a rename here.
        non_canonical = next(
            (
                (other_r, anchor_atom)
                for other_r, anchor_atom in partners
                if groups[other_r]["resname"] not in _CANONICAL_RESNAMES
            ),
            None,
        )
        if non_canonical is None:
            continue
        other_r, anchor_atom = non_canonical
        partner_resname = groups[other_r]["resname"]
        entry = lookup_anchor_variant(g["resname"], anchor_atom)

        bucket_key = (g["resname"], anchor_atom, partner_resname)
        if bucket_key in bucket_to_resname:
            new_resname = bucket_to_resname[bucket_key]
        else:
            base = (
                entry["variant"] if entry and entry["variant"] else g["resname"]
            )
            new_resname = _pick_bucket_resname(base, prefix_counter)
            bucket_to_resname[bucket_key] = new_resname

        canonical_renames[r_idx] = (g["resname"], new_resname)
        if entry is not None:
            for h_name in entry["drop_h"]:
                for ai in g["atom_idx"]:
                    if str(mol.name[int(ai)]) == h_name:
                        drop_atom_idxs.add(int(ai))
                        break

    # Build specs from the pre-mutation residue groups so we still know
    # the original resnames for non-canonical residues. For canonical
    # anchors, swap in the new resname when constructing UniqueResidueID.
    specs = []
    for r_idx, g in enumerate(groups):
        if g["resname"] in _CANONICAL_RESNAMES:
            continue
        residue_id = _residue_id(g)
        nonpep_count = len(nonpep_partners[r_idx])
        if chain_resident[r_idx]:
            is_n_term = not peptide_attached_n[r_idx]
            is_c_term = not peptide_attached_c[r_idx]
            if nonpep_count > 0:
                specs.append(
                    CrosslinkedNCAASpec(
                        resname=g["resname"],
                        residue=residue_id,
                        is_n_term=is_n_term,
                        is_c_term=is_c_term,
                    )
                )
            else:
                specs.append(
                    NCAASpec(
                        resname=g["resname"],
                        residue=residue_id,
                        is_n_term=is_n_term,
                        is_c_term=is_c_term,
                    )
                )
        else:
            if nonpep_count >= 2:
                specs.append(
                    ScaffoldSpec(resname=g["resname"], residue=residue_id)
                )
            elif nonpep_count == 1:
                specs.append(
                    CovalentLigandSpec(resname=g["resname"], residue=residue_id)
                )
            else:
                specs.append(
                    LigandSpec(resname=g["resname"], residue=residue_id)
                )

    # One CanonicalRenamedSpec per renamed canonical residue, in residue-
    # index order so the output is deterministic.
    for r_idx in sorted(canonical_renames):
        original_resname, new_resname = canonical_renames[r_idx]
        g = groups[r_idx]
        renamed_residue_id = UniqueResidueID(
            resname=new_resname,
            chain=g["chain"],
            resid=g["resid"],
            insertion=g["insertion"],
            segid=g["segid"],
        )
        specs.append(
            CanonicalRenamedSpec(
                residue=renamed_residue_id,
                original_resname=original_resname,
                new_resname=new_resname,
            )
        )

    # Apply the planned mutations to mol. Renames are safe at any time
    # (no atom-count change); H drops change mol.numAtoms, so go last.
    for r_idx, (_, new_resname) in canonical_renames.items():
        mol.resname[groups[r_idx]["atom_idx"]] = new_resname
    if drop_atom_idxs:
        drop_mask = np.zeros(mol.numAtoms, dtype=bool)
        drop_mask[list(drop_atom_idxs)] = True
        mol.remove(drop_mask, _logger=False)

    return specs
