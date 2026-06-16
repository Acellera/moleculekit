"""Discovery helper for non-standard residues that need user-driven AMBER
parameterization before building.

A "non-standard residue" is any residue whose ``resname`` is not in
moleculekit's canonical amino-acid / nucleic / water / ion sets.
:func:`detectNonStandardResidues` inspects the molecule (without
mutating it) and returns one spec per non-standard residue, plus one
per canonical residue covalently bonded to a non-canonical one:

  - :class:`ChainResidueSpec` - one spec per chain-resident residue that
    needs special handling during AMBER parameterization: a non-canonical
    amino acid embedded in a polypeptide chain (selenomethionine,
    norleucine, a stapled NCAA, etc.) OR a canonical amino acid whose
    sidechain is covalently bonded to something other than its peptide
    neighbours (a Cys forming a thioether or disulfide, an Asn
    N-glycosylated by a sugar, a Glu CD - Lys NZ isopeptide).
  - :class:`ScaffoldSpec` - free non-canonical residue with two or more
    non-peptide bonds to other residues (the central scaffold of a
    bicyclic peptide, a multi-anchor covalent inhibitor).
  - :class:`CovalentLigandSpec` - free non-canonical residue with exactly
    one non-peptide bond to another residue (single-anchor covalent
    inhibitor, NAG-Asn glycan stem, single-Cys heme).
  - :class:`LigandSpec` - free non-canonical residue with no covalent
    bonds (small-molecule binding-pocket ligand, fatty acid).

Pass the spec list to :func:`moleculekit.tools.preparation.systemPrepare`
via ``detect_specs=specs`` to apply the proposed renames + H-drops on
the prepared molecule.
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
from moleculekit.tools._anchor_variants import lookup_anchor
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
class ChainResidueSpec:
    """One spec per chain-resident residue that needs special handling
    during AMBER parameterization: a non-canonical amino acid embedded
    in a polypeptide chain (selenomethionine, norleucine, a stapled
    NCAA, etc.) OR a canonical amino acid whose sidechain is covalently
    bonded to something other than its peptide neighbours (a Cys
    forming a thioether or disulfide, an Asn N-glycosylated by a sugar,
    a Glu CD - Lys NZ isopeptide).

    Fields:

    - ``resname``: the residue's resname in the input
      ``Molecule`` (``"GLU"``, ``"NLE"``, ``"CYS"``, ...).
    - ``residue``: :class:`UniqueResidueID` for the residue (segid /
      chain / resid / insertion).
    - ``new_resname``: the resname to rename to before downstream
      parameterization. Set whenever a rename is needed:

        * Canonical AA at a junction: ``"CYX"`` for both ends of a
          CYS-SG <-> CYS-SG disulfide; an auto-generated 3-char ``XX#``
          name otherwise, shared across residues sharing the bucket
          ``(resname, anchor_atom, partner_resname, n_term,
          c_term)`` so antechamber runs once per unique chemistry.
        * NCAA appearing with multiple terminus configurations: the
          existing :func:`_disambiguate_terminus_resnames` prefixes
          ``"N"``/``"C"``/``"B"`` so each terminus form gets its own
          prepi (otherwise tLeap's second ``loadAmberPrep`` would
          clobber the first).
        * ``None`` when no rename is needed (plain mid-chain NCAA,
          single-terminus-config NCAA, etc.).
    - ``anchor_atom``: the name of the residue's sidechain atom that
      participates in a non-peptide inter-residue bond (``"SG"`` for a
      Cys-thioether, ``"CD"`` for a Glu CD-LYS NZ isopeptide, ``"NZ"``
      for the Lys end of the same isopeptide, ``"CE"`` for an NLE
      staple, ...). ``None`` when the residue has no non-peptide bond
      (plain chain-resident NCAA); ``anchor_atom is not None`` is the
      single source of truth for "is this residue at a non-peptide
      junction?" - what the old ``CrosslinkedNCAASpec``/``NCAASpec``
      distinction encoded. For residues with multiple non-peptide
      partners the detector picks the deterministically-first partner
      (sorted by partner residue index, then anchor atom name). For
      canonical-AA renamed entries this is also the partner used as the
      bucket key; for NCAA entries (where there's no bucket key) the
      same deterministic order applies.
    - ``is_n_term`` / ``is_c_term``: chain termini flags.
    """

    resname: str
    residue: UniqueResidueID
    new_resname: str | None = None
    anchor_atom: str | None = None
    is_n_term: bool = False
    is_c_term: bool = False


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



PerResidueSpec = Union[
    ChainResidueSpec,
    ScaffoldSpec,
    CovalentLigandSpec,
    LigandSpec,
]


# ---------------------------------------------------------------------------
# Canonical-resname sets and small helpers.
# ---------------------------------------------------------------------------


with open(os.path.join(__share_dir, "atomselect", "atomselect.json")) as _f:
    _ION_RESNAMES = set(json.load(_f).get("ion_resnames", []))


# Standard peptide-terminus caps. AMBER ff14SB / ff19SB ship parameters for
# these so they don't need user-driven parameterization.
_CAP_RESNAMES = {"ACE", "NME", "NHE", "NH2"}


# Backbone atoms of a canonical amino acid that never form a real
# inter-residue crosslink. A non-peptide bond landing on one of these is
# treated as a guessed-bond artifact (a spurious close contact from
# slightly-off coordinates) and ignored, but only when the bonds were
# guessed: explicit input bonds (CONECT / _struct_conn) are always trusted.
# ``C`` and ``N`` are intentionally excluded: a backbone C forms real
# isopeptide bonds (e.g. ubiquitin's C-terminal Gly to a target Lys NZ) and
# a backbone N can be acylated or cyclized.
_BACKBONE_NONANCHOR_ATOMS = {"O", "CA"}


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
PROTEIN_RESNAMES = set(PROTEIN_RESIDUE_NAMES)
for _rr in PROTEIN_RESIDUES:
    PROTEIN_RESNAMES.update(_rr.resname_variants)


def _residue_groups(mol):
    """Return ``(a2r, residues, atom_idxs)`` where ``a2r`` maps each atom
    index to a unique residue id, ``residues`` is a list of
    :class:`UniqueResidueID` ordered by residue index, and ``atom_idxs``
    holds the atom indices for each residue."""
    fields = ("resname", "resid", "insertion", "segid", "chain")
    a2r, idx_lists = mol.getResidues(fields=fields, return_idx=True)
    residues, atom_idxs = [], []
    for atom_idx in idx_lists:
        first = int(atom_idx[0])
        residues.append(
            UniqueResidueID(
                resname=str(mol.resname[first]),
                chain=str(mol.chain[first]),
                resid=int(mol.resid[first]),
                insertion=str(mol.insertion[first]),
                segid=str(mol.segid[first]),
            )
        )
        atom_idxs.append(np.asarray(atom_idx, dtype=np.int64))
    return a2r, residues, atom_idxs


def _ensure_bonds(mol, guess=True):
    """Return ``mol.bonds`` if populated. Otherwise fall back to distance-
    based bond guessing when ``guess`` is True, or an empty bond array when
    it is False. Does not mutate ``mol``."""
    if len(mol.bonds):
        return mol.bonds
    if not guess:
        return np.zeros((0, 2), dtype=np.int64)
    logger.warning(
        "Molecule has no bonds; falling back to distance-based bond guessing "
        "for non-standard residue detection. Pass guess_bonds=False to skip "
        "this and rely only on explicit input bonds."
    )
    bonds = mol._guessBonds(rdkit=False)
    return np.asarray(bonds, dtype=np.int64)


def _has_peptide_neighbour(mol, atom_idx, side):
    """Return True if this residue has a peptide-bond neighbour on the
    given side (``"N"`` = previous residue's C atom; ``"C"`` = next
    residue's N atom). Falls back to a distance check (peptide N-C is
    ~1.32 A; we accept anything under 1.6 A) so sparse CIF inputs that
    only carry the special inter-residue bonds still get correct
    chain-terminus flags for canonical amino acids."""
    other_name = "C" if side == "N" else "N"
    own_atoms = set(int(a) for a in atom_idx)
    self_atom_idxs = [int(a) for a in atom_idx if str(mol.name[int(a)]) == side]
    if not self_atom_idxs:
        return False
    other_mask = mol.name == other_name
    other_mask[list(own_atoms)] = False
    other_coords = mol.coords[other_mask, :, mol.frame]
    for self_atom in self_atom_idxs:
        for nb in mol.getNeighbors(self_atom):
            if int(nb) in own_atoms:
                continue
            if str(mol.name[int(nb)]) == other_name:
                return True
        if other_coords.size:
            self_pos = mol.coords[self_atom, :, mol.frame]
            if np.linalg.norm(other_coords - self_pos, axis=1).min() < 1.6:
                return True
    return False


_BUCKET_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


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
    if n >= len(_BUCKET_DIGITS):
        raise RuntimeError(
            f"Too many distinct anchor buckets sharing prefix {prefix!r}; "
            f"unable to fit a unique 3-char resname."
        )
    return f"{prefix}{_BUCKET_DIGITS[n]}"


# Terminus disambiguation.


def _disambiguate_terminus_resnames(specs):
    """Mutate :class:`ChainResidueSpec` entries for NCAAs in-place,
    setting ``new_resname`` when more than one ``(is_n_term, is_c_term)``
    combination is present for the same input resname. The prepi unit
    name is the residue resname, so two specs sharing a resname but
    differing in terminus produce two prepis whose second
    ``loadAmberPrep`` clobbers the first in tLeap. Mid-chain forms
    keep ``new_resname=None``; N-terminal -> ``"N"+resname``; C-terminal
    -> ``"C"+resname``; both-terminus -> ``"B"+resname``.

    Raise ``RuntimeError`` if disambiguation is required for an input
    resname already 4+ characters long, since prefixing would exceed the
    4-character AMBER prepi unit-name limit."""
    by_resname = {}
    for spec in specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        # Disambiguation only applies to NCAA-style entries (those whose
        # resname is NOT a canonical AA - canonical anchors are
        # already bucketed by terminus via the (n_term, c_term) bucket key).
        if spec.resname in PROTEIN_RESNAMES:
            continue
        by_resname.setdefault(spec.resname, []).append(spec)

    for resname, group in by_resname.items():
        configs = {(s.is_n_term, s.is_c_term) for s in group}
        if len(configs) < 2:
            continue
        if len(resname) >= 4:
            raise RuntimeError(
                f"Residue {resname!r} appears with more than one terminus "
                f"configuration ({sorted(configs)}) and needs a disambiguating "
                f"prefix, but the input resname is already {len(resname)} "
                f"characters long. Prefixing would exceed the 4-character "
                f"AMBER prepi unit-name limit. Rename the residue to a "
                f"3-character form before running detectNonStandardResidues."
            )
        for spec in group:
            n, c = spec.is_n_term, spec.is_c_term
            if n and c:
                spec.new_resname = f"B{resname}"
            elif n:
                spec.new_resname = f"N{resname}"
            elif c:
                spec.new_resname = f"C{resname}"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def detectNonStandardResidues(mol, guess_bonds=True):
    """Walk ``mol`` and emit one spec per residue that needs special
    handling by a downstream parameterizer / builder.

    Inspects ``mol.bonds`` (without mutating the molecule) and classifies
    every non-canonical residue plus every canonical residue at a
    non-peptide junction into one of four spec types:

    - :class:`ChainResidueSpec` — chain-resident residue that needs
      special parameterization: a non-canonical amino acid embedded in a
      polypeptide chain (selenomethionine, norleucine, stapled-peptide
      residues, ...) OR a canonical AA whose sidechain is covalently
      bonded to anything other than its peptide neighbours (Cys-Cys
      disulfide, Cys thioether to a heme, Asn N-glycan, Glu-Lys
      isopeptide, Tyr coordinating a metal, ...). Canonical AAs at a
      junction always receive ``new_resname``: ``"CYX"`` for both ends
      of a CYS-SG <-> CYS-SG disulfide; an auto-generated 3-char
      ``XX#`` name otherwise. Residues that share the same
      ``(canonical_resname, anchor_atom, partner_resname, is_n_term,
      is_c_term)`` bucket key collapse to the *same* ``XX#`` so the
      parameterizer emits one prepi shared across them (e.g. all
      mid-chain ASN-ND2-bonded-to-NAG residues land on one bucket).
      Chain-terminal forms get their own buckets because they carry
      extra atoms (``OXT`` on the C-terminus, ``H1/H2/H3`` on the
      N-terminus) and different charges.
    - :class:`ScaffoldSpec` — non-chain-resident residue with 2+
      non-peptide bonds (bicyclic-peptide central scaffold,
      multi-anchor covalent inhibitor).
    - :class:`CovalentLigandSpec` — non-chain-resident residue with
      exactly one non-peptide bond (single-anchor covalent inhibitor,
      NAG-Asn glycan stem, single-Cys heme).
    - :class:`LigandSpec` — non-chain-resident residue with no covalent
      bonds (free small-molecule ligand, fatty acid).

    Chain-resident NCAAs that appear with more than one terminus
    configuration in the same molecule are disambiguated post hoc by
    setting ``new_resname`` on the terminal specs (``"N"+resname`` for
    N-term, ``"C"+resname`` for C-term, ``"B"+resname`` for a single-
    residue chain). When every instance of an NCAA shares the same
    terminus configuration, ``new_resname`` stays ``None``.

    Metal-coordination contacts where the metal is a standalone ion
    residue (e.g. PDB ``LINK`` records between a Zn²⁺ residue and a
    Zn-chelating inhibitor, or 3PTB's Ca²⁺ coordinated by GLU/ASN/VAL
    oxygens) are skipped — the inhibitor stays a free :class:`LigandSpec`,
    and the protein residues are left alone. Coordinations where the
    metal lives *inside* a cofactor (e.g. Fe inside HEM coordinated to a
    Tyr-OH or Cys-SG) are **kept**: the cofactor becomes a
    :class:`CovalentLigandSpec` and the donating canonical AA becomes a
    :class:`ChainResidueSpec`, because the donor's protonation state
    changes (Tyr-O⁻, Cys-S⁻) and needs a custom prepi. Bonds touching
    water are always skipped.

    .. note::

        Plain Cys–Cys disulfides are **not** returned as separate specs for
        the caller to process. Both Cys residues are instead silently renamed
        to ``CYX`` inside :func:`moleculekit.tools.preparation.systemPrepare`
        (which calls this function internally). The ``ChainResidueSpec``
        entries for disulfide-bonded cysteines exist only to carry the
        ``new_resname="CYX"`` rename; the parameterization of the S–S bond
        is handled by AMBER's built-in CYX template, so no user intervention
        is required.

    Parameters
    ----------
    mol : :class:`moleculekit.molecule.Molecule`
        Input molecule. Should already carry covalent bonds (read from a
        PDB ``CONECT`` block, a CIF ``_struct_conn`` block, or set up via
        :meth:`Molecule.templateResidueFromSmiles`). If ``mol.bonds`` is
        empty and ``guess_bonds`` is True, the detector falls back to
        distance-based bond guessing via ``mol._guessBonds()`` and logs a
        warning. The molecule is not mutated.
    guess_bonds : bool
        When ``mol.bonds`` is empty, guess bonds from atom coordinates so
        crosslinks (disulfides, glycosidic bonds, ...) can still be found.
        Set to False to skip guessing and rely only on explicit input
        bonds: useful for modelled structures whose slightly-off geometry
        produces spurious close contacts that would otherwise be flagged as
        bonds. When guessing is on, non-peptide bonds landing on a canonical
        amino acid's backbone ``O`` / ``CA`` (atoms that never form a real
        crosslink) are treated as guessing artifacts and ignored; explicit
        input bonds are always trusted.

    Returns
    -------
    list[PerResidueSpec]
        Flat list mixing :class:`ChainResidueSpec`, :class:`ScaffoldSpec`,
        :class:`CovalentLigandSpec`, and :class:`LigandSpec` entries.
        Ordered by residue index in ``mol``. Empty when the molecule has
        no non-standard residues and no sidechain crosslinks.

    Raises
    ------
    RuntimeError
        If a canonical residue is bonded at an anchor atom that is not
        listed in
        :data:`moleculekit.tools._anchor_variants.ANCHOR_TABLE` (the
        anchor needs to be registered there before the residue can be
        re-templated). Also raised if an NCAA resname is 4+ characters
        long and requires terminus-disambiguation prefixing, which would
        exceed AMBER's 4-character prepi unit-name limit.

    Examples
    --------
    Detect non-standard residues, template them with SMILES, then prepare:

    >>> from moleculekit.molecule import Molecule
    >>> from moleculekit.tools.nonstandard_residues import detectNonStandardResidues
    >>> from moleculekit.tools.preparation import systemPrepare
    >>> mol = Molecule("3ptb")  # doctest: +SKIP
    >>> specs = detectNonStandardResidues(mol)  # doctest: +SKIP

    For a molecule that has a non-canonical residue (e.g. "LIG") that
    needs SMILES-based templating before preparation:

    >>> # Template the non-canonical residue with its SMILES
    >>> lig_mask = mol.resname == "LIG"  # doctest: +SKIP
    >>> mol.remove("hydrogen")  # doctest: +SKIP
    >>> mol.templateResidueFromSmiles(lig_mask, smiles="...", addHs=True)  # doctest: +SKIP
    >>> # Now pass the specs so systemPrepare does not re-detect
    >>> pmol, specs = systemPrepare(mol, detect_specs=specs)  # doctest: +SKIP

    When no non-standard-residue handling is needed, pass specs directly:

    >>> pmol, specs, df = systemPrepare(mol, return_details=True)  # doctest: +SKIP
    """
    bonds_guessed = guess_bonds and len(mol.bonds) == 0
    bonds = _ensure_bonds(mol, guess=guess_bonds)
    a2r, residues, atom_idxs = _residue_groups(mol)
    n_res = len(residues)

    # Walk every inter-residue bond once. For each residue we track its
    # non-peptide partners (with the atom name on this side); for every
    # residue we also track whether it has a peptide bond on its N side
    # and / or its C side, which feeds the chain-residency check and
    # the is_n_term / is_c_term flags on ChainResidueSpec.
    nonpep_partners = [set() for _ in range(n_res)]  # res -> {(other_res, this_atom_name)}
    peptide_attached_n = [False] * n_res
    peptide_attached_c = [False] * n_res
    has_peptide_bond = [False] * n_res

    seen_bonds = set()
    for a1, a2 in bonds:
        a1, a2 = int(a1), int(a2)
        bond_key = (a1, a2) if a1 < a2 else (a2, a1)
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)
        r1, r2 = int(a2r[a1]), int(a2r[a2])
        if r1 == r2 or r1 < 0 or r2 < 0:
            continue
        # Metal-coordination contacts (e.g. PDB LINK records between a Zn ion
        # and a Zn-chelating inhibitor) are stored as bonds but are not
        # covalent. Skipping them keeps such inhibitors classified as free
        # ligands rather than scaffolds. Bonds touching waters get the
        # same treatment: waters can appear in LINK records (coordinating
        # waters, water bridges) but are never real covalent partners.
        if (
            residues[r1].resname in _ION_RESNAMES
            or residues[r2].resname in _ION_RESNAMES
            or residues[r1].resname in WATER_RESIDUE_NAMES
            or residues[r2].resname in WATER_RESIDUE_NAMES
        ):
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
            # Drop guessed-bond artifacts that land on a canonical amino
            # acid's backbone O / CA (atoms that never form a real
            # crosslink); a slightly-off modelled structure otherwise yields
            # spurious close contacts here. Explicit input bonds are trusted.
            spurious = bonds_guessed and (
                (
                    residues[r1].resname in PROTEIN_RESNAMES
                    and n1 in _BACKBONE_NONANCHOR_ATOMS
                )
                or (
                    residues[r2].resname in PROTEIN_RESNAMES
                    and n2 in _BACKBONE_NONANCHOR_ATOMS
                )
            )
            if not spurious:
                nonpep_partners[r1].add((r2, n1))
                nonpep_partners[r2].add((r1, n2))

    # Distance fallback: residues with a backbone N or C atom but no
    # explicit peptide bond on that side may still be chain-resident.
    # Sparse CIF inputs frequently carry only the special inter-residue
    # bonds (via ``_struct_conn``) and omit canonical peptide bonds.
    for r_idx in range(n_res):
        if not peptide_attached_n[r_idx] and _has_peptide_neighbour(
            mol, atom_idxs[r_idx], "N"
        ):
            peptide_attached_n[r_idx] = True
            has_peptide_bond[r_idx] = True
        if not peptide_attached_c[r_idx] and _has_peptide_neighbour(
            mol, atom_idxs[r_idx], "C"
        ):
            peptide_attached_c[r_idx] = True
            has_peptide_bond[r_idx] = True

    chain_resident = [
        residues[r].resname in PROTEIN_RESNAMES or has_peptide_bond[r]
        for r in range(n_res)
    ]

    # Plan the canonical renames. Bucket key is (canonical_resname,
    # anchor_atom_name, partner_resname, n_term, c_term); residues
    # sharing a bucket get the same 3-char custom resname so the
    # parameterizer emits one shared prepi. anchor_atom is the
    # deterministically-first non-peptide partner (sorted by partner
    # residue index, then anchor atom name); stored on the spec so
    # downstream consumers don't have to re-walk mol.bonds.
    anchor_atoms = {}  # r_idx -> anchor atom name
    bucket_to_resname = {}
    prefix_counter = {}
    canonical_renames = {}  # r_idx -> new_resname

    for r_idx in range(n_res):
        if not nonpep_partners[r_idx]:
            continue
        anchor_partner = sorted(
            nonpep_partners[r_idx], key=lambda p: (p[0], p[1])
        )[0]
        other_r, anchor_atom = anchor_partner
        anchor_atoms[r_idx] = anchor_atom

        residue = residues[r_idx]
        # Only canonical amino acids get renamed. NCAAs keep their
        # original resname (the detector's job is to flag them, not
        # rename them); ions, water and caps are also "canonical" in
        # the no-need-to-parameterize sense, but they have no sidechain
        # to crosslink in the first place and never reach here.
        if residue.resname not in PROTEIN_RESNAMES:
            continue
        partner_resname = residues[other_r].resname

        # Validate the anchor against ANCHOR_TABLE.
        if lookup_anchor(residue.resname, anchor_atom) is None:
            if bonds_guessed:
                cause = (
                    "The molecule had no bonds, so they were guessed from "
                    "atom coordinates; this is most likely a spurious close "
                    "contact from slightly-off geometry (common with modelled "
                    "structures) rather than a real bond. Provide explicit "
                    "bonds (CONECT / _struct_conn records), or pass "
                    "guess_bonds=False to skip bond guessing."
                )
            else:
                cause = (
                    "This is most often a mistake in the input bonds (a "
                    "spurious or misassigned CONECT / LINK / _struct_conn "
                    "record); please check that this bond is real."
                )
            raise RuntimeError(
                f"Found an unrecognized bond between {residue.resname} "
                f"{residue.chain}:{residue.resid}{residue.insertion} atom "
                f"{anchor_atom} and {partner_resname}. {cause} If the bond is "
                f"correct, this sidechain chemistry is not yet supported: "
                f"please contact the developers to add the "
                f"{residue.resname}-{anchor_atom} anchor to "
                f"moleculekit.tools._anchor_variants.ANCHOR_TABLE."
            )

        n_term = not peptide_attached_n[r_idx]
        c_term = not peptide_attached_c[r_idx]

        # Disulfide special case: both ends share the fixed name 'CYX'.
        # AMBER's CYX ff14SB template handles the chemistry; htmd's
        # _canonical_variant_template selects the N/C-terminal variant
        # (NCYX / CCYX) at parameterization time from is_n_term/is_c_term.
        if (
            residue.resname in ("CYS", "CYX")
            and anchor_atom == "SG"
            and partner_resname in ("CYS", "CYX")
        ):
            canonical_renames[r_idx] = "CYX"
            continue

        bucket_key = (residue.resname, anchor_atom, partner_resname, n_term, c_term)
        if bucket_key in bucket_to_resname:
            canonical_renames[r_idx] = bucket_to_resname[bucket_key]
        else:
            new_resname = _pick_bucket_resname("XX", prefix_counter)
            bucket_to_resname[bucket_key] = new_resname
            canonical_renames[r_idx] = new_resname

    specs = []
    for r_idx, residue in enumerate(residues):
        is_canonical = residue.resname in _CANONICAL_RESNAMES
        is_renamed_canonical = r_idx in canonical_renames
        nonpep_count = len(nonpep_partners[r_idx])

        if is_canonical and not is_renamed_canonical:
            continue  # plain canonical AA, no special handling needed

        if chain_resident[r_idx] or is_renamed_canonical:
            specs.append(
                ChainResidueSpec(
                    resname=residue.resname,
                    residue=residue,
                    new_resname=canonical_renames.get(r_idx),
                    anchor_atom=anchor_atoms.get(r_idx),
                    is_n_term=not peptide_attached_n[r_idx],
                    is_c_term=not peptide_attached_c[r_idx],
                )
            )
        elif nonpep_count >= 2:
            specs.append(ScaffoldSpec(resname=residue.resname, residue=residue))
        elif nonpep_count == 1:
            specs.append(CovalentLigandSpec(resname=residue.resname, residue=residue))
        else:
            specs.append(LigandSpec(resname=residue.resname, residue=residue))

    _disambiguate_terminus_resnames(specs)
    return specs
