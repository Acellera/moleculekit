"""Discovery helper for non-standard residues that need user-driven AMBER
parameterization before building.

A "non-standard residue" is any residue whose ``resname`` is not in moleculekit's
canonical-AA / nucleic / water / ion sets. Each such residue is classified into:

  - ``"scaffolded_peptide"``: a HETATM scaffold covalently bonded (non-peptide)
    to ``>= min_anchors`` canonical-AA residues. The chemistry behind cyclic
    peptides held in shape by a small-molecule scaffold ("bicycles", strictly
    bicyclic/tricyclic etc. depending on anchor count).
  - ``"ncaa"``: a non-canonical amino acid embedded in a polymer chain via
    peptide N-C bonds to canonical AAs.
  - ``"covalent_ligand"``: a non-standard residue bonded to exactly one
    canonical-AA residue via a non-peptide covalent bond (NAG glycosylation
    on Asn, single-Cys heme attachment, covalent-inhibitor monoadducts, ...).
    Needs a ``custombonds`` entry at build time (and possibly an anchor
    rename via ``force_protonation``).
  - ``"ligand"``: a free, non-covalently bound non-standard residue (small
    drug molecules, fatty acids in binding pockets, ...). Parameterized
    standalone; needs no ``custombonds`` or ``force_protonation``.
  - ``"peptide_crosslink"``: a direct sidechain-to-sidechain covalent bond
    between two non-canonical amino acids that are themselves peptide-bonded
    into a polymer chain. Pattern-B stapled peptides (e.g. olefin-metathesis
    staples between two S5/R8 residues, click-chemistry staples). The two
    endpoint residues are classified separately as ``"ncaa"``; the crosslink
    itself is reported as its own spec so the bond can be emitted via
    ``custombondsFromSpecs``.

For each residue the helper writes a model-compound CIF ready for handing to
parameterization functions and returns metadata the caller uses to derive the
standard arguments to
:func:`moleculekit.tools.preparation.systemPrepare` (``force_protonation``) and
:func:`htmd.builder.amber.build` (``custombonds``). Two thin helpers do that
conversion: :func:`forceProtonationFromSpecs` and :func:`custombondsFromSpecs`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import numpy as np

from moleculekit.molecule import Molecule, UniqueAtomID, UniqueResidueID
from moleculekit.residues import (
    PROTEIN_RESIDUE_NAMES,
    NUCLEIC_RESIDUE_NAMES,
    MODIFIED_PROTEIN_RESIDUE_NAMES,
    PROTEIN_RESIDUES,
    NUCLEIC_RESIDUES,
    WATER_RESIDUE_NAMES,
)
from moleculekit.tools._anchor_variants import (
    ANCHOR_VARIANTS,
    lookup_anchor_variant,
)
from moleculekit import __share_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses for detector output
#
# Residue and atom identities reuse :class:`UniqueResidueID` /
# :class:`UniqueAtomID` from :mod:`moleculekit.molecule` so callers can
# round-trip them against any in-memory ``Molecule`` (via ``selectAtoms`` /
# ``selectAtom``).
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldAnchor:
    """One covalent anchor between a canonical-residue sidechain atom and a
    scaffolded-peptide scaffold atom. ``anchor_atom`` is the canonical-AA
    sidechain atom (e.g. Cys SG); ``scaffold_atom`` is the bonded atom on
    the non-canonical scaffold."""

    anchor_atom: UniqueAtomID
    scaffold_atom: UniqueAtomID


@dataclass
class ModelAtom:
    """Per-atom record in a scaffolded-peptide model compound. ``role`` is
    either ``"scaffold"`` or ``"stub"``; ``ff_type`` is the canonical-FF
    atom type for stub atoms (``None`` for scaffold atoms)."""

    role: str
    ff_type: Optional[str]


@dataclass
class ScaffoldedPeptideSpec:
    """A non-canonical residue covalently bonded (via non-peptide bonds) to
    two or more canonical residues."""

    category: ClassVar[str] = "scaffolded_peptide"
    resname: str
    residue: UniqueResidueID
    anchors: list[ScaffoldAnchor]
    model_compound_cif: Optional[str] = None
    model_atom_map: Optional[dict[str, ModelAtom]] = None


@dataclass
class NCAASpec:
    """A non-canonical amino acid embedded in a polymer chain via peptide
    N-C bonds to canonical AAs."""

    category: ClassVar[str] = "ncaa"
    resname: str
    residue: UniqueResidueID
    head_atom: str
    tail_atom: str
    is_n_term: bool
    is_c_term: bool
    model_compound_cif: Optional[str] = None


@dataclass
class LigandSpec:
    """Non-standard residue with no covalent bonds to canonical-AA residues:
    small molecules, drug ligands, fatty acids, etc. Parameterized
    standalone; needs no ``custombonds`` or ``force_protonation`` at build."""

    category: ClassVar[str] = "ligand"
    resname: str
    residue: UniqueResidueID
    model_compound_cif: Optional[str] = None


@dataclass
class CovalentLigandSpec:
    """Non-standard residue bonded to exactly one canonical-AA residue via a
    non-peptide covalent bond (e.g. NAG-Asn glycosylation, single-Cys heme
    attachment, covalent-inhibitor monoadducts). The ``anchor`` carries the
    canonical-AA atom and the ligand atom on the bond. Needs a custombond
    at build time, and possibly an anchor rename via ``force_protonation``
    if the canonical anchor atom appears in :data:`ANCHOR_VARIANTS`."""

    category: ClassVar[str] = "covalent_ligand"
    resname: str
    residue: UniqueResidueID
    anchor: ScaffoldAnchor
    model_compound_cif: Optional[str] = None


@dataclass
class PeptideCrosslinkSpec:
    """A direct sidechain-to-sidechain covalent bond between two
    non-canonical amino acids that are both peptide-bonded into a polymer
    chain (Pattern-B stapled peptides). Each endpoint NCAA is classified
    separately as :class:`NCAASpec`; this spec carries only the crosslink
    bond itself so it can be emitted via ``custombondsFromSpecs``."""

    category: ClassVar[str] = "peptide_crosslink"
    atom_a: UniqueAtomID
    atom_b: UniqueAtomID


NonStandardResidueSpec = Union[
    ScaffoldedPeptideSpec,
    NCAASpec,
    CovalentLigandSpec,
    LigandSpec,
    PeptideCrosslinkSpec,
]


with open(os.path.join(__share_dir, "atomselect", "atomselect.json")) as _f:
    _ION_RESNAMES = set(json.load(_f).get("ion_resnames", []))


# Standard peptide-terminus caps. AMBER ff14SB / ff19SB ship parameters for
# these so they don't need user-driven parameterization. ACE = acetyl
# (N-terminal); NME = N-methylamide and NHE/NH2 = ammonia (C-terminal).
# moleculekit treats ACE/NME as part of "protein" elsewhere
# (preparation, autosegment, metricdihedral).
_CAP_RESNAMES = {"ACE", "NME", "NHE", "NH2"}


# Resnames that should never be flagged as non-standard.
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
# When classifying inter-residue covalent bonds out of a non-standard residue,
# we only count bonds to a protein partner as "anchor" bonds (used both for
# the scaffolded-peptide >=N-anchors threshold and for the NCAA peptide-bond
# heuristic). Bonds to water, ions, nucleic residues, or other non-standard
# residues are ignored. Includes the standard ff14SB resname variants
# (CYX, HID/HIE/HIP, LYN, ...) so an already-prepared input still classifies.
_PROTEIN_ANCHOR_SET = set(PROTEIN_RESIDUE_NAMES)
for _rr in PROTEIN_RESIDUES:
    _PROTEIN_ANCHOR_SET.update(_rr.resname_variants)


_RESIDUE_FIELDS = ("resname", "resid", "insertion", "segid", "chain")


def _residue_groups(mol):
    """Return ``(a2r, groups)`` where ``a2r`` maps each atom index to a unique
    residue id (via :meth:`Molecule.getResidues`) and ``groups`` is a list of
    per-residue dicts ``{atom_idx, resname, resid, insertion, segid, chain}``
    in residue-id order."""
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
    """Return mol.bonds, guessing if empty. Does not mutate ``mol``."""
    if len(mol.bonds):
        return mol.bonds
    logger.warning(
        "Molecule has no bonds; falling back to distance-based bond guessing "
        "for non-standard residue detection."
    )
    bonds = mol._guessBonds(rdkit=False)
    return np.asarray(bonds, dtype=np.int64)


def _classify_residues(mol, a2r, groups, bonds, min_anchors_for_scaffold):
    """Return ``(classifications, inter_residue_bonds)`` where
    ``classifications`` is a dict ``{residue_index -> info}`` carrying a
    ``category`` plus category-specific fields, and
    ``inter_residue_bonds`` is a list of ``(atom_a, atom_b, residue_a, residue_b)``
    tuples for every non-peptide inter-residue bond in the molecule (used
    later for peptide-crosslink detection).

    Standard residues are absent from ``classifications``.
    """
    # Collect per-residue inter-residue bonds: list of (other_atom_idx, this_atom_idx, bond_idx)
    inter_bonds_per_res = [[] for _ in groups]
    inter_residue_bonds = []  # (a1, a2, r1, r2) for non-peptide bonds only
    for bi, (a1, a2) in enumerate(bonds):
        r1, r2 = a2r[a1], a2r[a2]
        if r1 == r2 or r1 < 0 or r2 < 0:
            continue
        inter_bonds_per_res[r1].append((a2, a1, bi))
        inter_bonds_per_res[r2].append((a1, a2, bi))
        if not _is_peptide_bond(str(mol.name[a1]), str(mol.name[a2])):
            inter_residue_bonds.append((int(a1), int(a2), int(r1), int(r2)))

    classifications = {}
    for r_idx, g in enumerate(groups):
        resname = g["resname"]
        if resname in _CANONICAL_RESNAMES:
            continue

        # Partition this residue's inter-residue bonds.
        # For each bond, the partner residue is canonical-AA (anchor candidate),
        # peptide-bonded (N-C between residues), or other (other non-standard, etc.).
        anchor_bonds = []  # bonds to canonical-protein residues (non-peptide)
        peptide_bonds = []  # standard peptide N<->C bonds
        for other_atom, this_atom, bi in inter_bonds_per_res[r_idx]:
            other_r = groups[a2r[other_atom]]
            other_resname = other_r["resname"]
            this_name = str(mol.name[this_atom])
            other_name = str(mol.name[other_atom])
            if _is_peptide_bond(this_name, other_name):
                peptide_bonds.append(
                    {
                        "this_atom": int(this_atom),
                        "this_atom_name": this_name,
                        "other_residue_idx": int(a2r[other_atom]),
                        "other_atom_name": other_name,
                    }
                )
                continue
            if other_resname in _PROTEIN_ANCHOR_SET:
                anchor_bonds.append(
                    {
                        "anchor_atom_idx": int(other_atom),
                        "anchor_atom_name": other_name,
                        "scaffold_atom_idx": int(this_atom),
                        "scaffold_atom_name": this_name,
                        "other_residue_idx": int(a2r[other_atom]),
                    }
                )

        if len(anchor_bonds) >= min_anchors_for_scaffold:
            classifications[r_idx] = {
                "category": "scaffolded_peptide",
                "anchors": anchor_bonds,
                "peptide_bonds": peptide_bonds,
            }
            continue

        if len(peptide_bonds) >= 1:
            # Embedded in a chain via at least one peptide bond.
            classifications[r_idx] = {
                "category": "ncaa",
                "peptide_bonds": peptide_bonds,
                "anchors": anchor_bonds,
            }
            continue

        if len(anchor_bonds) >= 1:
            # Non-peptide covalent bond(s) to canonical residue(s) but fewer
            # than the scaffold threshold: a covalent ligand / single-anchor
            # PTM (NAG glycosylation, single-Cys heme, ...).
            classifications[r_idx] = {
                "category": "covalent_ligand",
                "anchors": anchor_bonds,
            }
            continue

        # No covalent bonds to canonical residues: free non-covalent ligand.
        classifications[r_idx] = {"category": "ligand"}

    return classifications, inter_residue_bonds


def _is_peptide_bond(name_a, name_b):
    """Return True if (name_a, name_b) looks like a peptide N-C bond."""
    return {name_a, name_b} == {"N", "C"}


def _residue_id(g):
    """Build a :class:`UniqueResidueID` from a residue-group dict."""
    return UniqueResidueID(
        resname=g["resname"],
        chain=g["chain"],
        resid=g["resid"],
        insertion=g["insertion"],
        segid=g["segid"],
    )


def _load_known_resnames():
    """Return the set of resnames already covered by htmd's bundled AMBER
    cofactors / NCAA / PTM registry. Soft-imports htmd; returns an empty set
    when htmd is not available."""
    try:
        from htmd.home import home  # type: ignore
    except Exception:
        return set()
    base = os.path.join(home(shareDir=True), "builder", "amberfiles")
    out = set()
    for sub in ("cofactors", "ff-ncaa", "ff-ptm"):
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            full = os.path.join(d, entry)
            if os.path.isdir(full):
                out.add(entry)
    return out


# ---------------------------------------------------------------------------
# Model-compound construction
# ---------------------------------------------------------------------------


def _heavy_neighbor(mol, idx, exclude):
    """Return the index of a heavy-atom neighbor of ``idx`` (via
    :meth:`Molecule.getNeighbors`), excluding any indices in ``exclude``.
    ``None`` if there is no such neighbor."""
    excluded = set(exclude)
    for j in mol.getNeighbors(idx):
        j = int(j)
        if j not in excluded and mol.element[j] != "H":
            return j
    return None


def _idealized_methyl_positions(center, neighbor_pos, bond_length=1.09):
    """Return three idealized H positions around an sp3 ``center`` whose
    only other neighbour is at ``neighbor_pos``. Positions are placed
    tetrahedrally at ``bond_length``, opposite the neighbour direction."""
    direction = neighbor_pos - center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(direction, ref)
    x /= np.linalg.norm(x)
    y = np.cross(direction, x)
    cos_t = np.cos(np.deg2rad(180 - 109.471))
    sin_t = np.sin(np.deg2rad(180 - 109.471))
    out = []
    for k in range(3):
        phi = 2 * np.pi * k / 3
        d = cos_t * (-direction) + sin_t * (np.cos(phi) * x + np.sin(phi) * y)
        out.append(center + bond_length * d)
    return out


def _build_scaffold_model(mol, scaffold_group, anchors):
    """Construct the model compound for a scaffolded-peptide scaffold.

    The model retains the scaffold residue verbatim, then attaches a small
    canonical-sidechain stub at each anchor: the anchor heavy atom and its
    CB-equivalent next-sidechain neighbour are *sliced from the input
    molecule* so the stub keeps the real anchor-CB direction (important when
    the model compound is later fed to a parameterizer that fits to the
    conformation, e.g. RESP/ESP). The CB-equivalent is then capped with three
    methyl-style hydrogens at idealized sp3 positions; methyl rotation is
    free so we don't lose anything by idealizing those.

    Stub atom names follow the pattern ``<element><i>A<k>`` (e.g. ``S1A1``,
    ``C1A2``, ``H1A3``): antechamber parses the first 1-2 chars as element
    symbol, and with a digit in slot 2 both antechamber and moleculekit's
    name-based element guesser fall back cleanly to "element = first char".

    Returns ``(model_mol, atom_map)`` where ``atom_map`` is a dict
    ``{atom_name: {"role", "ff_type"}}``. ``role`` is ``"scaffold"`` or
    ``"stub"``; ``ff_type`` is the canonical-FF atom type for stub atoms,
    ``None`` for scaffold atoms.
    """
    n_scaffold_atoms = len(scaffold_group["atom_idx"])

    model = mol.copy(sel=scaffold_group["atom_idx"])
    # Use a single residue/segment so antechamber treats the whole compound as one molecule.
    model.resname[:] = scaffold_group["resname"][:4]
    model.resid[:] = 1
    model.segid[:] = "A"
    model.chain[:] = "A"
    model.insertion[:] = ""

    atom_map = {str(n): {"role": "scaffold", "ff_type": None} for n in model.name}

    for i, anc in enumerate(anchors, start=1):
        anchor_idx = int(anc.anchor_atom.selectAtom(mol))
        scaffold_idx = int(anc.scaffold_atom.selectAtom(mol))
        anchor_element = str(mol.element[anchor_idx]).upper()
        anchor_resname = str(anc.anchor_atom.resname)
        anchor_name_canon = str(anc.anchor_atom.name)

        variant = lookup_anchor_variant(anchor_resname, anchor_name_canon)
        if variant is None:
            logger.warning(
                "No ANCHOR_VARIANTS entry for (%s, %s); using a generic stub. "
                "Junction parameters for this anchor may be inaccurate.",
                anchor_resname,
                anchor_name_canon,
            )
            anchor_ff_type = None
            cb_ff_type = "CT"
            hb_ff_type = "HC"
        else:
            anchor_ff_type = variant["ff14sb_type"]
            cb_ff_type = variant.get("cb_ff14sb_type", "CT")
            hb_ff_type = variant.get("hb_ff14sb_type", "H1")

        # Slice real anchor + CB-equivalent positions from the input. CB is
        # the heavy neighbour of the anchor that isn't the scaffold side.
        # Falls back to a synthetic CB along the anchor-scaffold axis if the
        # input has no heavy bond out of the anchor.
        anchor_pos = mol.coords[anchor_idx, :, mol.frame].copy()
        cb_idx = _heavy_neighbor(mol, anchor_idx, exclude=[scaffold_idx])
        if cb_idx is not None:
            cb_pos = mol.coords[cb_idx, :, mol.frame].copy()
        else:
            scaffold_pos = mol.coords[scaffold_idx, :, mol.frame].copy()
            d = anchor_pos - scaffold_pos
            cb_pos = anchor_pos + (d / max(np.linalg.norm(d), 1e-6)) * 1.5

        # Three methyl Hs placed tetrahedrally around CB, opposite the anchor.
        h_positions = _idealized_methyl_positions(cb_pos, anchor_pos)

        anchor_name = f"{anchor_element}{i}A1"
        cb_name = f"C{i}A2"
        h_names = [f"H{i}A{k}" for k in (3, 4, 5)]

        stub = Molecule().empty(5)
        stub.name[:] = [anchor_name, cb_name, *h_names]
        stub.element[:] = [anchor_element, "C", "H", "H", "H"]
        stub.resname[:] = scaffold_group["resname"][:4]
        stub.resid[:] = 1
        stub.segid[:] = "A"
        stub.chain[:] = "A"
        stub.coords = np.vstack([anchor_pos, cb_pos, *h_positions])[:, :, np.newaxis].astype(np.float32)
        stub.record[:] = "HETATM"

        n_before = model.numAtoms
        model.append(stub, collisions=False)

        # Stub-internal bonds + scaffold-anchor to stub-anchor bond.
        a, cb, h0, h1, h2 = (n_before + k for k in range(5))
        scaffold_in_model = int(
            np.where(model.name[:n_scaffold_atoms] == anc.scaffold_atom.name)[0][0]
        )
        for a1, a2 in ((a, cb), (cb, h0), (cb, h1), (cb, h2), (scaffold_in_model, a)):
            model.addBond(a1, a2, "1")

        atom_map[anchor_name] = {"role": "stub", "ff_type": anchor_ff_type}
        atom_map[cb_name] = {"role": "stub", "ff_type": cb_ff_type}
        for h in h_names:
            atom_map[h] = {"role": "stub", "ff_type": hb_ff_type}

    return model, atom_map


def _build_cofactor_model(mol, group):
    """Return a model compound for a free / single-anchor non-standard residue.

    Currently this is just the residue itself, isolated as a single-residue
    Molecule. H caps for dangling valences are not added; the caller is
    expected to template hydrogens upstream (e.g. via
    ``mol.templateResidueFromSmiles``) before invoking the detector."""
    sub = mol.copy(sel=group["atom_idx"])
    sub.resid[:] = 1
    sub.segid[:] = "A"
    sub.chain[:] = "A"
    sub.insertion[:] = ""
    return sub


def _build_ncaa_model(mol, group):
    """Return a model compound for an NCAA. Just the bare residue; the
    downstream ``parameterizeNonCanonicalResidues`` handles ACE/NME capping
    internally via its ``_extend_residue`` step."""
    return _build_cofactor_model(mol, group)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detectNonStandardResidues(
    mol,
    outdir=None,
    write_models=True,
    include_known=False,
    min_anchors_for_scaffold=2,
):
    """Detect non-standard residues and (optionally) write model-compound CIFs.

    Parameters
    ----------
    mol : :class:`moleculekit.molecule.Molecule`
        Input molecule. Must already carry the bonds you care about (read from
        a PDB CONECT or CIF ``_struct_conn`` block); if ``mol.bonds`` is empty,
        the detector falls back to ``mol._guessBonds()``.
    outdir : str, optional
        Output directory for model-compound CIFs. Required when
        ``write_models=True``.
    write_models : bool, optional
        If True, write a CIF model compound per detected residue into
        ``outdir``. The path is recorded in each spec under
        ``model_compound_cif``.
    include_known : bool, optional
        If False (default), skip residues whose resname is already covered by
        htmd's bundled AMBER cofactor / NCAA / PTM registry.
    min_anchors_for_scaffold : int, optional
        Number of covalent bonds to canonical-AA residues a HETATM must have
        to be classified as a scaffolded peptide. Default 2.

    Returns
    -------
    list[NonStandardResidueSpec]
        One :class:`ScaffoldedPeptideSpec`, :class:`NCAASpec`,
        :class:`CovalentLigandSpec`, or :class:`LigandSpec` per detected
        residue.
    """
    if write_models:
        if outdir is None:
            raise ValueError("outdir is required when write_models=True")
        os.makedirs(outdir, exist_ok=True)

    bonds = _ensure_bonds(mol)
    a2r, groups = _residue_groups(mol)
    classifications, inter_residue_bonds = _classify_residues(
        mol, a2r, groups, bonds, min_anchors_for_scaffold
    )

    known = set() if include_known else _load_known_resnames()

    specs = []
    for r_idx, info in classifications.items():
        g = groups[r_idx]
        if g["resname"] in known:
            logger.info(
                "Skipping residue %s:%s%s: covered by htmd's built-in registry.",
                g["resname"],
                g["resid"],
                f"_{g['insertion']}" if g["insertion"] else "",
            )
            continue

        residue_id = _residue_id(g)
        cif_path = (
            os.path.join(outdir, f"{g['resname']}_model.cif") if write_models else None
        )

        if info["category"] == "scaffolded_peptide":
            anchors = [
                ScaffoldAnchor(
                    anchor_atom=UniqueAtomID.fromMolecule(
                        mol, idx=anc["anchor_atom_idx"]
                    ),
                    scaffold_atom=UniqueAtomID.fromMolecule(
                        mol, idx=anc["scaffold_atom_idx"]
                    ),
                )
                for anc in info["anchors"]
            ]
            atom_map_obj = None
            if write_models:
                model, atom_map_raw = _build_scaffold_model(mol, g, anchors)
                model.write(cif_path)
                atom_map_obj = {
                    name: ModelAtom(role=v["role"], ff_type=v["ff_type"])
                    for name, v in atom_map_raw.items()
                }
            specs.append(
                ScaffoldedPeptideSpec(
                    resname=g["resname"],
                    residue=residue_id,
                    anchors=anchors,
                    model_compound_cif=cif_path,
                    model_atom_map=atom_map_obj,
                )
            )

        elif info["category"] == "ncaa":
            head_attached = any(
                pb["this_atom_name"] == "N" for pb in info["peptide_bonds"]
            )
            tail_attached = any(
                pb["this_atom_name"] == "C" for pb in info["peptide_bonds"]
            )
            if write_models:
                _build_ncaa_model(mol, g).write(cif_path)
            specs.append(
                NCAASpec(
                    resname=g["resname"],
                    residue=residue_id,
                    head_atom="N",
                    tail_atom="C",
                    is_n_term=not head_attached,
                    is_c_term=not tail_attached,
                    model_compound_cif=cif_path,
                )
            )

        elif info["category"] == "covalent_ligand":
            anc = info["anchors"][0]
            anchor = ScaffoldAnchor(
                anchor_atom=UniqueAtomID.fromMolecule(
                    mol, idx=anc["anchor_atom_idx"]
                ),
                scaffold_atom=UniqueAtomID.fromMolecule(
                    mol, idx=anc["scaffold_atom_idx"]
                ),
            )
            if write_models:
                _build_cofactor_model(mol, g).write(cif_path)
            specs.append(
                CovalentLigandSpec(
                    resname=g["resname"],
                    residue=residue_id,
                    anchor=anchor,
                    model_compound_cif=cif_path,
                )
            )

        elif info["category"] == "ligand":
            if write_models:
                _build_cofactor_model(mol, g).write(cif_path)
            specs.append(
                LigandSpec(
                    resname=g["resname"],
                    residue=residue_id,
                    model_compound_cif=cif_path,
                )
            )

    # Pattern-B stapled peptides: scan inter-residue non-peptide bonds for
    # NCAA-NCAA crosslinks. Cys-Cys disulfides are skipped (they are handled
    # separately by amber.build's disulfide path).
    cys_resnames = {"CYS", "CYM", "CYX"}
    for a1, a2, r1, r2 in inter_residue_bonds:
        cat1 = classifications.get(r1, {}).get("category")
        cat2 = classifications.get(r2, {}).get("category")
        if cat1 != "ncaa" or cat2 != "ncaa":
            continue
        if (
            groups[r1]["resname"] in cys_resnames
            and groups[r2]["resname"] in cys_resnames
            and str(mol.name[a1]) == "SG"
            and str(mol.name[a2]) == "SG"
        ):
            continue
        specs.append(
            PeptideCrosslinkSpec(
                atom_a=UniqueAtomID.fromMolecule(mol, idx=a1),
                atom_b=UniqueAtomID.fromMolecule(mol, idx=a2),
            )
        )

    return specs


def _spec_anchors(spec):
    """Yield each :class:`ScaffoldAnchor` carried by a spec.
    :class:`ScaffoldedPeptideSpec` carries a list; :class:`CovalentLigandSpec`
    a single anchor; other spec classes none."""
    if isinstance(spec, ScaffoldedPeptideSpec):
        yield from spec.anchors
    elif isinstance(spec, CovalentLigandSpec):
        yield spec.anchor


def forceProtonationFromSpecs(specs):
    """Convert detector specs to a ``force_protonation`` list for systemPrepare.

    Each scaffolded-peptide / covalent-ligand anchor whose
    ``(resname, anchor_atom_name)`` is in ``ANCHOR_VARIANTS`` with a non-None
    ``variant`` produces a ``(atomselect_string, variant_resname)`` entry.
    Unknown anchors are skipped with a warning."""
    out = []
    for spec in specs:
        for anc in _spec_anchors(spec):
            atom_id = anc.anchor_atom
            entry = lookup_anchor_variant(atom_id.resname, atom_id.name)
            if entry is None:
                logger.warning(
                    "No ANCHOR_VARIANTS entry for (%s, %s); skipping force_protonation rename.",
                    atom_id.resname,
                    atom_id.name,
                )
                continue
            if entry["variant"] is None:
                logger.warning(
                    "ANCHOR_VARIANTS has no FF variant for (%s, %s); systemPrepare "
                    "will keep the canonical residue name. The displaced H must be "
                    "removed manually.",
                    atom_id.resname,
                    atom_id.name,
                )
                continue
            out.append((_residue_sel_from_id(atom_id), entry["variant"]))
    return out


def custombondsFromSpecs(specs):
    """Convert detector specs to a ``custombonds`` list for ``amber.build``.

    Emits one ``(anchor_sel, scaffold_atom_sel)`` pair per scaffolded-peptide
    or covalent-ligand anchor, plus one pair per peptide-crosslink bond.
    Atom-selection strings target a single atom by
    ``segid+chain+resid+insertion+name``."""
    out = []
    for spec in specs:
        if isinstance(spec, PeptideCrosslinkSpec):
            out.append(
                [
                    _atom_sel_from_id(spec.atom_a),
                    _atom_sel_from_id(spec.atom_b),
                ]
            )
            continue
        for anc in _spec_anchors(spec):
            out.append(
                [
                    _atom_sel_from_id(anc.anchor_atom),
                    _atom_sel_from_id(anc.scaffold_atom),
                ]
            )
    return out


def _residue_sel_from_id(uid):
    """Build an atomselect string targeting the residue containing this atom
    or residue ID. Accepts :class:`UniqueAtomID` or :class:`UniqueResidueID`."""
    parts = []
    segid = getattr(uid, "segid", "")
    chain = getattr(uid, "chain", "")
    if segid:
        parts.append(f"segid {segid}")
    if chain:
        parts.append(f"chain {chain}")
    parts.append(f"resid {int(uid.resid)}")
    insertion = getattr(uid, "insertion", "")
    if insertion:
        parts.append(f'insertion "{insertion}"')
    return " and ".join(parts)


def _atom_sel_from_id(uaid):
    """Build an atomselect string targeting a single :class:`UniqueAtomID`."""
    return _residue_sel_from_id(uaid) + f" and name {uaid.name}"
