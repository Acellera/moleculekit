# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from typing import TYPE_CHECKING
from moleculekit import __share_dir
import string
import json
import os
import numpy as np
import logging

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)


CHAIN_ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
SEGID_ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

with open(os.path.join(__share_dir, "atomselect", "atomselect.json")) as _f:
    _sel = json.load(_f)

# Protein residues are identified by the presence of these backbone atoms,
# nucleic residues by any of these backbone link atoms (with ' / * variants).
PROTEIN_BB = ("N", "CA", "C")
NUCLEIC_LINK = ("P", "O3'", "O3*", "C3'", "C3*")


def _residue_atom_coord(mol, res_atom_idx, names):
    """Coordinate of the first atom in ``res_atom_idx`` whose name is in ``names``.

    ``names`` may be a single name or a tuple of acceptable names (e.g. the
    ``O3'``/``O3*`` naming variants). Returns ``None`` if no atom matches.
    """
    if isinstance(names, str):
        names = (names,)
    hit = res_atom_idx[np.isin(mol.name[res_atom_idx], names)]
    if len(hit) == 0:
        return None
    return mol.coords[hit[0], :, 0]


def _classify_residues(mol, sel_mask):
    """Classify every residue in the selection as one of
    protein / nucleic / water / ion / other.

    Protein and nucleic are decided by *backbone-atom presence*, not by
    canonical resname, so noncanonical residues bonded into a chain are still
    treated as polymer.

    Returns
    -------
    cats : list of str
        Category for each residue, in file order.
    residue_idx : list of np.ndarray
        Global atom indices for each residue, in the same order as ``cats``.
    """
    from moleculekit.residues import WATER_RESIDUE_NAMES

    ion_names = set(_sel["ion_resnames"])

    sel_idx = np.where(sel_mask)[0]
    _, residue_idx = mol.getResidues(sel=sel_mask, return_idx=True)
    residue_idx = [sel_idx[idx] for idx in residue_idx]

    cats = []
    for idx in residue_idx:
        rep = idx[0]
        resname = mol.resname[rep]
        names = set(mol.name[idx])
        if resname in WATER_RESIDUE_NAMES:
            cats.append("water")
        elif resname in ion_names:
            cats.append("ion")
        elif all(a in names for a in PROTEIN_BB):
            cats.append("protein")
        elif any(a in names for a in NUCLEIC_LINK):
            cats.append("nucleic")
        else:
            cats.append("other")
    return cats, residue_idx


def _polymer_linked(
    mol,
    prev_idx,
    curr_idx,
    category,
    protein_cutoff,
    nucleic_cutoff,
    ca_fallback_cutoff,
    nucleic_fallback_cutoff,
):
    """Return True if residue ``curr_idx`` is backbone-continuous with ``prev_idx``.

    Protein: ``C(prev)-N(curr)`` within ``protein_cutoff``, else ``CA-CA`` within
    ``ca_fallback_cutoff``. Nucleic: ``O3'(prev)-P(curr)`` within
    ``nucleic_cutoff``, else ``C3'(prev)-P(curr)`` within
    ``nucleic_fallback_cutoff``. If no usable atom pair exists, returns False
    (treated as a break).
    """
    from moleculekit.tools.nonstandard_residues import (
        geometric_interresidue_links,
        _CANONICAL_RESNAMES,
    )

    if category == "protein":
        # Geometric backbone continuation via the shared primitive: a peptide
        # bond always links; a side-chain isopeptide links only at a NON-canonical
        # junction (geometric inference is less certain than a deposited bond, so
        # two canonical residues are not merged on a coincidental side-chain
        # contact - e.g. microcystin's ACB.CG->ARG.N links, a Gln-Lys crosslink
        # between two standard residues does not).
        noncanon = (
            str(mol.resname[prev_idx[0]]) not in _CANONICAL_RESNAMES
            or str(mol.resname[curr_idx[0]]) not in _CANONICAL_RESNAMES
        )
        for _, _, k in geometric_interresidue_links(
            mol, prev_idx, curr_idx, amide_dist=protein_cutoff
        ):
            if k == "peptide" or (k == "isopeptide" and noncanon):
                return True
        # CA-CA fallback ONLY when the backbone C/N needed for the precise check
        # is missing (an incomplete residue) - never for present-but-far C/N,
        # which is a real chain gap.
        if (
            _residue_atom_coord(mol, prev_idx, "C") is None
            or _residue_atom_coord(mol, curr_idx, "N") is None
        ):
            ca0 = _residue_atom_coord(mol, prev_idx, "CA")
            ca1 = _residue_atom_coord(mol, curr_idx, "CA")
            if ca0 is not None and ca1 is not None:
                if float(np.linalg.norm(ca0 - ca1)) <= ca_fallback_cutoff:
                    return True
        # Last resort: honor an explicit deposited backbone bond the geometry
        # above did not catch (a stretched or non-amide CONECT link).
        return _has_deposited_backbone_bond(
            mol, prev_idx, curr_idx, prev_names=("C",), curr_names=("N",)
        )

    if category == "nucleic":
        o3 = _residue_atom_coord(mol, prev_idx, ("O3'", "O3*"))
        p = _residue_atom_coord(mol, curr_idx, "P")
        if o3 is not None and p is not None:
            if float(np.linalg.norm(o3 - p)) <= nucleic_cutoff:
                return True
        else:
            c3 = _residue_atom_coord(mol, prev_idx, ("C3'", "C3*"))
            if c3 is not None and p is not None:
                if float(np.linalg.norm(c3 - p)) <= nucleic_fallback_cutoff:
                    return True
        return _has_deposited_backbone_bond(
            mol, prev_idx, curr_idx,
            prev_names=("O3'", "O3*", "C3'", "C3*"), curr_names=("P",),
        )

    return False


def _has_deposited_backbone_bond(mol, prev_idx, curr_idx, prev_names, curr_names):
    """Return True if an EXPLICIT (deposited) bond joins the two residues at the
    backbone: a bond from ``curr``'s backbone atom (any of ``curr_names``) to any
    atom of ``prev``, or from ``prev``'s backbone atom (any of ``prev_names``) to
    any atom of ``curr``. This honors a deposited CONECT link the geometric
    :func:`geometric_interresidue_links` check could miss (a stretched or non-amide bond),
    while NOT merging side-chain-only crosslinks (e.g. a disulfide), which touch
    neither backbone atom. Only consulted as a fallback when the geometric checks
    in :func:`_polymer_linked` fail, so the bond scan runs at most at chain breaks.
    """
    if mol.bonds is None or len(mol.bonds) == 0:
        return False
    prev_set = set(int(i) for i in prev_idx)
    curr_set = set(int(i) for i in curr_idx)
    curr_bb = {int(i) for i in curr_idx if str(mol.name[int(i)]) in curr_names}
    prev_bb = {int(i) for i in prev_idx if str(mol.name[int(i)]) in prev_names}
    if not curr_bb and not prev_bb:
        return False
    for a, b in mol.bonds:
        a, b = int(a), int(b)
        if (a in curr_bb and b in prev_set) or (b in curr_bb and a in prev_set):
            return True
        if (a in prev_bb and b in curr_set) or (b in prev_bb and a in curr_set):
            return True
    return False


def autoSegment(
    mol: "Molecule",
    sel: str | np.ndarray = "all",
    basename: str = "P",
    fields: tuple = ("segid",),
    protein_cutoff: float = 2.0,
    nucleic_cutoff: float = 2.2,
    ca_fallback_cutoff: float = 5.0,
    nucleic_fallback_cutoff: float = 3.2,
    single_other_segment: bool = False,
    _logger=True,
) -> "Molecule":
    """Segment a Molecule by physical backbone continuity.

    Walks the selected residues in file order and starts a new segment when the
    polymer type, ``chain`` or ``segid`` changes, or when the backbone link
    distance between consecutive residues exceeds the cutoff. Because continuity
    is decided from backbone geometry rather than ``resid`` numbering, residues
    deleted from a sequence with an intact backbone stay in one segment, and the
    whole-system bond graph is never built.

    Water residues collapse to a single segment, ions to a single segment, and
    remaining ("other") molecules are split by bonded connected components.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule object.
    sel : str or np.ndarray
        Atom selection to segment. A selection string, a boolean mask, or an
        integer index array. Atoms outside the selection keep their
        existing chain/segid.
    basename : str
        Base name for segment ids (e.g. 'P' -> 'P0', 'P1', ...).
    fields : tuple
        Fields to set: any combination of "segid" and "chain".
    protein_cutoff : float
        Max C(i)-N(i+1) distance (A) for two protein residues to be continuous.
    nucleic_cutoff : float
        Max O3'(i)-P(i+1) distance (A) for two nucleic residues to be continuous.
    ca_fallback_cutoff : float
        Max CA(i)-CA(i+1) distance (A) used when protein C/N atoms are missing.
    nucleic_fallback_cutoff : float
        Max C3'(i)-P(i+1) distance (A) used when nucleic O3' is missing.
    single_other_segment : bool
        If True, all non-polymer, non-water, non-ion ("other") residues are
        placed into a single segment. If False (default), they are split into
        separate segments by bonded connected components (one per molecule).

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>`
        A copy with the requested fields set.

    Example
    -------
    >>> newmol = autoSegment(mol, fields=("chain", "segid"))  # doctest: +SKIP
    """
    import networkx as nx

    if isinstance(fields, str):
        fields = (fields,)

    mol = mol.copy()
    sel_mask = mol.atomselect(sel)
    sel_idx = np.where(sel_mask)[0]
    if len(sel_idx) == 0:
        return mol

    cats, residue_idx = _classify_residues(mol, sel_mask)

    # seg_of_res[i] = integer segment index for residue i (filled below)
    seg_of_res = np.full(len(residue_idx), -1, dtype=int)
    seg_idx = -1

    # --- 1. Polymer residues: backbone-distance traversal in file order ---
    prev_i = None
    for i, cat in enumerate(cats):
        if cat not in ("protein", "nucleic"):
            continue
        rep = residue_idx[i][0]
        new_segment = True
        if prev_i is not None:
            prep = residue_idx[prev_i][0]
            same_meta = (
                cats[prev_i] == cat
                and mol.chain[prep] == mol.chain[rep]
                and mol.segid[prep] == mol.segid[rep]
            )
            if same_meta and _polymer_linked(
                mol,
                residue_idx[prev_i],
                residue_idx[i],
                cat,
                protein_cutoff,
                nucleic_cutoff,
                ca_fallback_cutoff,
                nucleic_fallback_cutoff,
            ):
                new_segment = False
        if new_segment:
            seg_idx += 1
            if _logger and prev_i is not None:
                pr, cr = residue_idx[prev_i][0], rep
                logger.info(
                    f"autoSegment: break before "
                    f"{mol.resname[cr]}:{mol.resid[cr]}{mol.insertion[cr]}:{mol.chain[cr]} "
                    f"(after {mol.resname[pr]}:{mol.resid[pr]}{mol.insertion[pr]}:{mol.chain[pr]})"
                )
        seg_of_res[i] = seg_idx
        prev_i = i

    # --- 2. Water: one segment; ions: one segment ---
    ion_seg = -1
    for bucket in ("water", "ion"):
        members = [i for i, c in enumerate(cats) if c == bucket]
        if members:
            seg_idx += 1
            if bucket == "ion":
                ion_seg = seg_idx
            for i in members:
                seg_of_res[i] = seg_idx

    # --- 3. Other: one segment, or split by bonded connected components ---
    other = [i for i, c in enumerate(cats) if c == "other"]
    if other and single_other_segment:
        seg_idx += 1
        for i in other:
            seg_of_res[i] = seg_idx
    elif other:
        other_atoms = np.hstack([residue_idx[i] for i in other])
        submol = mol.copy()
        submol.filter(other_atoms, _logger=False)
        submol.bonds = submol._getBonds()
        submol.bondtype = np.array([], dtype=object)
        # Map each global atom index to its residue position in `cats`
        atom_to_res = {}
        for i in other:
            for a in residue_idx[i]:
                atom_to_res[a] = i
        for comp in nx.connected_components(submol.toGraph(fields=[])):
            seg_idx += 1
            comp_global = other_atoms[list(comp)]
            for a in comp_global:
                seg_of_res[atom_to_res[a]] = seg_idx

    # --- 4. Assign chain/segid using the module's naming machinery ---
    if "chain" in fields:
        mol.chain[sel_mask] = ""
    if "segid" in fields:
        mol.segid[sel_mask] = ""

    def _segid_gen(basename):
        for base in [basename] + SEGID_ALPHABET:
            for k in range(1000):
                yield f"{base}{k}"

    segid_gen = _segid_gen(basename)
    preexisting_chains = set(mol.chain[~sel_mask]) - {""}
    available_chains = [x for x in CHAIN_ALPHABET if x not in preexisting_chains]

    nseg = seg_of_res.max() + 1 if len(seg_of_res) else 0
    for seg in range(nseg):
        res_members = np.where(seg_of_res == seg)[0]
        if len(res_members) == 0:
            continue
        atoms = np.hstack([residue_idx[i] for i in res_members])
        if "chain" in fields:
            mol.chain[atoms] = available_chains[seg % len(available_chains)]
        if "segid" in fields:
            segid = next(segid_gen)
            while segid in mol.segid:
                segid = next(segid_gen)
            mol.segid[atoms] = segid

    # Ions are collapsed into a single segment + chain. Ions that were
    # distinguished only by chain (e.g. 7BTI's five Mg, each resid 401 in
    # chains A-E) then collide on (resid, insertion) within that one chain, and
    # the downstream (resid, insertion, segid) renumbering would fold them into
    # a single residue, silently dropping the duplicates. Give each colliding
    # ion a unique resid so all survive. Scoped to the ion segment only - never
    # renumber polymer residues.
    if ion_seg >= 0:
        # Ions collapse into one segment + chain, so each needs a distinct resid
        # within it: residues distinguished only by chain (e.g. 7BTI's five Mg,
        # each resid 401 in chains A-E) would otherwise be folded into a single
        # residue by the downstream (resid, insertion, segid) renumbering and the
        # duplicates silently dropped. Only when such a collision exists do we
        # renumber the ions 1..N from scratch; otherwise their resids are kept.
        # (Resids may repeat across segments - only within-segment uniqueness
        # matters.)
        members = np.where(seg_of_res == ion_seg)[0]
        keys = [
            (int(mol.resid[residue_idx[i][0]]), str(mol.insertion[residue_idx[i][0]]))
            for i in members
        ]
        if len(set(keys)) != len(keys):
            for new_resid, i in enumerate(members, start=1):
                mol.resid[residue_idx[i]] = new_resid
                mol.insertion[residue_idx[i]] = ""

    return mol


def autoSegment2(
    mol: "Molecule",
    sel: str | np.ndarray = "(protein or resname ACE NME)",
    basename: str = "P",
    fields: tuple = ("segid",),
    residgaps: bool = False,
    residgaptol: int = 1,
    chaingaps: bool = True,
    _logger=True,
) -> "Molecule":
    """Deprecated alias of :func:`autoSegment`.

    Kept as a thin backward-compatible wrapper that forwards to
    :func:`autoSegment`. The ``residgaps``, ``residgaptol`` and ``chaingaps``
    arguments are accepted but ignored: :func:`autoSegment` decides continuity
    from backbone geometry rather than ``resid`` numbering.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule object.
    sel : str or np.ndarray
        Atom selection to segment. A selection string, a boolean mask, or an
        integer index array. Atoms outside the selection keep their
        existing chain/segid.
    basename : str
        Base name for segment ids (e.g. 'P' -> 'P0', 'P1', ...).
    fields : tuple
        Fields to set: any combination of "segid" and "chain".
    residgaps : bool
        Ignored. Accepted only for backward compatibility.
    residgaptol : int
        Ignored. Accepted only for backward compatibility.
    chaingaps : bool
        Ignored. Accepted only for backward compatibility.

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>`
        A copy with the requested fields set, as produced by
        :func:`autoSegment`.
    """
    import warnings

    warnings.warn(
        "autoSegment2 is deprecated and will be removed in a future release; "
        "use autoSegment instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _logger:
        logger.warning(
            "autoSegment2 is deprecated; forwarding to autoSegment. "
            "Update your code to call autoSegment directly."
        )
    return autoSegment(
        mol, sel=sel, basename=basename, fields=fields, _logger=_logger
    )
