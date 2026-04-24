# (c) 2015-2025 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""Side-chain reconstruction for residue mutation.

Rotamer library data from:
  Shapovalov & Dunbrack (2011) "A Smoothed Backbone-Dependent Rotamer Library
  for Proteins Derived from Adaptive Kernel Density Estimates and Regressions."
  Structure 19(6):844-858.  Licensed under Creative Commons CC BY 4.0.
"""

import gzip
import os
import functools
import logging

import numpy as np

from moleculekit.dihedral import dihedralAngle
from moleculekit.util import rotationMatrix
from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)

_SHARE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "share")
_CIF_DIR = os.path.join(_SHARE_DIR, "residue_cifs")
_ROTAMER_FILE = os.path.join(_SHARE_DIR, "better_5_rotamers.gz")

BACKBONE_ATOMS = {"N", "CA", "C", "O"}

RESIDUE_ORDER = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
    ],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    # Non-standard natural amino acids (21st and 22nd)
    "SEC": ["N", "CA", "C", "O", "CB", "SE"],
    "PYL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "C2",
        "O2",
        "CA2",
        "N2",
        "CE2",
        "CD2",
        "CG2",
        "CB2",
    ],
    # Modified amino acids
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
    "MLZ": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CM"],
    "MLY": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CH1", "CH2"],
    "M3L": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CM1", "CM2", "CM3"],
    "SEP": ["N", "CA", "C", "O", "CB", "OG", "P", "O1P", "O2P", "O3P"],
    "TPO": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "P", "O1P", "O2P", "O3P"],
    "PTR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "P",
        "O1P",
        "O2P",
        "O3P",
    ],
}

# Chi angle definitions: for each chi angle, the 4-atom dihedral and the
# rotation axis.  Atoms *after* axis[1] in RESIDUE_ORDER are rotated.
CHI_ANGLES = {
    "CHI1": {
        "CYS": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "SG"]},
        "ASP": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "SER": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "OG"]},
        "GLN": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "LYS": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "ILE": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG1"]},
        "PRO": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "THR": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "OG1"]},
        "PHE": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "ASN": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "HIS": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "LEU": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "ARG": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "TRP": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "VAL": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG1"]},
        "GLU": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "TYR": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "MET": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "SEC": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "SE"]},
        "PYL": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "MSE": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "MLZ": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "MLY": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "M3L": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
        "SEP": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "OG"]},
        "TPO": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "OG1"]},
        "PTR": {"axis": ["CA", "CB"], "ref_plane": ["N", "CA", "CB", "CG"]},
    },
    "CHI2": {
        "ASP": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "OD1"]},
        "GLN": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "LYS": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "ILE": {"axis": ["CB", "CG1"], "ref_plane": ["CA", "CB", "CG1", "CD1"]},
        "PRO": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "PHE": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD1"]},
        "ASN": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "OD1"]},
        "HIS": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "ND1"]},
        "LEU": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD1"]},
        "ARG": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "TRP": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD1"]},
        "GLU": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "TYR": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD1"]},
        "MET": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "SD"]},
        "PYL": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "MSE": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "SE"]},
        "MLZ": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "MLY": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "M3L": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD"]},
        "PTR": {"axis": ["CB", "CG"], "ref_plane": ["CA", "CB", "CG", "CD1"]},
    },
    "CHI3": {
        "ARG": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "NE"]},
        "GLN": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "OE1"]},
        "GLU": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "OE1"]},
        "LYS": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
        "MET": {"axis": ["CG", "SD"], "ref_plane": ["CB", "CG", "SD", "CE"]},
        "PYL": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
        "MSE": {"axis": ["CG", "SE"], "ref_plane": ["CB", "CG", "SE", "CE"]},
        "MLZ": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
        "MLY": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
        "M3L": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
    },
    "CHI4": {
        "ARG": {"axis": ["CD", "NE"], "ref_plane": ["CG", "CD", "NE", "CZ"]},
        "LYS": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "PYL": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "MLZ": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "MLY": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "M3L": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
    },
}

# Modified residues -> parent residue for rotamer library lookup.
# These have their own CIF templates and RESIDUE_ORDER entries, but reuse
# the parent's rotamers since the modifications are at the chain terminus.
_ROTAMER_PARENT = {
    "SEC": "CYS",
    "PYL": "LYS",
    "MSE": "MET",
    "MLZ": "LYS",
    "MLY": "LYS",
    "M3L": "LYS",
    "SEP": "SER",
    "TPO": "THR",
    "PTR": "TYR",
}

# Per-element VdW radii (Angstroms) for clash scoring
VDW_RADII = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
    "P": 1.8,
    "H": 1.2,
    "SE": 1.9,
}

# ──────────────────────────────────────────────────────────────────────
# CIF template loading
# ──────────────────────────────────────────────────────────────────────


# Map non-standard residue names in the library to standard names
_ROTLIB_RESNAME_MAP = {
    "CYH": "CYS",
    "CYD": "CYS",
    "CPR": "PRO",
    "TPR": "PRO",
}


@functools.lru_cache(maxsize=1)
def _load_rotamer_library():
    """Load the Dunbrack backbone-dependent rotamer library.

    Returns
    -------
    dict
        ``{resname: {(phi, psi): [(prob, chi1, chi2, chi3, chi4), ...]}}``
        Sorted by descending probability within each (phi, psi) bin.
    """
    lib = {}
    opener = gzip.open if _ROTAMER_FILE.endswith(".gz") else open
    with opener(_ROTAMER_FILE, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            resname = parts[0]
            resname = _ROTLIB_RESNAME_MAP.get(resname, resname)
            phi = int(parts[1])
            psi = int(parts[2])
            prob = float(parts[3])
            chi1 = float(parts[4])
            chi2 = float(parts[5])
            chi3 = float(parts[6])
            chi4 = float(parts[7])
            lib.setdefault(resname, {}).setdefault((phi, psi), []).append(
                (prob, chi1, chi2, chi3, chi4)
            )
    # Sort each bin by probability descending
    for res in lib.values():
        for key in res:
            res[key].sort(key=lambda x: x[0], reverse=True)
    return lib


def _build_template_on_backbone(resname, res: Molecule):
    """Superpose ideal template onto the backbone N/CA/C of a residue.

    Parameters
    ----------
    resname : str
        The name of the residue we want to mutate to.
    res : Molecule
        The residue whose backbone we want to align the template to.

    Returns
    -------
    tmpl : Molecule
        The template molecule (heavy atoms only) aligned to the backbone
        of ``res``.
    """
    cif_path = os.path.join(_CIF_DIR, f"{resname}.cif")
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"No CIF template for residue {resname} at {cif_path}")

    tmpl = Molecule(cif_path)

    allowed = set(RESIDUE_ORDER.get(resname, []))
    keep = (tmpl.element != "H") & np.isin(tmpl.name, list(allowed))
    tmpl.filter(keep, _logger=False)

    tmpl_idx = np.array(
        [np.where(tmpl.name == a)[0][0] for a in ("N", "CA", "C")], dtype=np.uint32
    )
    res_idx = np.array(
        [np.where(res.name == a)[0][0] for a in ("N", "CA", "C")], dtype=np.uint32
    )
    tmpl.align(tmpl_idx, refmol=res, refsel=res_idx, _logger=False)

    return tmpl


def _apply_chi_angles(tmpl, resname, chi_values):
    """Rotate side-chain atoms of ``tmpl`` to match the given chi angles.

    Parameters
    ----------
    tmpl : Molecule
        Template molecule, modified **in place** and returned. Coordinates
        of the current ``frame`` are rotated.
    resname : str
    chi_values : tuple
        ``(chi1, chi2, chi3, chi4)`` in **degrees**.

    Returns
    -------
    tmpl : Molecule
    """
    order = RESIDUE_ORDER.get(resname)
    if order is None:
        return tmpl

    frame = tmpl.frame
    name_to_idx = {n: i for i, n in enumerate(tmpl.name)}

    for chi_name, chi_target_deg in zip(("CHI1", "CHI2", "CHI3", "CHI4"), chi_values):
        chi_def = CHI_ANGLES.get(chi_name, {}).get(resname)
        if chi_def is None:
            continue

        ref_atoms = chi_def["ref_plane"]
        if any(a not in name_to_idx for a in ref_atoms):
            continue
        ref_coords = np.array(
            [tmpl.coords[name_to_idx[a], :, frame] for a in ref_atoms],
            dtype=np.float64,
        )
        current_angle = dihedralAngle(ref_coords)
        delta = current_angle - np.deg2rad(chi_target_deg)

        axis_atoms = chi_def["axis"]
        pivot = tmpl.coords[name_to_idx[axis_atoms[1]], :, frame].astype(np.float64)
        axis_vec = (
            tmpl.coords[name_to_idx[axis_atoms[0]], :, frame].astype(np.float64)
            - pivot
        )
        R = rotationMatrix(axis_vec, delta)

        pivot_order_idx = order.index(axis_atoms[1])
        to_rotate = [
            a for a in order[pivot_order_idx + 1 :] if a in name_to_idx
        ]
        if not to_rotate:
            continue
        tmpl.rotateBy(R, center=pivot, sel=np.isin(tmpl.name, to_rotate))

    return tmpl


def _score_clashes(new_coords, new_elements, surr_coords, surr_elements):
    """Score VdW clash energy between new side-chain atoms and surroundings.

    Uses a Lennard-Jones-like potential: ``(sigma/r)^12 - (sigma/r)^6``
    where ``sigma = r_vdw_i + r_vdw_j``.

    Parameters
    ----------
    new_coords : np.ndarray, shape (M, 3)
    new_elements : list of str, length M
    surr_coords : np.ndarray, shape (K, 3)
    surr_elements : list of str, length K

    Returns
    -------
    float
    """
    if len(new_coords) == 0 or len(surr_coords) == 0:
        return 0.0
    from moleculekit.distance import cdist

    new_radii = np.array([VDW_RADII.get(e, 1.7) for e in new_elements])
    surr_radii = np.array([VDW_RADII.get(e, 1.7) for e in surr_elements])
    sigma = new_radii[:, None] + surr_radii[None, :]

    dist = np.maximum(cdist(new_coords, surr_coords), 0.1)
    ratio = sigma / dist
    r6 = ratio**6
    return float((r6 * r6 - r6).sum())


def _compute_phi_psi(mol, res_idx):
    """Compute backbone phi/psi angles for a residue.

    Parameters
    ----------
    mol : Molecule
    res_idx : np.ndarray
        Atom indices for the residue.

    Returns
    -------
    phi : float or None
        In degrees, or None if N-terminal.
    psi : float or None
        In degrees, or None if C-terminal.
    """
    resid = mol.resid[res_idx[0]]
    chain = mol.chain[res_idx[0]]
    insertion = mol.insertion[res_idx[0]]
    segid = mol.segid[res_idx[0]]

    name_to_idx = {mol.name[i]: int(i) for i in res_idx}
    if not {"N", "CA", "C"}.issubset(name_to_idx):
        return None, None
    n_idx, ca_idx, c_idx = name_to_idx["N"], name_to_idx["CA"], name_to_idx["C"]

    chain_mask = (mol.chain == chain) & (mol.segid == segid)

    def _neighbor_idx(target_name, prev):
        resid_offset = -1 if prev else 1
        mask = chain_mask & (mol.name == target_name) & (mol.resid == resid + resid_offset)
        if insertion != "":
            ins_cmp = mol.insertion < insertion if prev else mol.insertion > insertion
            ins_mask = chain_mask & (mol.name == target_name) & (mol.resid == resid) & ins_cmp
            if ins_mask.any():
                mask = ins_mask
        idx = np.where(mask)[0]
        if not len(idx):
            return None
        return int(idx[-1] if prev else idx[0])

    phi = None
    prev_c_idx = _neighbor_idx("C", prev=True)
    if prev_c_idx is not None:
        phi = np.degrees(mol.getDihedral([prev_c_idx, n_idx, ca_idx, c_idx]))

    psi = None
    next_n_idx = _neighbor_idx("N", prev=False)
    if next_n_idx is not None:
        psi = np.degrees(mol.getDihedral([n_idx, ca_idx, c_idx, next_n_idx]))

    return phi, psi


def _snap_to_bin(angle, step=10):
    """Round an angle (degrees) to the nearest bin center."""
    if angle is None:
        return 0
    binned = int(round(angle / step)) * step
    if binned > 180:
        binned -= 360
    if binned < -180:
        binned += 360
    return binned


def _select_best_rotamer(tmpl, resname, rotamers, surr_coords, surr_elements):
    """Try each rotamer and leave ``tmpl`` holding the lowest-clash pose.

    Parameters
    ----------
    tmpl : Molecule
        Backbone-aligned template (pre-rotation), modified **in place**.
    resname : str
    rotamers : list of tuple
        ``[(prob, chi1, chi2, chi3, chi4), ...]``
    surr_coords : np.ndarray
    surr_elements : list of str

    Returns
    -------
    tmpl : Molecule
    """
    frame = tmpl.frame
    sc_mask = ~np.isin(tmpl.name, list(BACKBONE_ATOMS))
    sc_elems = tmpl.element[sc_mask].tolist()

    original_coords = tmpl.coords.copy()

    best_energy = float("inf")
    best_rotamer = None

    for rot in rotamers:
        tmpl.coords[...] = original_coords
        _apply_chi_angles(tmpl, resname, rot[1:])

        sc_coords = tmpl.coords[sc_mask, :, frame].astype(np.float64)
        energy = _score_clashes(sc_coords, sc_elems, surr_coords, surr_elements)
        if energy < best_energy:
            best_energy = energy
            best_rotamer = rot

    tmpl.coords[...] = original_coords
    if best_rotamer is not None:
        _apply_chi_angles(tmpl, resname, best_rotamer[1:])

    return tmpl


def _get_surrounding_atoms(mol, res_idx, cutoff=8.0):
    """Collect coordinates and elements of atoms surrounding a residue.

    An atom is a surrounding if its distance to the nearest residue atom
    is below ``cutoff``. Excludes the residue itself, hydrogens, and
    backbone atoms of the two adjacent residues (same chain/segid).

    Parameters
    ----------
    mol : Molecule
    res_idx : np.ndarray
    cutoff : float
        Distance cutoff in Angstroms.

    Returns
    -------
    surr_coords : np.ndarray, shape (K, 3)
    surr_elements : list of str
    """
    from moleculekit.distance import cdist

    frame = mol.frame
    resid = int(mol.resid[res_idx[0]])
    chain = mol.chain[res_idx[0]]
    segid = mol.segid[res_idx[0]]

    all_coords = mol.coords[:, :, frame]
    min_dists = cdist(all_coords[res_idx], all_coords).min(axis=0)

    exclude = np.zeros(mol.numAtoms, dtype=bool)
    exclude[res_idx] = True
    exclude |= mol.element == "H"
    exclude |= (
        (mol.chain == chain)
        & (mol.segid == segid)
        & ((mol.resid == resid - 1) | (mol.resid == resid + 1))
        & np.isin(mol.name, list(BACKBONE_ATOMS))
    )

    mask = (min_dists < cutoff) & ~exclude
    return all_coords[mask].astype(np.float64), mol.element[mask].tolist()


def _merge_template_bonds(mol, bb_sc_connections, chain, resid, insertion, segid):
    """Add backbone-to-sidechain bonds after inserting new side-chain atoms.

    Intra-sidechain bonds are merged automatically by ``Molecule.insert``;
    the backbone-sidechain connections (e.g. ``CA``-``CB``) are dropped
    when the template is filtered to side-chain atoms and must be
    re-added manually.
    """
    if not bb_sc_connections:
        return

    res_mask = (
        (mol.chain == chain)
        & (mol.resid == resid)
        & (mol.insertion == insertion)
        & (mol.segid == segid)
    )
    new_bonds = []
    for bb_name, sc_name in bb_sc_connections:
        bb_idx = np.where(res_mask & (mol.name == bb_name))[0]
        sc_idx = np.where(res_mask & (mol.name == sc_name))[0]
        if len(bb_idx) and len(sc_idx):
            new_bonds.append([bb_idx[0], sc_idx[0]])
    if not new_bonds:
        return

    new_bonds = np.array(new_bonds, dtype=mol.bonds.dtype)
    mol.bonds = np.append(mol.bonds, new_bonds, axis=0)
    mol.bondtype = np.append(
        mol.bondtype, np.array(["1"] * len(new_bonds), dtype=object)
    )


def mutate_residue(mol, sel, newres, rotamer_mode="best", minimize=False):
    """Mutate a residue, fully reconstructing the side-chain.

    The mutation proceeds in up to three phases:

    1. **Kabsch superposition** -- the ideal template for *newres* is aligned
       onto the existing backbone (N, CA, C) atoms.
    2. **Rotamer selection** -- the Dunbrack backbone-dependent rotamer library
       is queried using the residue's phi/psi angles, and the best rotamer
       (lowest VdW clash energy with surroundings) is selected.
    3. **Optional OpenMM minimization** -- if *minimize* is True and OpenMM is
       installed, a soft-potential energy minimization gently resolves remaining
       clashes.  Template bonds for the new residue are merged into ``mol.bonds``
       after insertion so minimization (and other tools) see correct connectivity.

    Parameters
    ----------
    mol : Molecule
        The molecule to mutate (modified **in place**).
    sel : str
        Atom selection string identifying a single residue.
    newres : str
        3-letter code of the target residue (e.g. ``"ARG"``).  Protonation
        variants such as ``"HID"``, ``"HIE"``, ``"HIP"``, ``"CYX"``,
        ``"ASH"``, ``"GLH"``, ``"LYN"`` etc. are also accepted -- the
        heavy-atom geometry is taken from the parent residue and the
        requested name is preserved.  The non-standard natural amino acids
        ``"SEC"`` (selenocysteine) and ``"PYL"`` (pyrrolysine) and the
        modified amino acids ``"MSE"``, ``"MLZ"``, ``"MLY"``, ``"M3L"``,
        ``"SEP"``, ``"TPO"`` and ``"PTR"`` are also supported -- they use
        their own CIF templates but reuse their parent residue's rotamer
        library entries.
    rotamer_mode : str, optional
        ``"best"`` (lowest clash energy, default), ``"first"`` (highest
        probability), or ``"random"`` (sampled by probability).
    minimize : bool, optional
        If True, run soft-potential OpenMM minimization after rotamer
        placement.  Requires OpenMM.  Default False.
    """
    from moleculekit.residues import ORIGINAL_RESIDUE_NAME_TABLE

    # Resolve protonation variants (HID->HIS, CYX->CYS, etc.) to the
    # base residue for template geometry and rotamer lookup.
    baseres = ORIGINAL_RESIDUE_NAME_TABLE.get(newres, newres)
    if baseres not in RESIDUE_ORDER:
        raise ValueError(
            f"Unknown target residue '{newres}'. "
            f"Supported: {sorted(RESIDUE_ORDER.keys())} and their "
            f"protonation variants."
        )

    # Modified residues (MSE, MLZ, MLY) have their own templates/chi defs
    # but reuse the parent residue's rotamer library entries.
    rotamer_res = _ROTAMER_PARENT.get(baseres, baseres)

    sel_mask = mol.atomselect(sel, strict=True)
    sel_idx = np.where(sel_mask)[0]
    if len(sel_idx) == 0:
        raise ValueError(f"Selection '{sel}' matched no atoms.")

    # Verify single residue
    unique_res = set(
        zip(mol.chain[sel_idx], mol.resid[sel_idx], mol.insertion[sel_idx])
    )
    if len(unique_res) != 1:
        raise ValueError(
            f"Selection '{sel}' must match exactly one residue, got {len(unique_res)}."
        )

    frame = mol.frame
    resid = mol.resid[sel_idx[0]]
    chain = mol.chain[sel_idx[0]]
    insertion = mol.insertion[sel_idx[0]]
    segid = mol.segid[sel_idx[0]]

    # Verify the residue has a full backbone to align the template onto
    res_names = set(mol.name[sel_idx])
    for req in ("N", "CA", "C"):
        if req not in res_names:
            raise ValueError(
                f"Backbone atom {req} not found in selected residue. "
                f"Cannot reconstruct side-chain."
            )

    # Warn about disulfide bonds (covers CYS/CYX/CYM). PDB-loaded
    # structures often miss inter-residue S-S bonds, so include guessed
    # bonds for the neighbor lookup.
    cys_aliases = {"CYS", "CYX", "CYM"}
    if mol.resname[sel_idx[0]] in cys_aliases:
        all_bonds = mol._getBonds(guessBonds=True)
        for sg in sel_idx[mol.name[sel_idx] == "SG"]:
            for nb in mol.getNeighbors(int(sg), bonds=all_bonds):
                if mol.name[nb] == "SG" and mol.resname[nb] in cys_aliases:
                    logger.warning(
                        f"Residue {chain}:{resid} is involved in a disulfide "
                        f"bond. Mutating it will break the bond."
                    )
                    break

    # Phase 1: Template superposition
    res = mol.copy(sel=sel_mask)
    tmpl = _build_template_on_backbone(baseres, res)

    # Phase 2: Rotamer selection
    sc_names = [a for a in RESIDUE_ORDER[baseres] if a not in BACKBONE_ATOMS]

    if sc_names and baseres not in ("ALA", "GLY"):
        phi, psi = _compute_phi_psi(mol, sel_idx)
        if phi is None or psi is None:
            logger.warning(
                f"Phi/psi angles not fully defined for residue "
                f"{chain}:{resid} (terminal residue?). "
                f"Using phi={phi}, psi={psi} (None → 0 for library lookup)."
            )
        phi_bin = _snap_to_bin(phi)
        psi_bin = _snap_to_bin(psi)

        rotlib = _load_rotamer_library()
        rotamers = rotlib.get(rotamer_res, {}).get((phi_bin, psi_bin), [])

        if not rotamers:
            logger.warning(
                f"No rotamers found for {rotamer_res} at phi={phi_bin}, psi={psi_bin}. "
                f"Using template geometry."
            )
        else:
            surr_coords, surr_elements = _get_surrounding_atoms(mol, sel_idx)

            if rotamer_mode == "best":
                tmpl = _select_best_rotamer(
                    tmpl, baseres, rotamers, surr_coords, surr_elements
                )
            elif rotamer_mode == "first":
                _apply_chi_angles(tmpl, baseres, rotamers[0][1:])
            elif rotamer_mode == "random":
                probs = np.array([r[0] for r in rotamers])
                probs /= probs.sum()
                chosen = rotamers[np.random.choice(len(rotamers), p=probs)]
                _apply_chi_angles(tmpl, baseres, chosen[1:])
            else:
                raise ValueError(
                    f"Unknown rotamer_mode '{rotamer_mode}'. "
                    f"Use 'best', 'first', or 'random'."
                )

    # ── Remove old side-chain and rename residue ─────────────────────
    mol.resname[sel_idx] = newres
    keep_backbone = np.isin(mol.name[sel_idx], list(BACKBONE_ATOMS))
    to_remove = sel_idx[~keep_backbone]
    if len(to_remove):
        mol.remove(to_remove, _logger=False)

    if not sc_names:
        return

    # Record backbone<->sidechain connecting bonds (e.g. CA-CB) so we can
    # re-add them in ``mol`` after filtering the template down to side-chain.
    sc_mask_tmpl = ~np.isin(tmpl.name, list(BACKBONE_ATOMS))
    bb_sc_connections = []
    for bb_name in BACKBONE_ATOMS:
        bb_where = np.where(tmpl.name == bb_name)[0]
        if not len(bb_where):
            continue
        for nb in tmpl.getNeighbors(int(bb_where[0])):
            if sc_mask_tmpl[nb]:
                bb_sc_connections.append((bb_name, tmpl.name[nb]))

    # Keep only the side-chain of the template for insertion
    sc_tmpl = tmpl.copy()
    sc_tmpl.filter(sc_mask_tmpl, _logger=False)
    sc_tmpl.record[:] = "ATOM"
    sc_tmpl.resname[:] = newres
    sc_tmpl.resid[:] = resid
    sc_tmpl.chain[:] = chain
    sc_tmpl.insertion[:] = insertion
    sc_tmpl.segid[:] = segid
    sc_tmpl.occupancy[:] = 1.0
    sc_tmpl.beta[:] = 0.0
    sc_tmpl.altloc[:] = ""
    sc_tmpl.atomtype[:] = ""

    # Match the parent molecule's frame count (CIF templates have 1 frame)
    if sc_tmpl.numFrames != mol.numFrames:
        new_coords = np.zeros(
            (sc_tmpl.numAtoms, 3, mol.numFrames), dtype=np.float32
        )
        new_coords[:, :, frame] = sc_tmpl.coords[:, :, 0]
        sc_tmpl.coords = new_coords

    # Refresh selection and insert right after the existing backbone atoms
    sel_mask_new = (
        (mol.chain == chain)
        & (mol.resid == resid)
        & (mol.insertion == insertion)
        & (mol.segid == segid)
    )
    sel_idx_new = np.where(sel_mask_new)[0]
    insert_pos = sel_idx_new[-1] + 1 if len(sel_idx_new) > 0 else mol.numAtoms

    mol.insert(sc_tmpl, insert_pos)
    _merge_template_bonds(mol, bb_sc_connections, chain, resid, insertion, segid)

    # ── Phase 3: Optional minimization ───────────────────────────────
    if minimize:
        new_indices = set(range(insert_pos, insert_pos + sc_tmpl.numAtoms))
        from moleculekit.openmmtools import minimize_soft_potential

        minimize_soft_potential(mol, new_indices)
