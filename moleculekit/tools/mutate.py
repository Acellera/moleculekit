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
        "MSE": {"axis": ["CG", "SE"], "ref_plane": ["CB", "CG", "SE", "CE"]},
        "MLZ": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
        "MLY": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
        "M3L": {"axis": ["CG", "CD"], "ref_plane": ["CB", "CG", "CD", "CE"]},
    },
    "CHI4": {
        "ARG": {"axis": ["CD", "NE"], "ref_plane": ["CG", "CD", "NE", "CZ"]},
        "LYS": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "MLZ": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "MLY": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
        "M3L": {"axis": ["CD", "CE"], "ref_plane": ["CG", "CD", "CE", "NZ"]},
    },
}

# Modified residues -> parent residue for rotamer library lookup.
# These have their own CIF templates and RESIDUE_ORDER entries, but reuse
# the parent's rotamers since the modifications are at the chain terminus.
_ROTAMER_PARENT = {
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


@functools.lru_cache(maxsize=32)
def _load_template(resname):
    """Load a CIF template for *resname* and return heavy-atom data.

    Uses ``Molecule`` to read the CIF file, then filters to heavy atoms
    that belong to the standard residue definition (``RESIDUE_ORDER``).

    Returns
    -------
    coords : dict
        ``{atom_name: np.ndarray([x, y, z])}`` for heavy atoms only.
    elements : dict
        ``{atom_name: element_symbol}``
    bonds : list of tuple
        ``[(atom1, atom2), ...]`` heavy-atom bonds.
    """
    from moleculekit.molecule import Molecule

    cif_path = os.path.join(_CIF_DIR, f"{resname}.cif")
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"No CIF template for residue {resname} at {cif_path}")

    tmpl = Molecule(cif_path)
    allowed = set(RESIDUE_ORDER.get(resname, []))

    coords = {}
    elements = {}
    idx_to_name = {}
    for i in range(tmpl.numAtoms):
        name = tmpl.name[i]
        if tmpl.element[i] == "H" or name not in allowed:
            continue
        coords[name] = tmpl.coords[i, :, 0].astype(np.float64)
        elements[name] = tmpl.element[i]
        idx_to_name[i] = name

    bonds = []
    for b in tmpl.bonds:
        a1, a2 = int(b[0]), int(b[1])
        if a1 in idx_to_name and a2 in idx_to_name:
            bonds.append((idx_to_name[a1], idx_to_name[a2]))

    return coords, elements, bonds


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


def _build_template_on_backbone(resname, bb_coords):
    """Phase 1: superpose ideal template onto actual backbone N/CA/C.

    Uses the Kabsch algorithm from :func:`moleculekit.align._pp_measure_fit`.

    Parameters
    ----------
    resname : str
    bb_coords : dict
        ``{"N": array, "CA": array, "C": array}``

    Returns
    -------
    placed : dict
        ``{atom_name: np.ndarray([x, y, z])}`` for all heavy atoms.
    elements : dict
    bonds : list of tuple
    """
    from moleculekit.align import _pp_measure_fit

    template_coords, elements, bonds = _load_template(resname)

    P = np.array(
        [template_coords["N"], template_coords["CA"], template_coords["C"]],
        dtype=np.float64,
    )
    Q = np.array(
        [bb_coords["N"], bb_coords["CA"], bb_coords["C"]],
        dtype=np.float64,
    )
    centroidP = P.mean(axis=0)
    centroidQ = Q.mean(axis=0)
    rot, _ = _pp_measure_fit(P - centroidP, Q - centroidQ)

    placed = {}
    for name, coord in template_coords.items():
        placed[name] = (coord - centroidP) @ rot.T + centroidQ

    return placed, elements, bonds


def _apply_chi_angles(coords, resname, chi_values):
    """Rotate side-chain atoms to match the given chi angles.

    Parameters
    ----------
    coords : dict
        ``{atom_name: np.ndarray}`` -- modified **in place** and returned.
    resname : str
    chi_values : tuple
        ``(chi1, chi2, chi3, chi4)`` in **degrees**.

    Returns
    -------
    coords : dict
    """
    order = RESIDUE_ORDER.get(resname)
    if order is None:
        return coords

    for chi_name, chi_target_deg in zip(("CHI1", "CHI2", "CHI3", "CHI4"), chi_values):
        if chi_target_deg == 0.0 and chi_name not in CHI_ANGLES:
            continue
        chi_def = CHI_ANGLES.get(chi_name, {}).get(resname)
        if chi_def is None:
            continue

        ref_atoms = chi_def["ref_plane"]
        if any(a not in coords for a in ref_atoms):
            continue
        current_angle = dihedralAngle(np.array([coords[a] for a in ref_atoms]))
        delta = current_angle - np.deg2rad(chi_target_deg)

        axis_atoms = chi_def["axis"]
        axis_vec = coords[axis_atoms[0]] - coords[axis_atoms[1]]
        pivot = coords[axis_atoms[1]]
        R = rotationMatrix(axis_vec, delta)

        pivot_idx = order.index(axis_atoms[1])
        for atom_name in order[pivot_idx + 1 :]:
            if atom_name in coords:
                coords[atom_name] = R @ (coords[atom_name] - pivot) + pivot

    return coords


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

    new_radii = np.array([VDW_RADII.get(e, 1.7) for e in new_elements])
    surr_radii = np.array([VDW_RADII.get(e, 1.7) for e in surr_elements])

    # sigma_ij = r_i + r_j
    sigma = new_radii[:, None] + surr_radii[None, :]

    diff = new_coords[:, None, :] - surr_coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    dist = np.maximum(dist, 0.1)  # avoid division by zero

    ratio = sigma / dist
    r6 = ratio**6
    energy = (r6 * r6 - r6).sum()
    return energy


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
    frame = mol.frame
    resid = mol.resid[res_idx[0]]
    chain = mol.chain[res_idx[0]]
    insertion = mol.insertion[res_idx[0]]
    segid = mol.segid[res_idx[0]]

    name_map = {}
    for idx in res_idx:
        name_map[mol.name[idx]] = mol.coords[idx, :, frame].astype(np.float64)

    if "N" not in name_map or "CA" not in name_map or "C" not in name_map:
        return None, None

    # Find previous residue's C atom for phi
    phi = None
    chain_mask = (mol.chain == chain) & (mol.segid == segid)
    prev_c_mask = chain_mask & (mol.name == "C") & (mol.resid == resid - 1)
    if insertion != "":
        prev_c_mask = (
            chain_mask
            & (mol.name == "C")
            & (mol.resid == resid)
            & (mol.insertion < insertion)
        )
        if not prev_c_mask.any():
            prev_c_mask = chain_mask & (mol.name == "C") & (mol.resid == resid - 1)
    prev_c_idx = np.where(prev_c_mask)[0]
    if len(prev_c_idx) > 0:
        prev_c = mol.coords[prev_c_idx[-1], :, frame].astype(np.float64)
        phi = np.degrees(
            dihedralAngle(
                np.array([prev_c, name_map["N"], name_map["CA"], name_map["C"]])
            )
        )

    # Find next residue's N atom for psi
    psi = None
    next_n_mask = chain_mask & (mol.name == "N") & (mol.resid == resid + 1)
    if insertion != "":
        next_n_mask = (
            chain_mask
            & (mol.name == "N")
            & (mol.resid == resid)
            & (mol.insertion > insertion)
        )
        if not next_n_mask.any():
            next_n_mask = chain_mask & (mol.name == "N") & (mol.resid == resid + 1)
    next_n_idx = np.where(next_n_mask)[0]
    if len(next_n_idx) > 0:
        next_n = mol.coords[next_n_idx[0], :, frame].astype(np.float64)
        psi = np.degrees(
            dihedralAngle(
                np.array([name_map["N"], name_map["CA"], name_map["C"], next_n])
            )
        )

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


def _select_best_rotamer(
    template_coords,
    template_elements,
    resname,
    rotamers,
    surr_coords,
    surr_elements,
):
    """Try each rotamer and pick the one with the lowest clash energy.

    Parameters
    ----------
    template_coords : dict
        Side-chain atom coords after Phase 1 placement (pre-rotation).
    template_elements : dict
    resname : str
    rotamers : list of tuple
        ``[(prob, chi1, chi2, chi3, chi4), ...]``
    surr_coords : np.ndarray
    surr_elements : list of str

    Returns
    -------
    best_coords : dict
    """
    sc_names = [a for a in RESIDUE_ORDER.get(resname, []) if a not in BACKBONE_ATOMS]

    best_energy = float("inf")
    best_coords = None

    for prob, chi1, chi2, chi3, chi4 in rotamers:
        trial = {k: v.copy() for k, v in template_coords.items()}
        _apply_chi_angles(trial, resname, (chi1, chi2, chi3, chi4))

        sc_coords = np.array([trial[a] for a in sc_names if a in trial])
        sc_elems = [template_elements[a] for a in sc_names if a in trial]

        energy = _score_clashes(sc_coords, sc_elems, surr_coords, surr_elements)
        if energy < best_energy:
            best_energy = energy
            best_coords = trial

    return best_coords



def _get_surrounding_atoms(mol, res_idx, cutoff=8.0):
    """Collect coordinates and elements of atoms surrounding a residue.

    Excludes:
    - The residue's own atoms
    - Backbone atoms of immediately adjacent residues

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
    frame = mol.frame
    resid = mol.resid[res_idx[0]]
    chain = mol.chain[res_idx[0]]
    segid = mol.segid[res_idx[0]]

    res_set = set(res_idx)
    center = mol.coords[res_idx, :, frame].mean(axis=0)

    all_coords = mol.coords[:, :, frame]
    dists = np.sqrt(((all_coords - center) ** 2).sum(axis=1))
    candidate_mask = dists < cutoff

    surr_coords = []
    surr_elements = []
    for i in np.where(candidate_mask)[0]:
        if i in res_set:
            continue
        # Skip backbone atoms of adjacent residues
        if (
            mol.chain[i] == chain
            and mol.segid[i] == segid
            and abs(int(mol.resid[i]) - int(resid)) == 1
            and mol.name[i] in BACKBONE_ATOMS
        ):
            continue
        # Skip hydrogens in surroundings
        if mol.element[i] == "H":
            continue
        surr_coords.append(all_coords[i].astype(np.float64))
        surr_elements.append(mol.element[i])

    if surr_coords:
        return np.array(surr_coords), surr_elements
    return np.empty((0, 3)), []


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
       clashes.

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
        requested name is preserved.  Modified amino acids ``"MSE"``,
        ``"MLZ"``, and ``"MLY"`` are also supported -- they use their own
        CIF templates but the parent's rotamer library entries.
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

    # Extract backbone coordinates
    bb_coords = {}
    for idx in sel_idx:
        if mol.name[idx] in ("N", "CA", "C"):
            bb_coords[mol.name[idx]] = mol.coords[idx, :, frame].astype(np.float64)

    for req in ("N", "CA", "C"):
        if req not in bb_coords:
            raise ValueError(
                f"Backbone atom {req} not found in selected residue. "
                f"Cannot reconstruct side-chain."
            )

    # Warn about disulfide bonds
    if mol.resname[sel_idx[0]] == "CYS":
        for idx in sel_idx:
            if mol.name[idx] == "SG":
                sg_bonds = mol.bonds[(mol.bonds == idx).any(axis=1)]
                for bond in sg_bonds:
                    partner = bond[0] if bond[1] == idx else bond[1]
                    if mol.name[partner] == "SG" and mol.resname[partner] == "CYS":
                        logger.warning(
                            f"Residue {chain}:{resid} CYS is involved in a "
                            f"disulfide bond. Mutating it will break the bond."
                        )

    # ── Phase 1: Template superposition ──────────────────────────────
    placed_coords, placed_elements, placed_bonds = _build_template_on_backbone(
        baseres, bb_coords
    )

    # ── Phase 2: Rotamer selection ───────────────────────────────────
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
                placed_coords = _select_best_rotamer(
                    placed_coords,
                    placed_elements,
                    baseres,
                    rotamers,
                    surr_coords,
                    surr_elements,
                )
            elif rotamer_mode == "first":
                _apply_chi_angles(
                    placed_coords,
                    baseres,
                    rotamers[0][1:],
                )
            elif rotamer_mode == "random":
                probs = np.array([r[0] for r in rotamers])
                probs /= probs.sum()
                chosen = rotamers[np.random.choice(len(rotamers), p=probs)]
                _apply_chi_angles(placed_coords, baseres, chosen[1:])
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

    # Refresh selection after removal
    sel_mask_new = (
        (mol.chain == chain)
        & (mol.resid == resid)
        & (mol.insertion == insertion)
        & (mol.segid == segid)
    )
    sel_idx_new = np.where(sel_mask_new)[0]

    # ── Insert new side-chain atoms ──────────────────────────────────
    if not sc_names:
        return

    from moleculekit.molecule import Molecule as Mol

    n_new = len(sc_names)
    new_mol = Mol()
    new_mol.empty(n_new)

    for i, atom_name in enumerate(sc_names):
        new_mol.record[i] = "ATOM"
        new_mol.name[i] = atom_name
        new_mol.resname[i] = newres
        new_mol.resid[i] = resid
        new_mol.chain[i] = chain
        new_mol.insertion[i] = insertion
        new_mol.segid[i] = segid
        new_mol.element[i] = placed_elements.get(atom_name, atom_name[0])
        new_mol.occupancy[i] = 1.0
        new_mol.beta[i] = 0.0
        new_mol.altloc[i] = ""
        new_mol.atomtype[i] = ""

    new_mol.coords = np.zeros((n_new, 3, mol.numFrames), dtype=np.float32)
    for i, atom_name in enumerate(sc_names):
        if atom_name in placed_coords:
            new_mol.coords[i, :, frame] = placed_coords[atom_name].astype(np.float32)

    # Insert right after the existing backbone atoms of this residue
    insert_pos = sel_idx_new[-1] + 1 if len(sel_idx_new) > 0 else mol.numAtoms
    mol.insert(new_mol, insert_pos)

    # ── Phase 3: Optional minimization ───────────────────────────────
    if minimize:
        new_indices = set(range(insert_pos, insert_pos + n_new))
        from moleculekit.openmmtools import minimize_soft_potential

        minimize_soft_potential(mol, new_indices, restrain_bonded=False)
