"""Serialize a Molecule into the MoleculeKitDict shape consumed by the
viewer-frontend (moleculeToCIF.ts)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

_ATOM_FIELDS = (
    "altloc", "atomtype", "beta", "chain", "charge", "element",
    "formalcharge", "insertion", "name", "occupancy", "record",
    "resid", "resname", "segid", "serial",
)


def molecule_to_dict(mol: "Molecule", frame: int = 0) -> dict:
    """Return a JSON-serializable MoleculeKitDict for a single frame.

    The returned dict holds the per-atom fields, bonds/bond types and a flat
    coordinate vector for the requested frame, plus the atom and frame counts.
    Multi-frame trajectories are sent separately as a binary blob via
    /coords/<slot>/<topohash>.

    Parameters
    ----------
    mol : Molecule
        The molecule to serialize.
    frame : int, optional
        The coordinate frame to include in ``coords``. Defaults to 0.

    Returns
    -------
    out : dict
        A JSON-serializable dict with the per-atom topology fields, ``bonds``,
        ``bondtype``, a flat float32 ``coords`` list for ``frame`` (ordered
        ``[x0..xN, y0..yN, z0..zN]``), and integer ``numAtoms`` / ``numFrames``.
    """
    out: dict = {}
    for field in _ATOM_FIELDS:
        out[field] = getattr(mol, field).tolist()

    out["bonds"] = (
        mol.bonds.astype(np.int64, copy=False).tolist() if mol.bonds.size else []
    )
    out["bondtype"] = mol.bondtype.tolist() if mol.bondtype.size else []

    num_atoms = mol.coords.shape[0]
    num_frames = mol.coords.shape[2]
    flat = mol.coords[:, :, frame].T.reshape(-1).astype(np.float32)
    out["coords"] = flat.tolist()
    out["numAtoms"] = int(num_atoms)
    out["numFrames"] = int(num_frames)
    return out
