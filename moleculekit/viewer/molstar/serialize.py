"""Serialize a Molecule into the MoleculeKitDict shape consumed by the
viewer-frontend (moleculeToCIF.ts)."""

from __future__ import annotations

import numpy as np

_ATOM_FIELDS = (
    "altloc", "atomtype", "beta", "chain", "charge", "element",
    "formalcharge", "insertion", "name", "occupancy", "record",
    "resid", "resname", "segid", "serial",
)


def molecule_to_dict(mol, frame: int = 0) -> dict:
    """Return a JSON-serializable MoleculeKitDict for a single frame.

    Multi-frame trajectories are sent separately as a binary blob via
    /coords/<slot>/<topohash>.
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
