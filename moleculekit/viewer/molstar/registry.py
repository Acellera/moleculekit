"""Slot registry and topology hashing for the molstar viewer."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


_TOPO_FIELDS = (
    "element", "name", "resname", "resid", "chain", "segid",
    "insertion", "altloc", "record", "atomtype", "serial",
    "bondtype", "formalcharge",
)


def topo_hash(mol) -> str:
    """Stable sha1 hex over the topology-defining fields of a Molecule.

    Skipped: coords, beta, occupancy, charge (partial charge) — these can
    vary without a structural change to the rendered scene.
    """
    h = hashlib.sha1()
    for field in _TOPO_FIELDS:
        arr = getattr(mol, field)
        if arr.dtype == object:
            h.update(b"\x00".join(str(x).encode("utf-8") for x in arr.tolist()))
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(b"|")
    h.update(np.ascontiguousarray(mol.bonds, dtype=np.int64).tobytes())
    return h.hexdigest()


def coords_to_bytes(mol) -> bytes:
    """Return all frames as little-endian float32 bytes.

    Layout: per frame, the flat [x0..xN, y0..yN, z0..zN] vector matching
    the per-frame layout used by molecule_to_dict.
    """
    num_atoms = mol.coords.shape[0]
    num_frames = mol.coords.shape[2]
    out = np.empty((num_frames, 3, num_atoms), dtype="<f4")
    for f in range(num_frames):
        out[f] = mol.coords[:, :, f].T
    return out.tobytes()


@dataclass
class Slot:
    uuid: str
    mol_ref: Any   # a Molecule (live, mutable reference held by the user)
    snapshot: Any  # a Molecule (mol.copy() snapshot for diffing)
    topo_hash: str
    label: str
    visible: bool = True


class Registry:
    """Thread-safe slot table for the molstar viewer."""

    def __init__(self):
        self._lock = threading.Lock()
        self.slots: dict[str, Slot] = {}

    def register(self, mol) -> str:
        with self._lock:
            for slot in self.slots.values():
                if slot.mol_ref is mol:
                    return slot.uuid
            uid = uuid.uuid4().hex[:6].upper()
            self.slots[uid] = Slot(
                uuid=uid,
                mol_ref=mol,
                snapshot=mol.copy(),
                topo_hash=topo_hash(mol),
                label=getattr(mol, "viewname", None) or f"mol_{len(self.slots) + 1}",
            )
            return uid

    def remove(self, uid: str) -> None:
        with self._lock:
            self.slots.pop(uid, None)

    def diff_and_snapshot(self) -> list[tuple[str, str, dict]]:
        """Diff each slot's live mol against its snapshot.

        Returns a list of (kind, slot_uuid, payload_hints) tuples. Updates
        each slot's snapshot/topo_hash in place after diffing. Empty list
        when nothing changed.
        """
        with self._lock:
            uids = list(self.slots.keys())

        changes: list[tuple[str, str, dict]] = []
        for uid in uids:
            with self._lock:
                slot = self.slots.get(uid)
                if slot is None:
                    continue
                mol = slot.mol_ref

            new_topo = topo_hash(mol)
            if new_topo != slot.topo_hash:
                with self._lock:
                    if uid in self.slots:
                        slot.snapshot = mol.copy()
                        slot.topo_hash = new_topo
                        changes.append(("topology", uid, {"new_topo_hash": new_topo}))
            elif not np.array_equal(slot.snapshot.coords, mol.coords):
                with self._lock:
                    if uid in self.slots:
                        slot.snapshot = mol.copy()
                        changes.append(("coords", uid, {"topo_hash": slot.topo_hash}))
        return changes
