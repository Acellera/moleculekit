"""Slot registry and topology hashing for the molstar viewer."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


_TOPO_FIELDS = (
    "element", "name", "resname", "resid", "chain", "segid",
    "insertion", "altloc", "record", "atomtype", "serial",
    "bondtype", "formalcharge",
)


def topo_hash(mol: "Molecule") -> str:
    """Stable sha1 hex over the topology-defining fields of a Molecule.

    Skipped: coords, beta, occupancy, charge (partial charge) — these can
    vary without a structural change to the rendered scene.

    Parameters
    ----------
    mol : Molecule
        The molecule to hash.

    Returns
    -------
    digest : str
        The hex sha1 digest of the topology fields and bonds.
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


def coords_to_bytes(mol: "Molecule") -> bytes:
    """Return all frames as little-endian float32 bytes.

    Layout: per frame, the flat [x0..xN, y0..yN, z0..zN] vector matching
    the per-frame layout used by molecule_to_dict.

    Parameters
    ----------
    mol : Molecule
        The molecule whose coordinates are encoded.

    Returns
    -------
    blob : bytes
        Little-endian float32 bytes for all frames, frame-major.
    """
    num_atoms = mol.coords.shape[0]
    num_frames = mol.coords.shape[2]
    out = np.empty((num_frames, 3, num_atoms), dtype="<f4")
    for f in range(num_frames):
        out[f] = mol.coords[:, :, f].T
    return out.tobytes()


@dataclass
class Slot:
    """A single registered molecule tracked by the viewer Registry.

    Attributes
    ----------
    uuid : str
        Short uppercase hex identifier for the slot.
    mol_ref : Molecule
        The live, mutable Molecule reference held by the user.
    snapshot : Molecule
        A ``mol.copy()`` snapshot used for diffing against ``mol_ref``.
    topo_hash : str
        The topology hash of ``snapshot`` (see ``topo_hash``).
    label : str
        Human-readable label shown in the viewer.
    visible : bool
        Whether the slot is currently shown. Defaults to True.
    """

    uuid: str
    mol_ref: Any   # a Molecule (live, mutable reference held by the user)
    snapshot: Any  # a Molecule (mol.copy() snapshot for diffing)
    topo_hash: str
    label: str
    visible: bool = True


class Registry:
    """Thread-safe slot table for the molstar viewer.

    Maps slot uuids to ``Slot`` entries, each holding a live Molecule reference
    and a snapshot for change detection. All mutating operations are guarded by
    an internal lock so the monitor thread and request handlers can share it.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.slots: dict[str, Slot] = {}

    def register(self, mol: "Molecule") -> str:
        """Register a molecule and return its slot uuid.

        If ``mol`` is already registered (same object identity), its existing
        uuid is returned and no new slot is created.

        Parameters
        ----------
        mol : Molecule
            The live molecule to track.

        Returns
        -------
        uid : str
            The slot uuid for ``mol``.
        """
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
        """Remove the slot with the given uuid, if present.

        Parameters
        ----------
        uid : str
            The slot uuid to remove. A missing uuid is a no-op.
        """
        with self._lock:
            self.slots.pop(uid, None)

    def diff_and_snapshot(self) -> list[tuple[str, str, dict]]:
        """Diff each slot's live mol against its snapshot.

        Updates each changed slot's snapshot (and topo_hash on a topology
        change) in place after diffing.

        Returns
        -------
        changes : list of tuple of (str, str, dict)
            One ``(kind, slot_uuid, payload_hints)`` tuple per changed slot,
            where ``kind`` is ``"topology"`` or ``"coords"``. Empty when
            nothing changed.
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
