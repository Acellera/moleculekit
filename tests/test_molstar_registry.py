import threading

import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar.registry import (
    Registry,
    Slot,
    coords_to_bytes,
    topo_hash,
)


@pytest.fixture
def small_mol():
    mol = Molecule().empty(3)
    mol.element[:] = ["C", "C", "O"]
    mol.name[:] = ["C1", "C2", "O3"]
    mol.resname[:] = ["LIG", "LIG", "LIG"]
    mol.resid[:] = [1, 1, 1]
    mol.chain[:] = ["A", "A", "A"]
    mol.segid[:] = ["L", "L", "L"]
    mol.record[:] = ["HETATM"] * 3
    mol.serial[:] = [1, 2, 3]
    mol.formalcharge[:] = [0, 0, -1]
    mol.bonds = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    mol.bondtype = np.array(["1", "2"], dtype=object)
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)
    return mol


def _make_slot_mol(name="A", element="C"):
    mol = Molecule().empty(1)
    mol.element[:] = [element]
    mol.name[:] = [name]
    mol.resname[:] = ["LIG"]
    mol.resid[:] = [1]
    mol.chain[:] = ["A"]
    mol.segid[:] = ["L"]
    mol.record[:] = ["HETATM"]
    mol.serial[:] = [1]
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    return mol


def test_hash_is_stable_for_same_mol(small_mol):
    h1 = topo_hash(small_mol)
    h2 = topo_hash(small_mol)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 40


def test_hash_is_stable_across_copy(small_mol):
    assert topo_hash(small_mol) == topo_hash(small_mol.copy())


def test_hash_changes_on_topology_change(small_mol):
    h0 = topo_hash(small_mol)
    small_mol.resname[0] = "ACE"
    assert topo_hash(small_mol) != h0


def test_hash_changes_on_formalcharge_change(small_mol):
    h0 = topo_hash(small_mol)
    small_mol.formalcharge[0] = 1
    assert topo_hash(small_mol) != h0


def test_hash_changes_on_bondtype_change(small_mol):
    h0 = topo_hash(small_mol)
    small_mol.bondtype[0] = "2"
    assert topo_hash(small_mol) != h0


def test_hash_unchanged_when_only_coords_change(small_mol):
    h0 = topo_hash(small_mol)
    small_mol.coords[:] = np.random.rand(3, 3, 1).astype(np.float32)
    assert topo_hash(small_mol) == h0


def test_hash_unchanged_when_only_partial_charge_changes(small_mol):
    h0 = topo_hash(small_mol)
    small_mol.charge[:] = [0.1, -0.1, -0.5]
    assert topo_hash(small_mol) == h0


def test_coords_to_bytes_layout(small_mol):
    small_mol.coords = np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2)
    blob = coords_to_bytes(small_mol)
    arr = np.frombuffer(blob, dtype="<f4")
    assert arr.size == 3 * 3 * 2
    expected_frame0 = small_mol.coords[:, :, 0].T.reshape(-1).astype(np.float32)
    assert np.allclose(arr[: expected_frame0.size], expected_frame0)


def test_register_returns_uuid():
    reg = Registry()
    mol = _make_slot_mol()
    uid = reg.register(mol)
    assert isinstance(uid, str)
    assert len(uid) == 6
    assert uid in reg.slots


def test_register_dedupes_by_identity():
    reg = Registry()
    mol = _make_slot_mol()
    uid1 = reg.register(mol)
    uid2 = reg.register(mol)
    assert uid1 == uid2
    assert len(reg.slots) == 1


def test_register_different_object_same_content_creates_new_slot():
    reg = Registry()
    mol1 = _make_slot_mol()
    mol2 = _make_slot_mol()
    uid1 = reg.register(mol1)
    uid2 = reg.register(mol2)
    assert uid1 != uid2
    assert len(reg.slots) == 2


def test_register_stores_initial_snapshot_and_topo_hash():
    reg = Registry()
    mol = _make_slot_mol()
    uid = reg.register(mol)
    slot = reg.slots[uid]
    assert slot.mol_ref is mol
    assert slot.snapshot is not mol
    assert slot.topo_hash == topo_hash(mol)


def test_remove_drops_slot():
    reg = Registry()
    mol = _make_slot_mol()
    uid = reg.register(mol)
    reg.remove(uid)
    assert uid not in reg.slots


def test_remove_unknown_slot_is_noop():
    reg = Registry()
    reg.remove("NOTHERE")


def test_diff_detects_coords_only_change():
    reg = Registry()
    mol = _make_slot_mol()
    uid = reg.register(mol)
    mol.coords[0, 0, 0] = 5.0
    changes = reg.diff_and_snapshot()
    assert len(changes) == 1
    kind, slot_uid, _payload = changes[0]
    assert kind == "coords"
    assert slot_uid == uid


def test_diff_detects_topology_change():
    reg = Registry()
    mol = _make_slot_mol()
    reg.register(mol)
    mol.resname[:] = ["XXX"]
    changes = reg.diff_and_snapshot()
    assert len(changes) == 1
    assert changes[0][0] == "topology"


def test_diff_returns_empty_when_unchanged():
    reg = Registry()
    mol = _make_slot_mol()
    reg.register(mol)
    reg.diff_and_snapshot()
    assert reg.diff_and_snapshot() == []


def test_diff_after_register_returns_empty():
    reg = Registry()
    mol = _make_slot_mol()
    reg.register(mol)
    assert reg.diff_and_snapshot() == []


def test_concurrent_register_remove_is_safe():
    reg = Registry()
    errors = []

    def hammer():
        try:
            for _ in range(50):
                mol = _make_slot_mol()
                uid = reg.register(mol)
                reg.remove(uid)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_slot_dataclass_has_label_and_visible():
    reg = Registry()
    mol = _make_slot_mol()
    uid = reg.register(mol)
    slot = reg.slots[uid]
    assert isinstance(slot, Slot)
    assert slot.visible is True
    assert isinstance(slot.label, str)
