import json

import numpy as np
import pytest

molviewspec = pytest.importorskip("molviewspec")

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar.mvs import build_mvs


@pytest.fixture
def ligand_mol():
    """Small hetero-only molecule (fewer than MIN_CARTOON_RESIDUES polymers)."""
    mol = Molecule().empty(2)
    mol.element[:] = ["C", "O"]
    mol.name[:] = ["C1", "O2"]
    mol.resname[:] = ["LIG", "LIG"]
    mol.resid[:] = [1, 1]
    mol.chain[:] = ["A", "A"]
    mol.segid[:] = ["L", "L"]
    mol.record[:] = ["HETATM", "HETATM"]
    mol.serial[:] = [1, 2]
    mol.formalcharge[:] = [0, -1]
    mol.coords = np.zeros((2, 3, 1), dtype=np.float32)
    mol.coords[1, :, 0] = [1.2, 0.0, 0.0]
    return mol


def test_build_mvs_embeds_structure_url(ligand_mol):
    url = "data:application/octet-stream;base64,QUJD"
    blob = build_mvs(ligand_mol, structure_url=url)
    json.loads(blob)
    assert url in blob
    # hetero-only -> rendered ball_and_stick, not cartoon
    assert "ball_and_stick" in blob
    assert "cartoon" not in blob


def test_build_mvs_representation_accepts_atom_indices(ligand_mol):
    blob = build_mvs(
        ligand_mol,
        structure_url="data:,",
        representations=[{"atom_indices": [0], "type": "spacefill",
                          "color": {"theme": "element-symbol"}}],
    )
    assert "spacefill" in blob


def test_build_mvs_formal_charge_label(ligand_mol):
    blob = build_mvs(ligand_mol, structure_url="data:,")
    # the -1 formal charge atom gets a label primitive
    assert "-1" in blob
