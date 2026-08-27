import json

import numpy as np
import pytest

molviewspec = pytest.importorskip("molviewspec")

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar.mvs import (
    MIN_CARTOON_RESIDUES,
    build_mvs,
    rotation_to_direction_up,
)


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


def _find_values(node, key):
    """Collect every value stored under `key` anywhere in a nested structure."""
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                found.append(v)
            found.extend(_find_values(v, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_values(item, key))
    return found


@pytest.fixture
def two_frame_charged_mol():
    """Two atoms, two frames, coordinates that differ between the frames."""
    mol = Molecule().empty(2)
    mol.element[:] = ["N", "O"]
    mol.name[:] = ["N1", "O2"]
    mol.resname[:] = ["LIG", "LIG"]
    mol.resid[:] = [1, 1]
    mol.chain[:] = ["A", "A"]
    mol.segid[:] = ["L", "L"]
    mol.record[:] = ["HETATM", "HETATM"]
    mol.serial[:] = [1, 2]
    mol.formalcharge[:] = [1, -1]
    mol.coords = np.zeros((2, 3, 2), dtype=np.float32)
    mol.coords[:, :, 0] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    mol.coords[:, :, 1] = [[0.0, 0.0, 70.0], [1.0, 0.0, 70.0]]
    return mol


def test_formal_charge_labels_follow_mol_frame(two_frame_charged_mol):
    """Labels are placed at mol.frame coordinates, not always at frame 0."""
    mol = two_frame_charged_mol

    mol.frame = 0
    positions = _find_values(json.loads(build_mvs(mol, structure_url="data:,")), "position")
    assert [0.0, 0.0, 0.0] in positions

    mol.frame = 1
    positions = _find_values(json.loads(build_mvs(mol, structure_url="data:,")), "position")
    assert [0.0, 0.0, 70.0] in positions
    assert [0.0, 0.0, 0.0] not in positions


def test_highlight_bonds_follow_mol_frame(two_frame_charged_mol):
    """Tube endpoints are taken at mol.frame."""
    mol = two_frame_charged_mol
    mol.frame = 1

    blob = build_mvs(
        mol,
        structure_url="data:,",
        highlight_bonds=[("name N1", "name O2")],
    )
    starts = _find_values(json.loads(blob), "start")

    assert [0.0, 0.0, 70.0] in starts


def test_rotation_default_is_the_identity_view():
    direction, up = rotation_to_direction_up(None)
    assert direction == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
    assert up == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotation_front_preset_equals_no_rotation():
    # pytest.approx cannot compare a tuple-of-tuples directly, so compare
    # direction and up separately.
    front_direction, front_up = rotation_to_direction_up("front")
    no_rotation_direction, no_rotation_up = rotation_to_direction_up((0, 0, 0))
    assert front_direction == pytest.approx(no_rotation_direction, abs=1e-9)
    assert front_up == pytest.approx(no_rotation_up, abs=1e-9)


def test_rotation_back_preset_is_antiparallel_to_front():
    front, _ = rotation_to_direction_up("front")
    back, _ = rotation_to_direction_up("back")
    assert np.dot(front, back) == pytest.approx(-1.0, abs=1e-9)


def test_rotation_top_preset_looks_down():
    """direction points position -> target, so looking down from above is -y."""
    direction, _ = rotation_to_direction_up("top")
    assert direction == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)


def test_rotation_right_preset_views_from_positive_x():
    direction, _ = rotation_to_direction_up("right")
    assert direction == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)


def test_rotation_preserves_orthonormality():
    direction, up = rotation_to_direction_up((33.0, -12.0, 87.0))
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-9)
    assert np.dot(direction, up) == pytest.approx(0.0, abs=1e-9)


def test_rotation_rejects_an_unknown_preset():
    with pytest.raises(ValueError, match="Unknown orientation"):
        rotation_to_direction_up("sideways")


def test_build_mvs_emits_focus_direction_and_zoom(ligand_mol):
    blob = build_mvs(
        ligand_mol, structure_url="data:,", rotate="top", zoom=2.0
    )
    scene = json.loads(blob)
    directions = _find_values(scene, "direction")
    factors = _find_values(scene, "radius_factor")

    assert any(d == pytest.approx([0.0, -1.0, 0.0], abs=1e-9) for d in directions)
    # zoom is the inverse of radius_factor: larger zoom means a tighter sphere
    assert 0.5 in factors
    assert "focus" in blob


def test_build_mvs_emits_background_colour(ligand_mol):
    blob = build_mvs(ligand_mol, structure_url="data:,", background_color="#101010")
    assert "#101010" in blob
    assert "background_color" in blob


def test_build_mvs_without_camera_args_emits_no_focus(ligand_mol):
    blob = build_mvs(ligand_mol, structure_url="data:,")
    assert "radius_factor" not in blob
    assert "direction" not in blob


def test_build_mvs_focus_sel_matching_emits_focus(ligand_mol):
    blob = build_mvs(ligand_mol, structure_url="data:,", focus_sel="name C1")
    assert "focus" in blob


def test_build_mvs_focus_sel_non_matching_without_camera_emits_no_focus(ligand_mol):
    """build_mvs's permissive contract: a focus_sel alone that matches no
    atoms is silently ignored. This must keep working, since an out-of-repo
    caller (the Acellera docs theme) depends on build_mvs and must not start
    raising for a selection that happens to miss."""
    blob = build_mvs(ligand_mol, structure_url="data:,", focus_sel="resname NOPE")
    assert "focus" not in blob
    assert "radius_factor" not in blob
    assert "direction" not in blob


def test_build_mvs_focus_sel_non_matching_with_camera_still_applies_camera(
    ligand_mol,
):
    """A center selection matching no atoms must not silently discard the
    requested rotate/zoom too: the camera direction and radius_factor still
    apply to the whole structure instead of being dropped along with focus."""
    blob = build_mvs(
        ligand_mol,
        structure_url="data:,",
        focus_sel="resname NOPE",
        rotate="top",
        zoom=2.0,
    )
    scene = json.loads(blob)
    directions = _find_values(scene, "direction")
    factors = _find_values(scene, "radius_factor")

    assert any(d == pytest.approx([0.0, -1.0, 0.0], abs=1e-9) for d in directions)
    assert 0.5 in factors
    assert "focus" in blob


from moleculekit.viewer.molstar.mvs import mvs_from_scene
from moleculekit.viewer.molstar.scene import build_scene


def test_mvs_from_scene_emits_the_described_components(ligand_mol):
    scene = build_scene(ligand_mol)
    blob = mvs_from_scene(scene, structure_url="data:,")
    parsed = json.loads(blob)
    assert "ball_and_stick" in blob
    assert "data:," in blob
    assert parsed  # parses as MVS JSON


def test_mvs_from_scene_matches_build_mvs_for_the_automatic_scene(ligand_mol):
    """The two encodings of one description must agree.

    Each call stamps its own wall-clock ``metadata.timestamp`` (molviewspec's
    ``GlobalMetadata.timestamp`` default factory), so that field is dropped
    before comparing; it is serialization noise, not part of either encoding
    of the scene description.
    """
    from_scene = json.loads(
        mvs_from_scene(build_scene(ligand_mol), structure_url="data:,")
    )
    from_builder = json.loads(build_mvs(ligand_mol, structure_url="data:,"))
    from_scene["metadata"].pop("timestamp", None)
    from_builder["metadata"].pop("timestamp", None)
    assert from_scene == from_builder


@pytest.fixture
def protein_mol():
    """Standard-residue polymer at or above MIN_CARTOON_RESIDUES, so the
    cartoon branch fires instead of ball-and-stick-everything."""
    n = MIN_CARTOON_RESIDUES + 2
    mol = Molecule().empty(n)
    mol.element[:] = "C"
    mol.name[:] = "CA"
    mol.resname[:] = "GLY"
    mol.resid[:] = np.arange(1, n + 1)
    mol.chain[:] = "A"
    mol.segid[:] = "P"
    mol.record[:] = "ATOM"
    mol.serial[:] = np.arange(1, n + 1)
    mol.coords = np.zeros((n, 3, 1), dtype=np.float32)
    mol.coords[:, 0, 0] = np.arange(n, dtype=np.float32) * 3.8
    return mol


def test_mvs_from_scene_matches_build_mvs_for_the_cartoon_scene(protein_mol):
    """The two encodings must also agree on the cartoon branch: what a
    protein actually renders, and otherwise exercised only indirectly through
    headless render tests elsewhere.

    Each call stamps its own wall-clock ``metadata.timestamp`` (molviewspec's
    ``GlobalMetadata.timestamp`` default factory), so that field is dropped
    before comparing; it is serialization noise, not part of either encoding
    of the scene description.
    """
    from_scene = json.loads(
        mvs_from_scene(build_scene(protein_mol), structure_url="data:,")
    )
    from_builder = json.loads(build_mvs(protein_mol, structure_url="data:,"))
    from_scene["metadata"].pop("timestamp", None)
    from_builder["metadata"].pop("timestamp", None)
    assert from_scene == from_builder


def test_build_mvs_representations_stay_additive(ligand_mol):
    """The docs theme depends on this: its highlights layer over the
    automatic scene rather than replacing it."""
    blob = build_mvs(
        ligand_mol,
        structure_url="data:,",
        representations=[{"sel": "name C1", "type": "spacefill"}],
    )
    assert "spacefill" in blob
    assert "ball_and_stick" in blob, "the automatic scene must survive"
