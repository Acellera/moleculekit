import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.representations import (
    VMD_COLORS,
    _Representation,
    _normalize,
)


def _mol():
    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)
    return mol


def test_translate_style_and_theme_color():
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("all", "NewCartoon", "Name"))
    assert out["type"] == "cartoon"
    assert out["color"] == {"theme": "element-symbol"}
    assert out["atom_indices"] == [0, 1, 2]


def test_translate_vdw_and_secondary_structure():
    mol = _mol()
    out = mol.reps._translateMolstar(
        _Representation("all", "VDW", "Secondary Structure")
    )
    assert out["type"] == "spacefill"
    assert out["color"] == {"theme": "secondary-structure"}


def test_translate_colorid_int_becomes_uniform_hex():
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("all", "Licorice", 1))
    assert out["type"] == "ball_and_stick"
    assert out["color"] == "#ff0000"


def test_translate_unmatched_selection_returns_none():
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("resname ZZZ", "Lines", "Name"))
    assert out is None


def _trypsin():
    from pathlib import Path

    return Molecule(str(Path(__file__).parent / "test_molecule" / "3ptb_filtered.pdb"))


def test_add_defaults_lists_the_automatic_scene():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _trypsin()
    mol.reps.addDefaults()
    assert [r.sel for r in mol.reps.replist] == [
        "protein or nucleic",
        "not (protein or nucleic)",
    ]

    # Every atom is drawn, and none of them twice.
    covered = sum(mol.atomselect(r.sel).astype(int) for r in mol.reps.replist)
    assert (covered == 1).all()

    scene = build_scene(mol, mol.reps.replist)
    types = [c["representation"]["type"] for c in scene["components"]]
    assert types == ["cartoon", "ball_and_stick"]


def test_add_defaults_on_a_small_molecule_is_a_single_representation():
    mol = _mol()
    mol.reps.addDefaults()
    assert [(r.sel, r.style) for r in mol.reps.replist] == [("all", "CPK")]


def test_add_defaults_omits_a_hetero_representation_matching_nothing():
    mol = _trypsin()
    mol.filter("protein")
    mol.reps.addDefaults()
    assert [r.sel for r in mol.reps.replist] == ["protein or nucleic"]


def test_unknown_style_is_rejected():
    """An unmapped style used to come out as ball-and-stick.

    Asking for a surface then produced sticks, which looks like the surface
    was drawn and failed rather than like a typo.
    """
    import pytest

    mol = _mol()
    with pytest.raises(ValueError, match="Unknown representation style"):
        mol.reps._translateMolstar(_Representation("all", "NoSuchStyle", "Name"))


def test_index_colours_by_sequence_id():
    """``residue-id`` is no theme Mol* knows.

    Passing it left Mol* to fall back to its default, so ``Index`` silently
    coloured by chain: on a two-chain structure it rendered byte-identical to
    ``Chain`` and differently from ``Name``.
    """
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("all", "VDW", "Index"))
    assert out["color"] == {"theme": "sequence-id"}


def test_new_styles_translate_to_their_molstar_types():
    mol = _mol()
    expected = {
        "Surf": "molecular_surface",
        "QuickSurf": "gaussian_surface",
        "Points": "point",
        "Labels": "label",
        "FormalCharges": "formal_charge",
    }
    for style, type_ in expected.items():
        out = mol.reps._translateMolstar(_Representation("all", style, "Name"))
        assert out["type"] == type_, style


def test_label_styles_are_skipped_for_the_other_viewers():
    """VMD and NGL have no representation for these, so they are not sent."""
    mol = _mol()
    for style in ("Labels", "FormalCharges"):
        assert mol.reps._translateNGL(_Representation("all", style, "Name")) is None


def test_formal_charge_rep_labels_charged_atoms_without_drawing_a_component():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.formalcharge[:] = [1, 0, -1]
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", "FormalCharges")

    scene = build_scene(mol, mol.reps.replist)
    assert len(scene["components"]) == 1
    assert [label["text"] for label in scene["labels"]] == ["+1", "-1"]


def test_representations_replace_the_automatic_charge_labels():
    """Labels follow the same replace rule as components.

    Charges are otherwise labelled on every render whatever the
    representations say, with no way to turn them off.
    """
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.formalcharge[:] = [1, 0, -1]
    assert build_scene(mol)["labels"], "the automatic scene still labels charges"

    mol.reps.add("all", "Licorice", "Name")
    assert "labels" not in build_scene(mol, mol.reps.replist)


def test_labels_with_nothing_to_draw_are_rejected():
    """A FormalCharges representation on its own renders a blank image."""
    import pytest

    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.formalcharge[:] = [1, 0, -1]
    mol.reps.add("all", "FormalCharges")
    with pytest.raises(ValueError, match="representations draw nothing"):
        build_scene(mol, mol.reps.replist)


def test_style_and_colour_names_ignore_spacing():
    """"SecondaryStructure" must mean the same as "Secondary Structure".

    Only the spaced spelling was a key, so the concatenated one fell through
    to being read as a colour name and drew a garbage uniform colour. Our own
    Molecule.render docstring used it.
    """
    mol = _mol()
    spellings = ["Secondary Structure", "SecondaryStructure", "secondary_structure"]
    for color in spellings:
        out = mol.reps._translateMolstar(_Representation("all", "NewCartoon", color))
        assert out["color"] == {"theme": "secondary-structure"}, color
    out = mol.reps._translateMolstar(_Representation("all", "new cartoon", "Name"))
    assert out["type"] == "cartoon"


def test_docstrings_list_every_style_and_colour():
    """The vocabulary is spelled out where users meet it, so it can go stale.

    Before this, both docstrings only linked to VMD's manual, which covers
    neither the Mol*-only styles nor the colour modes Mol* actually takes.
    """
    import re

    from moleculekit.representations import (
        MOLSTAR_STYLES,
        MOLSTAR_THEMES,
        Representations,
        _normalize,
    )

    for doc in (Representations.add.__doc__, Molecule.view.__doc__):
        listed = {_normalize(t) for t in re.findall(r"``([^`]+)``", doc)}
        assert not set(MOLSTAR_STYLES) - listed, set(MOLSTAR_STYLES) - listed
        assert not set(MOLSTAR_THEMES) - listed, set(MOLSTAR_THEMES) - listed


VOCABULARY = [
    ("NewCartoon", "cartoon", "cartoon"),
    ("CPK", "ball-and-stick", "ball_and_stick"),
    ("VDW", "spacefill", "spacefill"),
    ("Lines", "line", "line"),
    ("Surf", "molecular-surface", "molecular_surface"),
    ("QuickSurf", "gaussian-surface", "gaussian_surface"),
    ("Points", "point", "point"),
    ("Labels", "atom-label", "label"),
    ("Putty", "putty", "putty"),
]


@pytest.mark.parametrize("vmd,molstar,expected", VOCABULARY)
def test_vmd_and_molstar_style_names_are_interchangeable(vmd, molstar, expected):
    mol = _mol()
    for name in (vmd, molstar):
        out = mol.reps._translateMolstar(_Representation("all", name, "Name"))
        assert out["type"] == expected, name


@pytest.mark.parametrize(
    "vmd,molstar",
    [
        ("Name", "element-symbol"),
        ("Chain", "chain-id"),
        ("ResName", "residue-name"),
        ("Index", "sequence-id"),
        ("Secondary Structure", "secondary-structure"),
        ("Beta", "uncertainty"),
    ],
)
def test_vmd_and_molstar_colour_names_are_interchangeable(vmd, molstar):
    mol = _mol()
    rep = mol.reps._translateMolstar(_Representation("all", "VDW", vmd))
    assert rep["color"] == {"theme": molstar}
    assert mol.reps._translateMolstar(_Representation("all", "VDW", molstar)) == rep


@pytest.mark.parametrize("vmd,molstar,_expected", VOCABULARY)
def test_ngl_understands_both_vocabularies(vmd, molstar, _expected):
    """The NGL path had no test at all, and went a while raising NameError."""
    mol = _mol()
    translated = [
        mol.reps._translateNGL(_Representation("all", name, "Name"))
        for name in (vmd, molstar)
    ]
    # None means NGL has no equivalent, which both spellings must agree on.
    styles = {rep.style if rep is not None else None for rep in translated}
    assert len(styles) == 1, f"{vmd} and {molstar} translate apart: {styles}"


def test_size_reaches_every_viewer():
    mol = _mol()
    rep = _Representation("all", "Licorice", "Name", size=0.3)
    assert mol.reps._translateMolstar(rep)["size_factor"] == 0.3
    assert mol.reps._translateNGL(rep).size == 0.3


def test_size_scales_formal_charge_label_text():
    """`size` is documented as scaling label text, charges included."""
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.formalcharge[0] = 1
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", "FormalCharges", size=2.0)

    plain = _mol()
    plain.formalcharge[0] = 1
    plain.reps.add("all", "Licorice", "Name")
    plain.reps.add("all", "FormalCharges")

    scaled = build_scene(mol, mol.reps.replist)["labels"][0]["size"]
    assert scaled == 2.0 * build_scene(plain, plain.reps.replist)["labels"][0]["size"]


def test_b_factor_and_occupancy_colouring_reach_every_viewer():
    """VMD calls the B factor Beta, Mol* calls it uncertainty, NGL bfactor."""
    mol = _mol()
    for name in ("Beta", "uncertainty"):
        rep = _Representation("all", "NewCartoon", name)
        assert mol.reps._translateMolstar(rep)["color"] == {"theme": "uncertainty"}
        assert mol.reps._translateNGL(rep).color == "bfactor"
    # Mol*'s spelling is translated for VMD; VMD's own is sent as written.
    assert VMD_COLORS[_normalize("uncertainty")] == "Beta"
    assert _normalize("Beta") not in VMD_COLORS

    rep = _Representation("all", "NewCartoon", "Occupancy")
    assert mol.reps._translateMolstar(rep)["color"] == {"theme": "occupancy"}
    assert mol.reps._translateNGL(rep).color == "occupancy"
