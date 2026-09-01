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
    The how-to page enumerates the same vocabulary and can go stale the same
    way, so it is checked here too.
    """
    import re

    from moleculekit.representations import (
        MOLSTAR_STYLES,
        MOLSTAR_THEMES,
        Representations,
        _normalize,
    )

    from pathlib import Path

    page = Path(__file__).parents[1] / "doc/source/how-to/choose-representations.md"
    docs = [Representations.add.__doc__, Molecule.view.__doc__, page.read_text()]
    for doc in docs:
        # One or more backticks, no newline inside: matches RST's ``name``
        # and Markdown's `name` alike.
        listed = {_normalize(t) for t in re.findall(r"`+([^`\n]+)`+", doc)}
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


@pytest.mark.parametrize(
    "style,expected",
    [("Backbone", "backbone"), ("Ellipsoid", "ellipsoid"), ("backbone", "backbone")],
)
def test_backbone_and_ellipsoid_styles(style, expected):
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("all", style, "Name"))
    assert out["type"] == expected


@pytest.mark.parametrize(
    "name,theme",
    [
        ("Element-Index", "element-index"),
        ("Entity-ID", "entity-id"),
        ("Polymer-ID", "polymer-id"),
        ("Model-Index", "model-index"),
        ("Structure-Index", "structure-index"),
        ("Illustrative", "illustrative"),
    ],
)
def test_the_remaining_molstar_colour_themes(name, theme):
    mol = _mol()
    rep = mol.reps._translateMolstar(_Representation("all", "NewCartoon", name))
    assert rep["color"] == {"theme": theme}


def test_c_atom_color_recolours_only_carbon():
    """Carbon carries the entity's colour while N, O and S keep theirs.

    Mol* exposes this as the element theme's carbonColor, which is also what
    it uses to colour carbon by chain when nothing else is asked for.
    """
    mol = _mol()
    rep = mol.reps._translateMolstar(
        _Representation("all", "Licorice", "Name", c_atom_color="#66ccff")
    )
    assert rep["color"] == {"theme": "element-symbol", "carbon": {"uniform": "#66ccff"}}

    rep = mol.reps._translateMolstar(
        _Representation("all", "Licorice", "Name", c_atom_color="chain-id")
    )
    assert rep["color"] == {"theme": "element-symbol", "carbon": {"theme": "chain-id"}}


def test_c_atom_color_is_ignored_when_the_colour_is_not_by_element():
    """There is no carbon to recolour in a uniform or per-residue colouring."""
    mol = _mol()
    for color in ("red", "ResName"):
        rep = mol.reps._translateMolstar(
            _Representation("all", "Licorice", color, c_atom_color="#66ccff")
        )
        assert "carbon" not in str(rep["color"])


def test_size_theme_is_passed_through_and_checked():
    mol = _mol()
    rep = mol.reps._translateMolstar(
        _Representation("all", "VDW", "Name", size_theme="Uniform")
    )
    assert rep["size_theme"] == "uniform"

    with pytest.raises(ValueError, match="Unknown size_theme"):
        mol.reps._translateMolstar(
            _Representation("all", "VDW", "Name", size_theme="enormous")
        )


def test_backbone_reaches_ngl_but_ellipsoid_does_not():
    """NGL draws a backbone; it has nothing for the rest of the Mol*-only set."""
    mol = _mol()
    assert mol.reps._translateNGL(_Representation("all", "Backbone", "Name")).style == (
        "backbone"
    )
    for style in ("Ellipsoid", "Putty", "Labels", "FormalCharges"):
        assert mol.reps._translateNGL(_Representation("all", style, "Name")) is None


def test_labels_can_carry_any_per_atom_field():
    """Mol* writes a label of its own choosing, so fields are built here.

    That is the same mechanism the formal charge labels use, one transform per
    label, which is why the same cap applies.
    """
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.chain[:] = "A"
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", "Labels", label_fields=["chain", "resname", "resid", "name"])
    assert [lab["text"] for lab in build_scene(mol, mol.reps.replist)["labels"]] == [
        "A ALA 1 N",
        "A ALA 1 CA",
        "A ALA 1 C",
    ]


def test_label_fields_accept_a_bare_string_and_other_viewers_names():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", "Labels", label_fields="index")
    assert [lab["text"] for lab in build_scene(mol, mol.reps.replist)["labels"]] == [
        "0",
        "1",
        "2",
    ]

    other = _mol()
    other.reps.add("all", "Licorice", "Name")
    other.reps.add("all", "Labels", label_fields=["residueName", "residueIndex"])
    assert build_scene(other, other.reps.replist)["labels"][0]["text"] == "ALA 1"


def test_an_unknown_label_field_is_rejected():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", "Labels", label_fields=["bfactor_typo"])
    with pytest.raises(ValueError, match="Cannot label by 'bfactor_typo'"):
        build_scene(mol, mol.reps.replist)


def test_labels_without_fields_stay_a_molstar_representation():
    """The cheap path: one representation for the whole selection."""
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "Labels", "black")
    scene = build_scene(mol, mol.reps.replist)
    assert [c["representation"]["type"] for c in scene["components"]] == ["label"]
    assert "labels" not in scene


def test_update_changes_only_what_it_is_given():
    """Recolouring a representation must not cost it its size or selection."""
    mol = _mol()
    mol.reps.add("name CA", "VDW", "Name", size=0.8, opacity=0.5)
    mol.reps.update(0, color="Chain")

    rep = mol.reps.replist[0]
    assert (rep.sel, rep.style, rep.color) == ("name CA", "VDW", "Chain")
    assert (rep.size, rep.opacity) == (0.8, 0.5)


def test_update_keeps_a_representation_where_it_was():
    """The point of updating rather than removing and re-adding: the index of
    every other representation stays what the caller last saw."""
    mol = _mol()
    mol.reps.add("all", "Lines", "Name")
    mol.reps.add("name CA", "VDW", "Name")
    mol.reps.add("all", "NewCartoon", "Name")
    mol.reps.update(1, style="Licorice")

    assert [r.style for r in mol.reps.replist] == ["Lines", "Licorice", "NewCartoon"]


def test_a_hidden_representation_draws_nothing_but_keeps_its_index():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "Lines", "Name")
    mol.reps.add("all", "VDW", "Name")
    mol.reps.update(1, visibility=False)

    scene = build_scene(mol, mol.reps.replist)
    assert [c["representation"]["type"] for c in scene["components"]] == ["line"]
    assert len(mol.reps.replist) == 2
    assert "hidden" in str(mol.reps)

    mol.reps.update(1, visibility=True)
    scene = build_scene(mol, mol.reps.replist)
    assert [c["representation"]["type"] for c in scene["components"]] == [
        "line",
        "spacefill",
    ]


def test_label_style_reaches_every_label():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add(
        "all",
        "Labels",
        label_fields="name",
        label_style={"bg_color": "#003366", "bg_opacity": 0.9, "offset_y": 1.5},
    )
    labels = build_scene(mol, mol.reps.replist)["labels"]
    assert len(labels) == 3
    for label in labels:
        assert label["bg_color"] == "#003366"
        assert label["bg_opacity"] == 0.9
        assert label["offset_y"] == 1.5


def test_an_unknown_label_style_key_is_rejected():
    """A misspelled cosmetic that silently drew a default label would look
    like the option had simply not worked."""
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", "Labels", label_fields="name", label_style={"bgcolor": "red"})
    with pytest.raises(ValueError, match="Unknown label_style 'bgcolor'"):
        build_scene(mol, mol.reps.replist)


def test_update_sel_every_frame_is_carried_but_not_drawn():
    """It belongs to a live viewer following a trajectory. A rendered image is
    one frame, so it reaches the scene and changes nothing about it."""
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _mol()
    mol.reps.add("all", "VDW", "Name", update_sel_every_frame=True)
    assert mol.reps.replist[0].update_sel_every_frame
    # Not in the scene: nothing there could act on it.
    assert "update_sel_every_frame" not in mol.reps._translateMolstar(
        mol.reps.replist[0]
    )

    scene = build_scene(mol, mol.reps.replist)
    assert [c["representation"]["type"] for c in scene["components"]] == ["spacefill"]
