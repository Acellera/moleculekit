"""The rule that decides what a Mol* scene looks like, as plain data.

Both the interactive viewer and the headless renderer consume the dict this
module produces, so they cannot drift: one decides, one applies. MolViewSpec is
one possible encoding of the same description (see mvs.py) and is used by the
notebook viewer and the docs theme, which cannot reach the shared bundle.

The vocabulary is MolViewSpec's: selector names, representation type names and
Mol* colour theme names are spelled exactly as MVS spells them, so translating
in either direction is mechanical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from moleculekit.residues import (
    NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS,
    PROTEIN_RESIDUE_NAMES_WITH_VARIANTS,
)

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)

# These moved here from mvs.py: scene.py owns the rule, mvs.py re-exports them.
MIN_CARTOON_RESIDUES = 6
BALL_AND_STICK_SIZE_FACTOR = 0.6
MAX_FORMAL_CHARGE_LABELS = 200

#: Fields a Labels representation can write, beyond the molecule's own per-atom
#: arrays. The names on the left are what other viewers call them.
LABEL_FIELD_ALIASES = {
    "residuename": "resname",
    "residueindex": "resid",
    "atomname": "name",
    "chainid": "chain",
}
_BALL_AND_STICK_SELECTORS = ("ligand", "ion", "water", "branched")

# Which resnames count as canonical polymer, deciding cartoon versus
# ball-and-stick. Derived from residues.py so the two cannot drift: the
# WITH_VARIANTS sets are exactly "canonical residue, including force-field
# renames" (HIS -> HID/HIE/HIP, CYS -> CYX/CYM, and so on).
STANDARD_POLYMER_RESNAMES = (
    PROTEIN_RESIDUE_NAMES_WITH_VARIANTS
    | NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS
    # Spellings residues.py does not carry: ARN is the neutral-arginine name
    # some force fields use (residues.py spells it AR0), and these RNA and
    # deoxyuridine names appear in older PDB-derived files. Dropping them would
    # push those residues out of the cartoon into ball-and-stick.
    | {"ARN", "DU", "RA", "RC", "RG", "RU"}
)

DEFAULT_DIRECTION = (0.0, 0.0, -1.0)
DEFAULT_UP = (0.0, 1.0, 0.0)

# `direction` points from the camera position to the target, so "top" (looking
# down from above) is -y, which is Rx(-90) applied to DEFAULT_DIRECTION.
ORIENTATION_PRESETS = {
    "front": (0.0, 0.0, 0.0),
    "back": (0.0, 180.0, 0.0),
    "left": (0.0, -90.0, 0.0),
    "right": (0.0, 90.0, 0.0),
    "top": (-90.0, 0.0, 0.0),
    "bottom": (90.0, 0.0, 0.0),
}


def _count_standard_polymer_residues(mol) -> int:
    seen: dict = {}
    for resid, ins, chain, segid, resname in zip(
        mol.resid.tolist(),
        mol.insertion.tolist(),
        mol.chain.tolist(),
        mol.segid.tolist(),
        mol.resname.tolist(),
    ):
        seen[(resid, ins, chain, segid)] = resname
    return sum(1 for rn in seen.values() if rn in STANDARD_POLYMER_RESNAMES)


def rotation_to_direction_up(rotate):
    """Resolve a rotation into the MVS ``direction`` and ``up`` vectors.

    Parameters
    ----------
    rotate : str or tuple of float or None
        A preset name from ``ORIENTATION_PRESETS``, a tuple of ``(rx, ry, rz)``
        rotations in degrees applied about the x, y and z axes in that order, or
        None for the default view.

    Returns
    -------
    direction : tuple of float
        Unit vector from the camera position toward the target.
    up : tuple of float
        Unit vector controlling the roll about ``direction``.

    Raises
    ------
    ValueError
        If ``rotate`` is a string that names no known preset.
    """
    if rotate is None:
        return DEFAULT_DIRECTION, DEFAULT_UP

    if isinstance(rotate, str):
        key = rotate.lower()
        if key not in ORIENTATION_PRESETS:
            raise ValueError(
                f"Unknown orientation {rotate!r}. Use one of "
                f"{sorted(ORIENTATION_PRESETS)} or a (rx, ry, rz) tuple in degrees."
            )
        rotate = ORIENTATION_PRESETS[key]

    rx, ry, rz = (np.deg2rad(float(angle)) for angle in rotate)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rot = rot_z @ rot_y @ rot_x

    direction = rot @ np.array(DEFAULT_DIRECTION)
    up = rot @ np.array(DEFAULT_UP)
    return tuple(float(v) for v in direction), tuple(float(v) for v in up)


_ELEMENT = {"theme": "element-symbol"}


def _builtin(name: str) -> dict:
    return {"kind": "builtin", "name": name}


def _atoms(indices) -> dict:
    return {"kind": "atoms", "indices": [int(i) for i in indices]}


def _ball_and_stick(select: dict, color: dict) -> dict:
    return {
        "select": select,
        "representation": {
            "type": "ball_and_stick",
            "size_factor": BALL_AND_STICK_SIZE_FACTOR,
        },
        "color": color,
    }


def _automatic_components(mol) -> list[dict]:
    """The scene shown when the user has set no representations."""
    if _count_standard_polymer_residues(mol) < MIN_CARTOON_RESIDUES:
        return [_ball_and_stick(_builtin("all"), _ELEMENT)]

    components = [
        {
            "select": _builtin("polymer"),
            "representation": {"type": "cartoon"},
            "color": {"theme": "secondary-structure"},
        }
    ]
    components += [
        _ball_and_stick(_builtin(name), _ELEMENT) for name in _BALL_AND_STICK_SELECTORS
    ]
    # Not redundant with the ligand/ion/water/branched builtins just above:
    # _bcif_bytes() (inline.py) writes only _atom_site, with no entity,
    # chem_comp or struct_conn categories. With none of those present, Mol*
    # cannot classify any atom as ligand, ion, water or branched, so all four
    # of those components draw nothing on every render; this resname-based
    # component is what actually keeps hetero atoms visible in the automatic
    # scene. Do not remove it as a believed-redundant cleanup without first
    # adding those mmCIF categories to the BinaryCIF writer and re-checking
    # coverage (see tests/test_molstar_render.py's automatic-scene
    # ligand/ion test).
    other = sorted(set(mol.resname.tolist()) - STANDARD_POLYMER_RESNAMES)
    if other:
        components.append(
            _ball_and_stick({"kind": "resname", "names": other}, _ELEMENT)
        )
    return components


def default_representations(mol) -> list[tuple]:
    """The automatic scene written as representations any viewer can draw.

    The automatic scene selects with Mol*'s builtin classifiers, which VMD and
    NGL have no equivalent of, so the split here is made with atom selection
    strings instead. It is a faithful starting point rather than a mirror: a
    residue moleculekit calls protein but that is spelled differently from the
    names in ``STANDARD_POLYMER_RESNAMES`` lands in the cartoon here and in
    ball-and-stick there.

    Parameters
    ----------
    mol : Molecule
        The molecule the scene would be built for.

    Returns
    -------
    reps : list of tuple
        ``(sel, style, color)`` triples in drawing order.
    """
    if _count_standard_polymer_residues(mol) < MIN_CARTOON_RESIDUES:
        return [("all", "CPK", "Name")]

    reps = [("protein or nucleic", "NewCartoon", "Secondary Structure")]
    hetero = "not (protein or nucleic)"
    # Skipped when it matches nothing, so a bare protein does not carry a
    # representation that build_scene would only warn about and drop.
    if mol.atomselect(hetero).any():
        reps.append((hetero, "CPK", "Name"))
    return reps


def _components_from_reps(mol, reps) -> tuple[list[dict], list[dict]]:
    """Translate user representations, which replace the automatic scene.

    Parameters
    ----------
    mol : Molecule
        The molecule the selections are resolved against.
    reps : list
        The representations to translate.

    Returns
    -------
    components : list of dict
        One scene component per drawable representation.
    labels : list of dict
        Formal charge labels contributed by ``FormalCharges`` representations.
    """
    from moleculekit.representations import Representations

    components = []
    labels = []
    dropped = []
    for rep in reps:
        translated = Representations(mol)._translateMolstar(rep)
        if translated is None:
            dropped.append(rep.sel)
            logger.warning(
                "Representation selection %r matched no atoms and was dropped.",
                rep.sel,
            )
            continue
        if translated["type"] == "label" and "label_fields" in translated:
            # Mol*'s own label representation writes a label of its own
            # choosing, so a chosen set of fields is built here instead.
            labels.extend(
                _field_labels(
                    mol,
                    translated["atom_indices"],
                    translated["label_fields"],
                    translated.get("size_factor", 1.0),
                )
            )
            continue
        if translated["type"] == "formal_charge":
            # Not a component: this draws the same per-atom "+1"/"-1" text the
            # automatic scene puts on charged atoms, restricted to the atoms
            # this representation selected.
            labels.extend(
                _labels(mol, translated["atom_indices"], translated.get("size_factor", 1.0))
            )
            continue
        color = translated.get("color")
        if color is None:
            color_spec = _ELEMENT
        elif isinstance(color, dict):
            color_spec = color
        else:
            color_spec = {"uniform": color}
        representation = {"type": translated["type"]}
        if translated["type"] == "ball_and_stick":
            # The size the automatic scene uses, so reps.addDefaults() draws
            # the same picture as setting no representations at all.
            representation["size_factor"] = BALL_AND_STICK_SIZE_FACTOR
        if "size_factor" in translated:
            representation["size_factor"] = translated["size_factor"]
        if "size_theme" in translated:
            representation["size_theme"] = translated["size_theme"]
        component = {
            "select": _atoms(translated["atom_indices"]),
            "representation": representation,
            "color": color_spec,
        }
        if "opacity" in translated:
            component["opacity"] = translated["opacity"]
        components.append(component)

    if not components and not labels:
        raise ValueError(
            "Every representation selection matched no atoms "
            f"({', '.join(repr(s) for s in dropped)}), which would render an "
            "empty scene. Check the selections, or clear mol.reps to get the "
            "automatic scene."
        )
    return components, labels


def _field_labels(mol, indices, fields, size=1.0) -> list[dict]:
    """Text beside each atom, built from the molecule's own per-atom fields.

    Mol*'s label representation writes a label of its own choosing, so anything
    else has to be built here and placed atom by atom, as the formal charge
    labels are. That costs one transform per label, which is why the same cap
    applies.

    Parameters
    ----------
    mol : Molecule
        The molecule the fields are read from.
    indices : list of int
        The atoms to label.
    fields : list of str
        Per-atom fields to write, joined by spaces.
    size : float, optional
        Scales the text.

    Returns
    -------
    labels : list of dict
        One label per atom, empty when there are more than the cap allows.

    Raises
    ------
    ValueError
        If a field is not a per-atom field of the molecule.
    """
    resolved = []
    for field in fields:
        name = LABEL_FIELD_ALIASES.get(field.lower().replace("_", ""), field.lower())
        if name == "index":
            values = np.arange(mol.numAtoms)
        else:
            values = getattr(mol, name, None)
        if values is None or len(np.atleast_1d(values)) != mol.numAtoms:
            raise ValueError(
                f"Cannot label by {field!r}: the molecule has no per-atom field "
                f"of that name. Try name, element, resname, resid, chain or index."
            )
        resolved.append(values)

    if len(indices) > MAX_FORMAL_CHARGE_LABELS:
        logger.warning(
            "Skipping labels: %d atoms exceeds the cap of %d. Each label is "
            "drawn separately, so labelling a whole structure is slow and "
            "unreadable; select the atoms worth naming.",
            len(indices),
            MAX_FORMAL_CHARGE_LABELS,
        )
        return []

    frame = mol.frame
    labels = []
    for i in indices:
        text = " ".join(str(values[i]) for values in resolved)
        labels.append(
            {
                "atom": int(i),
                "position": [float(mol.coords[i, axis, frame]) for axis in range(3)],
                "text": text,
                "size": 0.7 * size,
                "color": "black",
                "offset": 1.0,
            }
        )
    return labels


def _labels(mol, indices=None, size=1.0) -> list[dict]:
    charges = mol.formalcharge
    within = range(len(charges)) if indices is None else indices
    charged = [i for i in within if int(charges[i]) != 0]
    if not charged:
        return []
    if len(charged) > MAX_FORMAL_CHARGE_LABELS:
        logger.warning(
            "Skipping formal charge labels: %d charged atoms exceeds cap %d "
            "(likely a solvated/ionised system; show a prepared structure to "
            "keep labels meaningful).",
            len(charged),
            MAX_FORMAL_CHARGE_LABELS,
        )
        return []
    frame = mol.frame
    labels = []
    for i in charged:
        q = int(charges[i])
        labels.append(
            {
                "atom": int(i),
                "position": [float(mol.coords[i, axis, frame]) for axis in range(3)],
                "text": f"+{q}" if q > 0 else f"{q}",
                "size": 0.7 * size,
                "color": "black",
                "offset": 1.0,
            }
        )
    return labels


def _tubes(mol, highlight_bonds) -> list[dict]:
    tubes = []
    frame = mol.frame
    for sel_a, sel_b in highlight_bonds or []:
        ia = mol.atomselect(sel_a, indexes=True)
        ib = mol.atomselect(sel_b, indexes=True)
        if len(ia) != 1 or len(ib) != 1:
            raise ValueError(
                "highlight_bonds selections must each pick exactly one atom; "
                f"got {len(ia)} for {sel_a!r} and {len(ib)} for {sel_b!r}"
            )
        tubes.append(
            {
                "start": [float(v) for v in mol.coords[int(ia[0]), :, frame]],
                "end": [float(v) for v in mol.coords[int(ib[0]), :, frame]],
                "radius": 0.3,
                "color": "orange",
            }
        )
    return tubes


def focus_sphere(mols, focus_sel=None):
    """The sphere the camera should frame, across every object in the scene.

    With one molecule the browser can work this out from the selected atoms
    itself, but a selection spanning several structures cannot be expressed as
    one set of atom indices, so it is computed here from the coordinates.

    Parameters
    ----------
    mols : list
        The molecules in the scene. The frame used is each molecule's own
        ``mol.frame``.
    focus_sel : str or np.ndarray or None, optional
        Atom selection to frame. None frames everything.

    Returns
    -------
    center : list of float
        Middle of the framed atoms.
    radius : float
        Distance from the centre to the furthest framed atom, never zero.
    """
    points = []
    for mol in mols:
        coords = mol.coords[:, :, mol.frame]
        if focus_sel is not None:
            mask = mol.atomselect(focus_sel)
            if not mask.any():
                continue
            coords = coords[mask]
        points.append(coords)
    if not points:
        return None, None
    stacked = np.vstack(points)
    center = (stacked.min(axis=0) + stacked.max(axis=0)) / 2
    radius = float(np.linalg.norm(stacked - center, axis=1).max())
    # A single atom has no extent; give the camera something to frame.
    return [float(v) for v in center], max(radius, 1.0)


def build_scene(
    mol: "Molecule",
    reps=None,
    *,
    ball_and_stick_sel: "str | np.ndarray | None" = None,
    highlight_bonds: "list[tuple[str, str]] | None" = None,
    focus_sel: "str | np.ndarray | None" = None,
    rotate: "str | tuple[float, float, float] | None" = None,
    zoom: float | None = None,
    background_color: str | None = None,
    fog: float | None = None,
    clip: float | None = None,
) -> dict:
    """Describe the scene for ``mol`` as a plain dict.

    When ``reps`` is empty a cartoon is used for the polymer if ``mol`` has at
    least ``MIN_CARTOON_RESIDUES`` standard polymer residues, with ligands,
    ions, water, branched entities and non-standard residues as ball-and-stick,
    and ball-and-stick throughout otherwise. When ``reps`` is non-empty those
    representations are the whole scene, matching how the VMD and NGL backends
    treat ``mol.reps``.

    Parameters
    ----------
    mol : Molecule
        The molecule whose topology and coordinates drive the scene. The frame
        used for label and tube positions is ``mol.frame``.
    reps : list or None, optional
        Representations to render: normally the ones held in ``mol.reps``,
        together with any one-off representation added by ``view()``'s
        ``sel``, ``style`` and ``color`` arguments. An empty list or None
        gives the automatic scene.
    ball_and_stick_sel : str or np.ndarray or None, optional
        An extra atom selection to additionally draw as ball-and-stick. Ignored
        when it matches no atoms.
    highlight_bonds : list of tuple of (str, str) or None, optional
        Pairs of atom selections, each of which must pick exactly one atom. An
        orange tube is drawn between the two atoms of each pair.
    focus_sel : str or np.ndarray or None, optional
        An atom selection the camera frames on. Ignored when it matches no
        atoms.
    rotate : str or tuple of float or None, optional
        Camera orientation, as a preset name or ``(rx, ry, rz)`` in degrees.
    zoom : float or None, optional
        Camera tightness. Larger values move the camera closer.
    background_color : str or None, optional
        Canvas background as an SVG colour name or hex string.
    fog : float or None, optional
        Depth cueing strength, from 0 for none to 100 for the strongest. Fog
        fades distant geometry into the background colour. None leaves Mol*'s
        own strength.
    clip : float or None, optional
        Half-thickness in Angstrom of the slab drawn around what the camera
        frames. Geometry nearer to or further from the camera than this is cut
        away. None draws the whole structure.

    Returns
    -------
    scene : dict
        The scene description, with ``components`` always present and
        ``labels``, ``tubes``, ``camera`` and ``canvas`` present when they
        carry anything.

    Raises
    ------
    ValueError
        If every representation selection matches no atoms, if a
        ``highlight_bonds`` selection does not pick exactly one atom, if
        ``rotate`` names no known orientation preset, if ``fog`` falls outside
        0 to 100, or if ``clip`` is not positive.
    """
    if fog is not None and not 0 <= float(fog) <= 100:
        raise ValueError(f"fog must be between 0 and 100, got {fog}")
    if clip is not None and float(clip) <= 0:
        raise ValueError(f"clip must be a positive distance, got {clip}")
    if reps:
        # Labels follow the same replace-the-automatic-scene rule as the
        # components: with representations set, charges are labelled only
        # where a FormalCharges representation asks for it.
        components, labels = _components_from_reps(mol, reps)
    else:
        components = _automatic_components(mol)
        labels = _labels(mol)

    if ball_and_stick_sel is not None:
        mask = mol.atomselect(ball_and_stick_sel)
        if mask.any():
            components.append(_ball_and_stick(_atoms(mask.nonzero()[0]), _ELEMENT))

    if not components:
        # Reachable through a FormalCharges representation, which contributes
        # labels and nothing to draw. On its own it renders a blank image:
        # there is no geometry, so nothing anchors the camera either.
        raise ValueError(
            "The representations draw nothing: a FormalCharges representation "
            "labels atoms that another representation has to draw. Add one for "
            "the atoms themselves."
        )

    scene: dict = {"components": components}

    if labels:
        scene["labels"] = labels

    tubes = _tubes(mol, highlight_bonds)
    if tubes:
        scene["tubes"] = tubes

    has_camera = rotate is not None or zoom is not None
    if focus_sel is not None or has_camera:
        camera: dict = {}
        if has_camera:
            direction, up = rotation_to_direction_up(rotate)
            camera["direction"] = list(direction)
            camera["up"] = list(up)
        if zoom is not None:
            camera["radius_factor"] = 1.0 / float(zoom)
        if focus_sel is not None:
            mask = mol.atomselect(focus_sel)
            if mask.any():
                camera["focus"] = _atoms(mask.nonzero()[0])
        if camera:
            # Only alongside a camera: with none, Mol*'s own fit sets the
            # radius from the whole scene and nothing is clipped anyway.
            if clip is not None:
                camera["clip"] = float(clip)
            scene["camera"] = camera

    canvas: dict = {}
    if background_color is not None:
        canvas["background"] = background_color
    if fog is not None:
        canvas["fog"] = float(fog)
    if canvas:
        scene["canvas"] = canvas

    return scene
