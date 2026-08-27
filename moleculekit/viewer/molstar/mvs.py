"""Translate the shared scene description (see scene.py) into MolViewSpec.

``mvs_from_scene`` encodes an already-built scene dict and is what the inline
notebook viewer calls (see inline.py's ``build_inline_view``). ``build_mvs``
builds the automatic protein/nucleic + hetero scene from a ``Molecule``
directly (calling ``build_scene`` itself, then ``mvs_from_scene``); it has no
caller left inside this repository, but do not delete it as dead code: it is
the entry point the out-of-repo Acellera Sphinx docs theme depends on to
render structures with no live viewer. The structure data URL is supplied by
the caller in both cases (a published .bcif URL in docs, an inlined data: URL
in notebooks)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from moleculekit.viewer.molstar.scene import (  # noqa: F401  (re-exported)
    BALL_AND_STICK_SIZE_FACTOR,
    DEFAULT_DIRECTION,
    DEFAULT_UP,
    MAX_FORMAL_CHARGE_LABELS,
    MIN_CARTOON_RESIDUES,
    ORIENTATION_PRESETS,
    STANDARD_POLYMER_RESNAMES,
    rotation_to_direction_up,
)

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


def _import_mvs():
    try:
        import molviewspec as mvs
        from molviewspec.nodes import ComponentExpression
    except ImportError as exc:  # pragma: no cover - exercised via packaging
        raise ImportError(
            "The inline molstar viewer needs molviewspec. Install with: "
            "pip install moleculekit[notebook]"
        ) from exc
    return mvs, ComponentExpression


def _serialize(state) -> str:
    return (
        state.dumps()
        if hasattr(state, "dumps")
        else state.model_dump_json(exclude_none=True)
    )


def _selector(select: dict, ComponentExpression):
    """Turn a description selector into an MVS component selector."""
    kind = select["kind"]
    if kind == "builtin":
        return select["name"]
    if kind == "resname":
        return [ComponentExpression(label_comp_id=rn) for rn in select["names"]]
    return [ComponentExpression(atom_index=int(i)) for i in select["indices"]]


def mvs_from_scene(scene: dict, *, structure_url: str) -> str:
    """Encode a scene description as a MolViewSpec (mvsj) JSON string.

    Parameters
    ----------
    scene : dict
        A description as produced by
        :func:`moleculekit.viewer.molstar.scene.build_scene`.
    structure_url : str
        The href the viewer downloads and parses as BinaryCIF.

    Returns
    -------
    mvsj : str
        The serialized MolViewSpec scene as a JSON string.
    """
    mvs, ComponentExpression = _import_mvs()

    builder = mvs.create_builder()
    structure = (
        builder.download(url=structure_url).parse(format="bcif").model_structure()
    )

    for comp in scene["components"]:
        selector = _selector(comp["select"], ComponentExpression)
        rep_kwargs = dict(comp["representation"])
        component = structure.component(selector=selector).representation(**rep_kwargs)
        color = comp.get("color") or {}
        if "theme" in color:
            component.color(custom={"molstar_color_theme_name": color["theme"]})
        elif "uniform" in color:
            component.color(color=color["uniform"])
        if "opacity" in comp:
            component.opacity(opacity=comp["opacity"])

    for tube in scene.get("tubes", []):
        group = structure.primitives(color=tube["color"])
        group.tube(
            start=tuple(tube["start"]),
            end=tuple(tube["end"]),
            radius=tube["radius"],
        )

    camera = scene.get("camera")
    if camera:
        focus = camera.get("focus")
        selector = (
            _selector(focus, ComponentExpression) if focus is not None else "all"
        )
        focus_kwargs = {}
        if "direction" in camera:
            focus_kwargs["direction"] = tuple(camera["direction"])
            focus_kwargs["up"] = tuple(camera["up"])
        if "radius_factor" in camera:
            focus_kwargs["radius_factor"] = camera["radius_factor"]
        structure.component(selector=selector).focus(**focus_kwargs)

    labels = scene.get("labels", [])
    if labels:
        primitives = builder.primitives()
        for label in labels:
            primitives.label(
                position=list(label["position"]),
                text=label["text"],
                label_size=label["size"],
                label_color=label["color"],
                label_offset=label["offset"],
            )

    canvas = scene.get("canvas")
    if canvas and canvas.get("background") is not None:
        builder.canvas(background_color=canvas["background"])

    return _serialize(builder.get_state())


def build_mvs(
    mol: "Molecule",
    *,
    structure_url: str,
    ball_and_stick_sel: str | np.ndarray | None = None,
    representations: list[dict] | None = None,
    highlight_bonds: list[tuple[str, str]] | None = None,
    focus_sel: str | np.ndarray | None = None,
    rotate: str | tuple[float, float, float] | None = None,
    zoom: float | None = None,
    background_color: str | None = None,
) -> str:
    """Build the MolViewSpec (mvsj) JSON string describing the scene for ``mol``.

    A cartoon representation is used for the polymer when ``mol`` has at least
    ``MIN_CARTOON_RESIDUES`` standard polymer residues; ligands, ions, water,
    branched entities and any non-standard residues are drawn as ball-and-stick.
    Otherwise the whole structure is drawn as ball-and-stick. Formal-charge
    labels are added for charged atoms (up to ``MAX_FORMAL_CHARGE_LABELS``).

    Parameters
    ----------
    mol : Molecule
        The molecule whose topology/coordinates drive the scene. The structure
        data itself is fetched by the viewer from ``structure_url``; ``mol`` is
        used here to decide components, resolve selections and place labels.
    structure_url : str
        The href the viewer downloads and parses as BinaryCIF (a published
        ``.bcif`` URL in docs, or an inlined ``data:`` URL in notebooks).
    ball_and_stick_sel : str or np.ndarray or None, optional
        An extra atom selection to additionally draw as ball-and-stick. Ignored
        when it matches no atoms.
    representations : list of dict or None, optional
        Extra representations to add. Each dict may carry ``atom_indices`` or a
        ``sel`` atom selection (one is required to pick atoms), plus ``color``,
        ``opacity`` and any representation keywords (``type`` defaults to
        ``"ball_and_stick"``). ``color`` is ``None`` (element theme), a
        ``{"theme": name}`` dict, or an SVG/hex color string.
    highlight_bonds : list of tuple of (str, str) or None, optional
        Pairs of atom selections, each of which must pick exactly one atom; an
        orange tube primitive is drawn between the two atoms of each pair.
    focus_sel : str or np.ndarray or None, optional
        An atom selection the camera is focused on. Ignored when it matches no
        atoms.
    rotate : str or tuple of float or None, optional
        Camera orientation, as a preset name from ``ORIENTATION_PRESETS`` or a
        tuple of ``(rx, ry, rz)`` rotations in degrees.
    zoom : float or None, optional
        Camera tightness. Larger values move the camera closer. Emitted as the
        reciprocal ``radius_factor``.
    background_color : str or None, optional
        Canvas background as an SVG colour name or hex string.

    Returns
    -------
    mvsj : str
        The serialized MolViewSpec scene as a JSON string.

    Raises
    ------
    ValueError
        If any ``highlight_bonds`` selection does not pick exactly one atom,
        or if ``rotate`` is a string that names no known orientation preset.
    """
    from moleculekit.viewer.molstar.scene import build_scene

    scene = build_scene(
        mol,
        None,  # the automatic scene: this entry point never takes mol.reps
        ball_and_stick_sel=ball_and_stick_sel,
        highlight_bonds=highlight_bonds,
        focus_sel=focus_sel,
        rotate=rotate,
        zoom=zoom,
        background_color=background_color,
    )
    # representations= is additive on purpose: the out-of-repo docs theme layers
    # highlights over the automatic scene and must keep doing so. mol.reps gets
    # replace semantics, but that decision belongs to build_scene's callers.
    for rep in representations or []:
        spec = dict(rep)
        indices = spec.pop("atom_indices", None)
        sel = spec.pop("sel", None)
        color = spec.pop("color", None)
        opacity = spec.pop("opacity", None)
        spec.setdefault("type", "ball_and_stick")
        if indices is None:
            mask = mol.atomselect(sel)
            if not mask.any():
                continue
            indices = [int(i) for i in mask.nonzero()[0]]
        if not indices:
            continue
        component = {
            "select": {"kind": "atoms", "indices": [int(i) for i in indices]},
            "representation": spec,
            "color": {"theme": "element-symbol"}
            if color is None
            else (color if isinstance(color, dict) else {"uniform": color}),
        }
        if opacity is not None:
            component["opacity"] = float(opacity)
        scene["components"].append(component)

    return mvs_from_scene(scene, structure_url=structure_url)
