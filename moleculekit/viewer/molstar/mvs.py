"""Pure MolViewSpec scene builder shared by the Sphinx docs theme and the
inline notebook viewer. Builds the same protein/nucleic + hetero scene
moleculekit's viewer shows. The structure data URL is supplied by the caller
(a published .bcif URL in docs, an inlined data: URL in notebooks)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_FORMAL_CHARGE_LABELS = 200
BALL_AND_STICK_SIZE_FACTOR = 0.6
MIN_CARTOON_RESIDUES = 6
_BALL_AND_STICK_SELECTORS = ("ligand", "ion", "water", "branched")

STANDARD_POLYMER_RESNAMES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
    "LYN", "ARN", "TYM",
    "A", "U", "G", "C", "T", "DA", "DT", "DG", "DC", "DU", "RA", "RU",
    "RG", "RC",
})


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


def _apply_color(component, color):
    """color is None (element theme), a {"theme": name} dict, or an
    SVG/hex string (uniform)."""
    if color is None:
        component.color(custom={"molstar_color_theme_name": "element-symbol"})
    elif isinstance(color, dict) and "theme" in color:
        component.color(custom={"molstar_color_theme_name": color["theme"]})
    else:
        component.color(color=color)


def build_mvs(
    mol,
    *,
    structure_url: str,
    ball_and_stick_sel: str | None = None,
    representations: list[dict] | None = None,
    highlight_bonds: list[tuple[str, str]] | None = None,
    focus_sel: str | None = None,
) -> str:
    """Build the mvsj string for ``mol`` with ``structure_url`` as the download
    href. See the design spec for the representation semantics."""
    mvs, ComponentExpression = _import_mvs()

    builder = mvs.create_builder()
    structure = (
        builder.download(url=structure_url).parse(format="bcif").model_structure()
    )
    if _count_standard_polymer_residues(mol) >= MIN_CARTOON_RESIDUES:
        structure.component(selector="polymer").representation(
            type="cartoon"
        ).color(custom={"molstar_color_theme_name": "secondary-structure"})
        for selector in _BALL_AND_STICK_SELECTORS:
            structure.component(selector=selector).representation(
                type="ball_and_stick", size_factor=BALL_AND_STICK_SIZE_FACTOR
            ).color(custom={"molstar_color_theme_name": "element-symbol"})
        other_resnames = sorted(
            set(mol.resname.tolist()) - STANDARD_POLYMER_RESNAMES
        )
        if other_resnames:
            extra = [ComponentExpression(label_comp_id=rn) for rn in other_resnames]
            structure.component(selector=extra).representation(
                type="ball_and_stick", size_factor=BALL_AND_STICK_SIZE_FACTOR
            ).color(custom={"molstar_color_theme_name": "element-symbol"})
    else:
        structure.component(selector="all").representation(
            type="ball_and_stick", size_factor=BALL_AND_STICK_SIZE_FACTOR
        ).color(custom={"molstar_color_theme_name": "element-symbol"})

    if ball_and_stick_sel is not None:
        mask = mol.atomselect(ball_and_stick_sel)
        if mask.any():
            indices = [int(i) for i in mask.nonzero()[0]]
            extra = [ComponentExpression(atom_index=i) for i in indices]
            structure.component(selector=extra).representation(
                type="ball_and_stick", size_factor=BALL_AND_STICK_SIZE_FACTOR
            ).color(custom={"molstar_color_theme_name": "element-symbol"})

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
        extra = [ComponentExpression(atom_index=int(i)) for i in indices]
        component = structure.component(selector=extra).representation(**spec)
        _apply_color(component, color)
        if opacity is not None:
            component.opacity(opacity=opacity)

    if highlight_bonds:
        bonds_group = structure.primitives(color="orange")
        for sel_a, sel_b in highlight_bonds:
            ia = mol.atomselect(sel_a, indexes=True)
            ib = mol.atomselect(sel_b, indexes=True)
            if len(ia) != 1 or len(ib) != 1:
                raise ValueError(
                    "highlight_bonds selections must each pick exactly one "
                    f"atom; got {len(ia)} for {sel_a!r} and {len(ib)} for "
                    f"{sel_b!r}"
                )
            sa = mol.coords[int(ia[0]), :, 0]
            sb = mol.coords[int(ib[0]), :, 0]
            bonds_group.tube(
                start=(float(sa[0]), float(sa[1]), float(sa[2])),
                end=(float(sb[0]), float(sb[1]), float(sb[2])),
                radius=0.3,
            )

    if focus_sel is not None:
        mask = mol.atomselect(focus_sel)
        if mask.any():
            indices = [int(i) for i in mask.nonzero()[0]]
            structure.component(
                selector=[ComponentExpression(atom_index=int(i)) for i in indices]
            ).focus()

    _add_formal_charge_labels(builder, mol)
    return _serialize(builder.get_state())


def _add_formal_charge_labels(builder, mol) -> None:
    charges = mol.formalcharge
    coords = mol.coords
    charged = [i for i in range(len(charges)) if int(charges[i]) != 0]
    if not charged:
        return
    if len(charged) > MAX_FORMAL_CHARGE_LABELS:
        logger.warning(
            "Skipping formal charge labels: %d charged atoms exceeds cap %d "
            "(likely a solvated/ionised system; show a prepared structure to "
            "keep labels meaningful).",
            len(charged),
            MAX_FORMAL_CHARGE_LABELS,
        )
        return
    primitives = builder.primitives()
    for i in charged:
        q = int(charges[i])
        text = f"+{q}" if q > 0 else f"{q}"
        position = [
            float(coords[i, 0, 0]),
            float(coords[i, 1, 0]),
            float(coords[i, 2, 0]),
        ]
        primitives.label(
            position=position,
            text=text,
            label_size=0.7,
            label_color="black",
            label_offset=1.0,
        )
