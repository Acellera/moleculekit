"""Single source of truth for per-anchor facts about canonical amino-acid
sidechain atoms that participate in covalent crosslinks (a Cys SG bonded
to a scaffold, an Asn ND2 N-glycosylated by a sugar, a Glu CD - Lys NZ
isopeptide, ...).

One row per ``(resname, atom_name)`` anchor.

Keys are ``(resname, atom_name)``. Values are dicts with these fields:

- ``displaced_heavy``: tuple of heavy atom names the crosslink displaces
  from the canonical residue (``("OE2",)`` for a GLU CD isopeptide,
  empty tuple when only a hydrogen is displaced).
- ``smiles_variant``: which ``RESIDUE_SMILES`` key to use when re-templating
  this residue after a crosslink. The variant picks the right protonation
  state so rdkit's valence math gives the correct H count at the junction
  (e.g. ``"LYN"`` for LYS NZ -> neutral secondary amide; ``"HID"`` /
  ``"HIE"`` for HIS).
- ``ff_variant``: the AMBER force-field variant resname this anchor renames
  to when treated via the "well-known FF variant" fast-path (CYS-SG ->
  ``"CYX"``, ASN-ND2 -> ``"NLN"``, LYS-NZ -> ``"LYN"``, ...). ``None`` when
  no fixed FF variant exists - the rename target is then auto-generated
  by :func:`detectNonStandardResidues` as a fresh ``XX#`` name.
- ``drop_h``: list of hydrogen atom names that should not appear on the
  anchored residue after the rename. Read by htmd's defense-in-depth
  rename in ``amber.build`` (when a custombond is passed without specs)
  and informational otherwise.
- ``ff14sb_type``, ``cb_ff14sb_type``, ``hb_ff14sb_type``: the ff14SB
  atom types of the anchor atom / CB-equivalent / CB-attached-H, used by
  htmd's junction-frcmod rewriter to splice ff14SB types back onto the
  canonical-side atom types of cross-junction bonded terms. Absent for
  anchors htmd hasn't yet parameterized (the new GLU/ASP/ASN/GLN amide-
  carbonyl entries) - htmd's lookup tolerates absence and falls back to
  the spec's ``resname`` canonical template.

Variant-to-base resname normalisation (e.g. ``CYX -> CYS``, ``LYN -> LYS``)
is delegated to :data:`moleculekit.residues.ORIGINAL_RESIDUE_NAME_TABLE`.
"""

ANCHOR_TABLE = {
    # ---- Anchors with a well-known AMBER ff14SB variant ---------------
    ("CYS", "SG"): {
        "displaced_heavy": (),
        "smiles_variant": "CYS",
        "ff_variant": "CYX",
        "drop_h": ["HG"],
        "ff14sb_type": "S",
        "cb_ff14sb_type": "2C",
        "hb_ff14sb_type": "H1",
    },
    ("LYS", "NZ"): {
        "displaced_heavy": (),
        "smiles_variant": "LYN",
        "ff_variant": "LYN",
        "drop_h": ["HZ3"],
        "ff14sb_type": "N3",
        "cb_ff14sb_type": "CT",
        "hb_ff14sb_type": "H1",
    },
    ("TYR", "OH"): {
        "displaced_heavy": (),
        "smiles_variant": "TYR",
        "ff_variant": "TYM",
        "drop_h": ["HH"],
        "ff14sb_type": "OH",
        "cb_ff14sb_type": "CA",
        "hb_ff14sb_type": "HA",
    },
    ("HIS", "ND1"): {
        "displaced_heavy": (),
        "smiles_variant": "HID",
        "ff_variant": "HID",
        "drop_h": ["HD1"],
        "ff14sb_type": "NA",
        "cb_ff14sb_type": "CC",
        "hb_ff14sb_type": "H4",
    },
    ("HIS", "NE2"): {
        "displaced_heavy": (),
        "smiles_variant": "HIE",
        "ff_variant": "HIE",
        "drop_h": ["HE2"],
        "ff14sb_type": "NA",
        "cb_ff14sb_type": "CC",
        "hb_ff14sb_type": "H4",
    },
    ("ASN", "ND2"): {
        "displaced_heavy": (),
        "smiles_variant": "ASN",
        "ff_variant": "NLN",
        "drop_h": ["HD22"],
        "ff14sb_type": "N",
        "cb_ff14sb_type": "C",
        "hb_ff14sb_type": "H",
    },
    # ---- Anchors with no well-known FF variant ------------------------
    ("SER", "OG"): {
        "displaced_heavy": (),
        "smiles_variant": "SER",
        "ff_variant": None,
        "drop_h": ["HG"],
        "ff14sb_type": "OH",
        "cb_ff14sb_type": "3C",
        "hb_ff14sb_type": "H1",
    },
    ("THR", "OG1"): {
        "displaced_heavy": (),
        "smiles_variant": "THR",
        "ff_variant": None,
        "drop_h": ["HG1"],
        "ff14sb_type": "OH",
        "cb_ff14sb_type": "3C",
        "hb_ff14sb_type": "H1",
    },
    # ---- New canonical-canonical amide-carbonyl anchors ---------------
    # No ff_variant (renamed to auto-generated X## by detector); no
    # ff14sb_type fields yet (htmd hasn't parameterized these junctions
    # specifically and falls back to resname's canonical
    # template for the splice).
    ("GLU", "CD"): {
        "displaced_heavy": ("OE2",),
        "smiles_variant": "GLU",
        "ff_variant": None,
        "drop_h": [],
    },
    ("ASP", "CG"): {
        "displaced_heavy": ("OD2",),
        "smiles_variant": "ASP",
        "ff_variant": None,
        "drop_h": [],
    },
    ("ASN", "CG"): {
        "displaced_heavy": ("ND2",),
        "smiles_variant": "ASN",
        "ff_variant": None,
        "drop_h": [],
    },
    ("GLN", "CD"): {
        "displaced_heavy": ("NE2",),
        "smiles_variant": "GLN",
        "ff_variant": None,
        "drop_h": [],
    },
}


def lookup_anchor(resname, atom_name):
    """Return :data:`ANCHOR_TABLE` entry for ``(resname, atom_name)``,
    normalizing variant resnames via
    :data:`moleculekit.residues.ORIGINAL_RESIDUE_NAME_TABLE` (e.g.
    ``CYX -> CYS``, ``LYN -> LYS``). Returns ``None`` if no matching
    entry exists.

    Replaces the old ``lookup_anchor_variant``; consumers reading
    ``entry["variant"]`` should switch to ``entry["ff_variant"]``."""
    from moleculekit.residues import ORIGINAL_RESIDUE_NAME_TABLE

    base = ORIGINAL_RESIDUE_NAME_TABLE.get(str(resname), str(resname))
    return ANCHOR_TABLE.get((base, str(atom_name)))


def canonical_anchor_smiles(resname, atom_name):
    """Return the canonical-residue SMILES (from
    :data:`moleculekit.residues.RESIDUE_SMILES`) to use when re-templating
    a residue whose ``atom_name`` participates in a sidechain crosslink.
    Reads :data:`ANCHOR_TABLE`'s ``smiles_variant`` field for the anchor.

    Heavy atoms displaced by the crosslink are NOT removed from the
    SMILES here - they're auto-stripped by
    :func:`moleculekit.rdkittools._try_strip_unmatched_terminals` inside
    :func:`templateResidueFromSmiles` once the residue's cross-residue
    bond is in ``mol.bonds``.

    Raises ``ValueError`` if ``(resname, atom_name)`` is not in
    :data:`ANCHOR_TABLE`."""
    from moleculekit.residues import RESIDUE_SMILES

    entry = lookup_anchor(resname, atom_name)
    if entry is None:
        raise ValueError(
            f"Unsupported canonical-sidechain crosslink anchor "
            f"{resname}-{atom_name}. Add it to "
            f"moleculekit.tools._anchor_variants.ANCHOR_TABLE."
        )
    variant = entry["smiles_variant"]
    smiles = RESIDUE_SMILES.get(variant)
    if smiles is None:
        raise ValueError(
            f"No canonical SMILES for variant {variant!r} "
            f"(anchor {resname}-{atom_name})."
        )
    return smiles
