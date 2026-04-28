"""Lookup table mapping canonical amino-acid sidechain atoms to their
"deprotonated anchor" AMBER residue variants.

When a canonical amino-acid sidechain atom is covalently bonded to a
non-canonical residue (for example a Cys SG bonded to a scaffold carbon
in a bicyclic peptide, or an Asn ND2 N-glycosylated by a sugar), the
residue must be renamed to a force-field variant in which the displaced
hydrogen is absent and the partial charges are set up for the bonded
state. Without the rename, AMBER's standard residue template would place
an extra hydrogen on the anchor atom and use the wrong charges.

This table is the single source of truth for those rename rules. It is
shared by:
  - moleculekit's :func:`detectNonStandardResidues` and
    :func:`forceProtonationFromSpecs` (which feed
    ``systemPrepare(..., force_protonation=...)``);
  - htmd's ``amber.build`` defense-in-depth rename (which applies the
    same rename when the user supplies ``custombonds`` directly without
    going through ``systemPrepare``);
  - htmd's helper that splits an antechamber-generated frcmod into
    scaffold-internal and junction terms (the ``ff14sb_type`` /
    ``cb_ff14sb_type`` / ``hb_ff14sb_type`` fields tell that helper which
    canonical force-field atom types to rewrite the GAFF2 stub-atom
    types back to).

Variant-to-base resname normalisation (e.g. ``CYX -> CYS``) is reused
from :data:`moleculekit.residues.ORIGINAL_RESIDUE_NAME_TABLE` rather
than duplicated here.

Keys are ``(input_resname, anchor_atom_name)``. Each value is a dict:
  - ``variant``: the AMBER residue name to use after the rename (e.g.
    ``"CYX"``). ``None`` means no force-field variant exists for this
    anchor. In that case the bond's valence is satisfied solely by the
    explicit bond command emitted at build time, and the ``drop_h`` atom
    must be removed manually before building.
  - ``drop_h``: list of hydrogen atom names that must not appear on the
    anchored residue (the H atoms displaced by forming the new bond).
  - ``ff14sb_type``: ff14SB atom type of the anchor atom in the canonical
    residue (e.g. ``"S"`` for Cys SG, ``"N"`` for Asn ND2).
  - ``cb_ff14sb_type``: ff14SB type of the heavy atom one bond closer to
    the backbone than the anchor (the "CB-equivalent" the model-compound
    builder caps with a methyl group).
  - ``hb_ff14sb_type``: ff14SB type of hydrogens attached to that
    CB-equivalent atom.
"""

ANCHOR_VARIANTS = {
    ("CYS", "SG"): {
        "variant": "CYX",
        "drop_h": ["HG"],
        "ff14sb_type": "S",
        "cb_ff14sb_type": "2C",
        "hb_ff14sb_type": "H1",
    },
    ("LYS", "NZ"): {
        "variant": "LYN",
        "drop_h": ["HZ3"],
        "ff14sb_type": "N3",
        "cb_ff14sb_type": "CT",
        "hb_ff14sb_type": "H1",
    },
    ("TYR", "OH"): {
        "variant": "TYM",
        "drop_h": ["HH"],
        "ff14sb_type": "OH",
        "cb_ff14sb_type": "CA",
        "hb_ff14sb_type": "HA",
    },
    ("HIS", "ND1"): {
        "variant": "HID",
        "drop_h": ["HD1"],
        "ff14sb_type": "NA",
        "cb_ff14sb_type": "CC",
        "hb_ff14sb_type": "H4",
    },
    ("HIS", "NE2"): {
        "variant": "HIE",
        "drop_h": ["HE2"],
        "ff14sb_type": "NA",
        "cb_ff14sb_type": "CC",
        "hb_ff14sb_type": "H4",
    },
    ("SER", "OG"): {
        "variant": None,
        "drop_h": ["HG"],
        "ff14sb_type": "OH",
        "cb_ff14sb_type": "3C",
        "hb_ff14sb_type": "H1",
    },
    ("THR", "OG1"): {
        "variant": None,
        "drop_h": ["HG1"],
        "ff14sb_type": "OH",
        "cb_ff14sb_type": "3C",
        "hb_ff14sb_type": "H1",
    },
    # Asn N-glycosylation: ND2 loses one of its two amide hydrogens to
    # accept a glycosidic C-N bond from a sugar's anomeric carbon. NLN is
    # the AMBER glycoprotein-junction residue (provided by GLYCAM); GLYCAM
    # must be loaded by amber.build for this rename to resolve.
    ("ASN", "ND2"): {
        "variant": "NLN",
        "drop_h": ["HD22"],
        "ff14sb_type": "N",
        "cb_ff14sb_type": "C",
        "hb_ff14sb_type": "H",
    },
}


def lookup_anchor_variant(resname, atom_name):
    """Return the :data:`ANCHOR_VARIANTS` entry for ``(resname, atom_name)``,
    normalizing variant resnames to their base via
    :data:`moleculekit.residues.ORIGINAL_RESIDUE_NAME_TABLE` (e.g. ``CYX -> CYS``).
    Returns ``None`` if no matching entry exists."""
    from moleculekit.residues import ORIGINAL_RESIDUE_NAME_TABLE

    resname = str(resname)
    atom_name = str(atom_name)
    base = ORIGINAL_RESIDUE_NAME_TABLE.get(resname, resname)
    return ANCHOR_VARIANTS.get((base, atom_name))
