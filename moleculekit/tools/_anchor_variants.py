"""Lookup table for canonical-residue anchor atoms in scaffolded peptides.

When a canonical-AA sidechain atom is covalently bonded to a non-peptidic
scaffold (a "scaffolded peptide" / bicyclic peptide), the residue must be
renamed to its deprotonated variant so AMBER's force field skips the displaced
hydrogen and uses the right charges. This table is the single source of truth
shared by moleculekit (`detectNonStandardResidues`, `systemPrepare` via
`force_protonation`) and htmd (`amber.build` defense-in-depth rename, the
junction-frcmod splitter).

Variant-to-base resname normalization (e.g. ``CYX -> CYS``) is reused from
:data:`moleculekit.residues.ORIGINAL_RESIDUE_NAME_TABLE` rather than
duplicated here.

Keys are ``(input_resname, anchor_atom_name)``. Values:
  - ``variant``: the AMBER residue name to use (e.g. ``CYX``). ``None`` means
    no FF variant exists; the bond's valence will be satisfied by the explicit
    bond command at build time and the ``drop_h`` atom must be removed manually.
  - ``drop_h``: list of hydrogen atom names that must not appear on the
    anchored residue.
  - ``ff14sb_type``: ff14SB atom type of the anchor atom in the canonical
    residue.
  - ``cb_ff14sb_type``: ff14SB type of the heavy atom one bond away from the
    anchor (the "CB-equivalent" we cap with a methyl in the model compound).
  - ``hb_ff14sb_type``: ff14SB type of hydrogens attached to that CB-equivalent.
The last three fields are used by htmd's junction-frcmod splitter to rewrite
GAFF2 types of model-compound stub atoms back to the canonical FF types.
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
