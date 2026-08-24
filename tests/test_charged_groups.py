from moleculekit.residues import (
    CHARGED_RESIDUE_ATOMS,
    ANIONIC_ION_RESIDUE_NAMES,
    ION_RESIDUE_NAMES,
    RESIDUE_SMILES,
)
import pytest


# Every resname systemPrepare can put in a returned molecule for a titratable
# residue. Sourced from the protonation table in its docstring plus the
# canonical residues those variants come from.
_TITRATABLE_RESNAMES = [
    "ASP", "ASH",
    "GLU", "GLH",
    "LYS", "LYN", "LSN",
    "ARG", "AR0",
    "HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
    "CYS", "CYM", "CYX",
    "TYR", "TYM",
]

# Excluded from the RESIDUE_SMILES cross-check below, by name and with reason.
# Asserted rather than commented so neither exclusion can rot silently.
_SMILES_CROSSCHECK_EXCLUSIONS = {
    # [S-] stands in for the open valence to the partner cysteine, not a real
    # charge, so the depiction is -1 while the residue is neutral.
    "CYX": "open-valence stand-in",
    # Absent from RESIDUE_SMILES. Adding a key there changes the
    # canonical-residue re-templating path in preparation.py, which is a build
    # path this work must not touch.
    "TYM": "no RESIDUE_SMILES entry",
}


def test_charge_table_covers_every_titratable_resname():
    missing = [r for r in _TITRATABLE_RESNAMES if r not in CHARGED_RESIDUE_ATOMS]
    assert missing == [], f"no charge entry for {missing}"


def test_charge_table_covers_every_variant_of_every_titratable_residue():
    """Derived from PROTEIN_RESIDUES rather than a hardcoded list. A
    protonation variant added to residues.py without a charge entry is
    silently unclassifiable, so the test has to notice on its own."""
    from moleculekit.residues import PROTEIN_RESIDUES

    titratable_parents = {"ASP", "GLU", "HIS", "LYS", "ARG", "TYR", "CYS"}
    missing = []
    for rr in PROTEIN_RESIDUES:
        if rr.resname not in titratable_parents:
            continue
        for name in (rr.resname,) + tuple(rr.resname_variants):
            if name not in CHARGED_RESIDUE_ATOMS:
                missing.append(name)
    assert missing == [], f"no charge entry for {missing}"


def test_charge_table_agrees_with_residue_smiles():
    from rdkit import Chem

    checked = 0
    for resn in _TITRATABLE_RESNAMES:
        if resn in _SMILES_CROSSCHECK_EXCLUSIONS:
            continue
        smiles = RESIDUE_SMILES[resn]
        got = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))
        assert got == CHARGED_RESIDUE_ATOMS[resn].charge, (
            f"{resn}: table says {CHARGED_RESIDUE_ATOMS[resn].charge}, "
            f"RESIDUE_SMILES says {got}"
        )
        checked += 1
    assert checked == len(_TITRATABLE_RESNAMES) - len(_SMILES_CROSSCHECK_EXCLUSIONS)


def test_smiles_crosscheck_exclusions_are_still_warranted():
    from rdkit import Chem

    # CYX is excluded because its depiction disagrees. If someone corrects that
    # SMILES, this fails and the exclusion should be dropped.
    cyx = Chem.GetFormalCharge(Chem.MolFromSmiles(RESIDUE_SMILES["CYX"]))
    assert cyx != CHARGED_RESIDUE_ATOMS["CYX"].charge

    # TYM is excluded because it has no entry. If one is added, this fails and
    # the exclusion should be dropped.
    assert "TYM" not in RESIDUE_SMILES


def test_charged_entries_name_their_atoms_and_neutral_ones_do_not():
    for resn, entry in CHARGED_RESIDUE_ATOMS.items():
        if entry.charge == 0:
            assert entry.atoms == (), f"{resn} is neutral but names atoms"
            assert entry.center is None, f"{resn} is neutral but names a center"
        else:
            assert entry.atoms, f"{resn} is charged but names no atoms"
            assert entry.center is not None, f"{resn} is charged but names no center"


def test_every_charged_table_entry_has_a_report_label():
    """The label lookup in chargedGroups is unconditional, so a charged
    resname in the table with no label raises KeyError in the middle of a scan.
    HSP did exactly that: it sat in the table next to a function that died on
    it. Derived from the table so a future addition fails here rather than in a
    user's scan."""
    from moleculekit.tools.charged_groups import _CHARGE_LABELS

    required = {
        (resn, entry.charge)
        for resn, entry in CHARGED_RESIDUE_ATOMS.items()
        if entry.charge != 0
    }
    missing = sorted(required - set(_CHARGE_LABELS))
    assert missing == [], f"no report label for {missing}"


def test_charge_labels_name_nothing_the_table_does_not():
    """The reverse direction, so a label left behind by a removed table entry
    does not sit there looking supported."""
    from moleculekit.tools.charged_groups import _CHARGE_LABELS

    for resn, charge in _CHARGE_LABELS:
        assert resn in CHARGED_RESIDUE_ATOMS, f"{resn} is labelled but not in the table"
        assert CHARGED_RESIDUE_ATOMS[resn].charge == charge, (
            f"{resn} is labelled at {charge:+d} but the table says "
            f"{CHARGED_RESIDUE_ATOMS[resn].charge:+d}"
        )


def test_anionic_ions_are_a_subset_of_known_ions():
    assert ANIONIC_ION_RESIDUE_NAMES <= ION_RESIDUE_NAMES


def test_anionic_ions_hold_the_halides_present_in_ion_residue_names():
    for resn in ("CL", "CLA", "IOD"):
        assert resn in ANIONIC_ION_RESIDUE_NAMES


def test_cationic_ions_are_a_subset_of_known_ions():
    from moleculekit.residues import (
        CATIONIC_ION_RESIDUE_NAMES,
        METAL_ION_RESIDUE_NAMES,
    )

    assert CATIONIC_ION_RESIDUE_NAMES <= ION_RESIDUE_NAMES
    # Kept OUT of METAL_ION_RESIDUE_NAMES on purpose: that set is defined as
    # the element-symbol naming convention and autoSegment consumes it to
    # classify residues, so widening it would change segmentation.
    assert not (CATIONIC_ION_RESIDUE_NAMES & METAL_ION_RESIDUE_NAMES)


def test_every_ion_code_is_classified_or_polyatomic():
    """An ion code in none of the three sign sets produces no ChargedGroup, so
    reviewProtonation's metal scan never sees it and its metal rule silently
    never fires for it. The unreported charge is disclosed via unclassified;
    the unreported SELECTION is disclosed nowhere. Listing the survivors
    explicitly so a new code added to ION_RESIDUE_NAMES without a
    classification fails here instead."""
    from moleculekit.residues import (
        CATIONIC_ION_RESIDUE_NAMES,
        METAL_ION_RESIDUE_NAMES,
    )

    unclassified = (
        ION_RESIDUE_NAMES
        - METAL_ION_RESIDUE_NAMES
        - CATIONIC_ION_RESIDUE_NAMES
        - ANIONIC_ION_RESIDUE_NAMES
    )
    # Every survivor is polyatomic, so the single-atom guard in the ion rule
    # excludes it anyway and no sign can be claimed for it:
    #   CUA  the dinuclear Cu-A centre
    #   MO3, MO5, MO6  molybdenum oxide clusters
    #   NAW  a sodium ion with its coordinated water
    #   OC7  a calcium ion with its seven coordinated waters
    assert sorted(unclassified) == ["CUA", "MO3", "MO5", "MO6", "NAW", "OC7"]


import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def prepared_3ptb():
    """3PTB with hydrogens and protonation states assigned. Module-scoped
    because systemPrepare is slow; tests that mutate it must copy first."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    pmol, _ = systemPrepare(mol, pH=7.4, _logger_level="ERROR")
    return pmol


def test_charged_groups_finds_aspartate_carboxylate(prepared_3ptb):
    from moleculekit.tools.charged_groups import chargedGroups

    groups, _ = chargedGroups(prepared_3ptb)
    asp = [g for g in groups if g.resname == "ASP" and g.source == "table"]
    assert len(asp) > 0
    g = asp[0]
    assert g.charge == -1
    assert g.label == "carboxylate"
    names = set(prepared_3ptb.name[g.atoms])
    assert names == {"OD1", "OD2"}
    assert prepared_3ptb.name[g.center] == "CG"


def test_charged_groups_skips_neutral_table_entries(prepared_3ptb):
    from moleculekit.tools.charged_groups import chargedGroups

    groups, _ = chargedGroups(prepared_3ptb)
    # HIE / HID / TYR / CYS are in the table at charge 0 and must yield no group
    for resn in ("HIE", "HID", "TYR", "CYS"):
        assert not [g for g in groups if g.resname == resn and g.source == "table"]


def test_charged_groups_finds_free_termini(prepared_3ptb):
    from moleculekit.tools.charged_groups import chargedGroups

    groups, _ = chargedGroups(prepared_3ptb)
    termini = [g for g in groups if g.source == "terminus"]
    # Verified against this exact fixture: prepared 3PTB is one chain A, with a
    # single OXT on ASN 245 and H1/H2/H3 on the N of ILE 16. So exactly one
    # C-terminal carboxylate and one N-terminal ammonium.
    assert sorted(g.charge for g in termini) == [-1, 1]
    cterm = [g for g in termini if g.charge == -1][0]
    assert set(prepared_3ptb.name[cterm.atoms]) == {"O", "OXT"}
    nterm = [g for g in termini if g.charge == 1][0]
    assert prepared_3ptb.name[nterm.center] == "N"
    assert nterm.label == "ammonium"


def test_n_terminal_lysine_yields_both_its_groups(prepared_3ptb):
    """The load-bearing case: a residue with a table entry AND a terminus must
    return both groups, not just the one a priority lookup would find first."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    # Make the N-terminal residue a lysine so it carries a sidechain charge too
    first_resid = int(mol.resid[mol.atomselect("protein", indexes=True)[0]])
    mol.mutateResidue(f"protein and resid {first_resid}", "LYS")
    pmol, _ = systemPrepare(mol, pH=7.4, _logger_level="ERROR")

    groups, _ = chargedGroups(pmol)
    at_first = [g for g in groups if g.resid == first_resid]
    sources = sorted(g.source for g in at_first)
    assert sources == ["table", "terminus"], (
        f"expected both the sidechain and the terminal group, got {sources}"
    )


def test_charged_groups_finds_nucleic_phosphates():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "1bna.pdb"))
    groups, _ = chargedGroups(mol)
    phos = [g for g in groups if g.source == "phosphate"]
    assert len(phos) > 0
    g = phos[0]
    assert g.charge == -1
    assert g.label == "phosphate"
    assert set(mol.name[g.atoms]) <= {"OP1", "OP2", "O1P", "O2P"}
    assert mol.name[g.center] in ("OP2", "O2P")


def test_charged_groups_honours_sel(prepared_3ptb):
    from moleculekit.tools.charged_groups import chargedGroups

    all_groups, _ = chargedGroups(prepared_3ptb)
    asp_groups, _ = chargedGroups(prepared_3ptb, prepared_3ptb.resname == "ASP")
    assert len(asp_groups) < len(all_groups)
    assert {g.resname for g in asp_groups} == {"ASP"}


def test_a_ligand_is_not_mistaken_for_an_n_terminus():
    """A non-polymer residue with an atom named N and SMILES-style sequential
    hydrogens must not be reported as an N-terminal ammonium. A fabricated +1
    near a titratable site is the exact error class this module prevents."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    ben = mol.atomselect("resname BEN", indexes=True)
    # Rename BEN's atoms into the shape that would trip a naive name match
    mol.name[ben[0]] = "N"
    mol.name[ben[1]] = "H1"
    mol.name[ben[2]] = "H2"
    mol.name[ben[3]] = "H3"
    groups, _ = chargedGroups(mol)
    assert not [g for g in groups if g.resname == "BEN" and g.source == "terminus"]


def test_a_capped_c_terminus_yields_no_terminus_group():
    """The mirror of the ligand case. NME in the tleap convention is
    N, H, C, H1, H2, H3, so four of its atom names are N-terminal ammonium
    names and a name-count rule fires on a neutral C-terminal amide cap. That
    fabricates a +1 into the charge column of every titratable residue near the
    capped terminus. ACE has no N at all, and NME / NMA / NHE / NH2 are neutral
    amides, so no capping group can carry a real terminal charge."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups
    from moleculekit.residues import CAP_RESIDUE_NAMES

    for rel in (
        ("test_bondguesser", "3ptb_solvated.pdb"),
        ("test_readers", "binpos", "alanine-dipeptide-explicit.pdb"),
    ):
        mol = Molecule(os.path.join(curr_dir, *rel))
        caps = mol.atomselect(f"resname {' '.join(CAP_RESIDUE_NAMES)}", indexes=True)
        assert len(caps) > 0, f"{rel} must contain a capping group"
        # The exact shape that trips a name count: NME's N carries H, H1, H2, H3
        nme = mol.atomselect("resname NME", indexes=True)
        assert {"H", "H1", "H2", "H3"} <= set(mol.name[nme].tolist())

        groups, _ = chargedGroups(mol)
        fabricated = [
            g
            for g in groups
            if g.resname in CAP_RESIDUE_NAMES and g.source == "terminus"
        ]
        assert fabricated == [], (
            f"{rel}: a neutral cap was reported as "
            f"{[(g.resname, g.resid, g.charge, g.label) for g in fabricated]}"
        )


def test_charged_groups_reads_charmm_histidine_names():
    """The consumer test the table-only tests could not stand in for. HSP was
    in CHARGED_RESIDUE_ATOMS and chargedGroups raised KeyError on it, because
    no test ever ran the scanner over a structure using CHARMM residue
    names."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "test_writers", "villin.pdb"))
    assert (mol.resname == "HSP").any(), "villin fixture must contain an HSP"
    groups, _ = chargedGroups(mol)
    hsp = [g for g in groups if g.resname == "HSP"]
    assert hsp, "every HSP is charged and must yield a group"
    for g in hsp:
        assert g.charge == 1
        assert g.label == "imidazolium"
        assert g.source == "table"
        assert set(mol.name[g.atoms]) == {"ND1", "NE2"}
        assert mol.name[g.center] == "CE1"


def test_a_truncated_charged_sidechain_is_reported_unclassified():
    """A residue the table says is charged, whose charge-carrying atoms are not
    in the file, can be measured against nothing. It must still be visible:
    yielding neither a group nor an unclassified entry is total silence about a
    charge, in the module whose whole thesis is that silence is the bug."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    asp = mol.atomselect("resname ASP and name OD1 OD2", indexes=True)
    resid = int(mol.resid[asp[0]])
    chain = str(mol.chain[asp[0]])
    # Unmodelled sidechain density: the carboxylate oxygens are simply absent
    mol.remove(
        (mol.resid == resid) & (mol.chain == chain) & np.isin(mol.name, ["OD1", "OD2"]),
        _logger=False,
    )
    groups, unclassified = chargedGroups(mol)
    assert not [g for g in groups if g.resid == resid and g.chain == chain]
    assert ("ASP", resid, "", chain) in unclassified


def test_a_truncated_charged_sidechain_is_reported_on_a_real_fixture():
    """The same case as deposited. 8QFZ carries one GLU whose carboxylate is
    unmodelled."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "test_autosegment", "8qfz.pdb"))
    _, unclassified = chargedGroups(mol)
    assert [u for u in unclassified if u[0] == "GLU" and u[1] == 62]


def test_charged_group_equality_does_not_raise():
    """ChargedGroup holds a numpy array, so a generated field-by-field __eq__
    returns an array and any truth test on it raises. A list of groups is
    searched and compared all over the review code, so == has to work."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    groups, _ = chargedGroups(mol)
    assert len(groups) > 1
    assert groups[0] == groups[0]
    assert groups[0] != groups[1]
    assert groups[0] in groups
    assert len({id(g) for g in groups}) == len(groups)


def test_charged_groups_classifies_a_charmm_ion_name():
    """CAL / CES / POT / SOD are the CHARMM ion names and CU1 / MN3 are
    deposited codes for Cu(I) and Mn(III). None is an element symbol, so
    METAL_ION_RESIDUE_NAMES does not hold them, and before they were
    classified they produced no group at all: reviewProtonation's metal scan
    then saw no metal and its metal rule silently never fired."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups
    from moleculekit.residues import CATIONIC_ION_RESIDUE_NAMES

    for resn in sorted(CATIONIC_ION_RESIDUE_NAMES):
        mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
        mol.remove("water", _logger=False)
        ca = mol.atomselect("resname CA and not protein", indexes=True)
        assert len(ca) == 1, "3ptb fixture must carry its single calcium ion"
        mol.resname[ca] = resn
        groups, unclassified = chargedGroups(mol)
        ion = [g for g in groups if g.resname == resn]
        assert len(ion) == 1, f"{resn} produced no charged group"
        assert ion[0].charge > 0
        assert ion[0].source == "ion"
        assert ion[0].sign_only, "an ion's charge is a sign, not a magnitude"
        assert not [u for u in unclassified if u[0] == resn]


def test_a_polyatomic_residue_with_a_cation_code_is_not_called_a_cation():
    """The single-atom guard. CAL and ZN1 also name unrelated polyatomic
    entries in the PDB chemical component dictionary, so the guard is what
    keeps a real molecule out of the cation branch."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    ben = mol.atomselect("resname BEN", indexes=True)
    assert len(ben) > 1
    mol.resname[ben] = "CAL"
    groups, unclassified = chargedGroups(mol)
    assert not [g for g in groups if g.resname == "CAL"]
    assert [u for u in unclassified if u[0] == "CAL"]


def test_charged_groups_finds_a_metal_cation():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    groups, _ = chargedGroups(mol)
    zn = [g for g in groups if g.resname == "ZN"]
    assert len(zn) == 1
    assert zn[0].charge > 0
    assert zn[0].label == "metal cation"
    assert zn[0].source == "ion"


def test_charged_groups_reads_assigned_formal_charges():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    ben = mol.atomselect("resname BEN", indexes=True)
    assert len(ben) > 0, "3ptb fixture must contain the benzamidine"
    # A freshly read structure carries no formal charges, so BEN is
    # unclassified. This is the stage-dependence the report has to surface.
    groups, unclassified = chargedGroups(mol)
    assert not [g for g in groups if g.resname == "BEN"]
    assert ("BEN", int(mol.resid[ben[0]]), str(mol.insertion[ben[0]]),
            str(mol.chain[ben[0]])) in unclassified

    # Assign one, and it becomes a group from the formalcharge source.
    amidinium_n = mol.atomselect("resname BEN and name N1", indexes=True)
    mol.formalcharge[amidinium_n[0]] = 1
    groups, unclassified = chargedGroups(mol)
    ben_groups = [g for g in groups if g.resname == "BEN"]
    assert len(ben_groups) == 1
    assert ben_groups[0].charge == 1
    assert ben_groups[0].source == "formalcharge"
    assert ben_groups[0].label == "formal charge"
    assert not [u for u in unclassified if u[0] == "BEN"]


def test_formal_charge_groups_split_by_sign():
    """A residue carrying both a + and a - formal charge yields two groups, so
    a zwitterion is not collapsed into a single misleading net charge."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    ben = mol.atomselect("resname BEN", indexes=True)
    mol.formalcharge[ben[0]] = 1
    mol.formalcharge[ben[1]] = -1
    groups, _ = chargedGroups(mol)
    ben_groups = sorted(
        (g for g in groups if g.resname == "BEN"), key=lambda g: g.charge
    )
    assert [g.charge for g in ben_groups] == [-1, 1]


def test_water_is_not_reported_unclassified():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    _, unclassified = chargedGroups(mol)
    assert not [u for u in unclassified if u[0] in ("HOH", "WAT", "TIP3")]


def test_ordinary_amino_acids_are_not_reported_unclassified():
    """The unclassified list must name residues whose charge could not be
    determined, not every neutral residue in the structure. CHARGED_RESIDUE_ATOMS
    holds only the titratable resnames, so ALA, GLY, SER and the rest are absent
    from it: keying the unclassified test on that table would report about 150 of
    prepared 3PTB's 223 residues and bury the signal."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    _, unclassified = chargedGroups(mol)
    reported = {u[0] for u in unclassified}
    for resn in ("ALA", "GLY", "SER", "THR", "ASN", "GLN", "LEU", "ILE", "VAL",
                 "PHE", "TRP", "MET", "PRO", "CYS", "TYR", "HIS", "LYS", "ARG",
                 "ASP", "GLU"):
        assert resn not in reported, f"{resn} is a known residue, not unclassified"
    # The genuinely unknown ones ARE reported
    assert "BEN" in reported
    assert len(unclassified) < 10, (
        f"unclassified should be a short, readable list, got {len(unclassified)}: "
        f"{sorted(reported)}"
    )


def test_unknown_ion_code_is_unclassified_not_guessed():
    """An ion code in none of the three sign sets must not be guessed as a
    cation. A guessed sign is a wrong charge entering a report."""
    from moleculekit.molecule import Molecule
    from moleculekit.residues import (
        ION_RESIDUE_NAMES,
        METAL_ION_RESIDUE_NAMES,
        CATIONIC_ION_RESIDUE_NAMES,
        ANIONIC_ION_RESIDUE_NAMES,
    )
    from moleculekit.tools.charged_groups import chargedGroups

    unknown = sorted(
        ION_RESIDUE_NAMES
        - METAL_ION_RESIDUE_NAMES
        - CATIONIC_ION_RESIDUE_NAMES
        - ANIONIC_ION_RESIDUE_NAMES
    )
    if not unknown:
        pytest.skip("every ION_RESIDUE_NAMES code is classified")

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    ben = mol.atomselect("resname BEN", indexes=True)
    mol.resname[ben[0]] = unknown[0]
    mol.resid[ben[0]] = 9999
    groups, unclassified = chargedGroups(mol)
    assert not [g for g in groups if g.resname == unknown[0]]
    assert [u for u in unclassified if u[0] == unknown[0]]


def test_assigned_formal_charge_is_read_on_a_neutral_table_residue():
    """A CYS has a neutral table entry, but an assigned -1 on its SG is
    direct evidence of a thiolate. Gating formal charges by resname would
    report neither a group nor an unclassified entry."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    sg = mol.atomselect("resname CYS and name SG", indexes=True)
    assert len(sg) > 0
    mol.formalcharge[sg[0]] = -1
    groups, _ = chargedGroups(mol)
    hits = [g for g in groups if g.source == "formalcharge" and sg[0] in g.atoms]
    assert len(hits) == 1
    assert hits[0].charge == -1


def test_a_charged_table_residue_is_not_double_counted():
    """A prepared ASP whose OD1/OD2 also carry assigned formal charges must
    yield ONE group, not one per source."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    od = mol.atomselect("resname ASP and name OD1 OD2", indexes=True)
    assert len(od) >= 2
    mol.formalcharge[od[0]] = -1
    groups, _ = chargedGroups(mol)
    rid = int(mol.resid[od[0]])
    ch = str(mol.chain[od[0]])
    same = [g for g in groups if g.resid == rid and g.chain == ch]
    assert len(same) == 1, f"expected one group, got {[g.source for g in same]}"
    assert same[0].source == "table"


def test_charged_lipids_are_reported_unclassified():
    """A phospholipid head group carries real charge that no source models,
    so a membrane system must SAY it could not classify them rather than
    silently reporting no lipid charges."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups

    mol = Molecule(
        os.path.join(curr_dir, "test_autosegment", "membrane.pdb")
    )
    _, unclassified = chargedGroups(mol)
    reported = {u[0] for u in unclassified}
    assert "POPC" in reported
    # water in the same fixture stays out of it
    assert not [u for u in unclassified if u[0] in ("TIP3", "HOH", "WAT")]


# Charge centers get_protein_charged returns for tests/test_interactions/
# 3PTB_prepared.pdb, measured on that exact fixture. Bovine trypsin: 14 LYS
# NZ, 2 ARG CZ and 1 HIP CE1 positive; 6 ASP CG and 4 GLU CD negative.
_3PTB_PREPARED_POS = [
    559, 604, 696, 991, 1308, 1349, 1457, 1810, 1968, 2019, 2158, 2451, 2641,
    2838, 2874, 2968, 3121,
]
_3PTB_PREPARED_NEG = [744, 758, 850, 886, 1219, 1918, 2111, 2420, 2470, 2527]


def test_get_protein_charged_pins_its_output_on_a_fixture():
    """The charge centers must come from CHARGED_RESIDUE_ATOMS, so the library
    holds one charge table rather than two that can disagree. Pinned as exact
    index sets, and cross-checked against a selection written independently of
    the table: asserting only that the returned indices have the resname and
    name they were selected by restates the implementation and cannot fail."""
    from moleculekit.molecule import Molecule
    from moleculekit.interactions.interactions import get_protein_charged

    mol = Molecule(
        os.path.join(curr_dir, "test_interactions", "3PTB_prepared.pdb")
    )
    pos, neg = get_protein_charged(mol)
    assert pos.dtype == np.uint32 and neg.dtype == np.uint32

    assert sorted(int(i) for i in pos) == _3PTB_PREPARED_POS
    assert sorted(int(i) for i in neg) == _3PTB_PREPARED_NEG

    # Independently written: one selection string per sign rather than a walk
    # over the table, so a table entry silently dropped from the consumer is
    # caught here and not just by the pinned lists above.
    indep_pos = mol.atomselect(
        "protein and ((resname LYS and name NZ) or (resname ARG and name CZ) "
        "or (resname HIP HSP and name CE1))",
        indexes=True,
    )
    indep_neg = mol.atomselect(
        "protein and ((resname ASP and name CG) or (resname GLU and name CD))",
        indexes=True,
    )
    assert sorted(int(i) for i in indep_pos) == _3PTB_PREPARED_POS
    assert sorted(int(i) for i in indep_neg) == _3PTB_PREPARED_NEG


def test_get_protein_charged_now_sees_cym():
    """CYM was invisible to the old hardcoded table. A prepared structure can
    carry it, and a missed thiolate is a missed salt bridge."""
    from moleculekit.molecule import Molecule
    from moleculekit.interactions.interactions import get_protein_charged

    mol = Molecule(
        os.path.join(curr_dir, "test_interactions", "2P95_prepared.pdb")
    )
    cym = mol.atomselect("resname CYM and name SG", indexes=True)
    assert len(cym) > 0, "2P95_prepared fixture must contain a CYM"
    _, neg = get_protein_charged(mol)
    assert set(int(i) for i in cym) <= set(int(i) for i in neg)
