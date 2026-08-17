import pytest


def test_glycam_resname_construction():
    from moleculekit.tools.glycans import glycamResname

    assert glycamResname("NAG", ()) == "0YB"
    assert glycamResname("NAG", (4,)) == "4YB"
    assert glycamResname("NAG", (4, 6)) == "UYB"
    assert glycamResname("NDG", ()) == "0YA"
    assert glycamResname("BMA", (3, 6)) == "VMB"
    assert glycamResname("MAN", (2,)) == "2MA"
    assert glycamResname("MAN", (3, 6)) == "VMA"
    assert glycamResname("FUC", ()) == "0fA"
    assert glycamResname("SIA", ()) == "0SA"
    assert glycamResname("SIA", (8,)) == "8SA"
    assert glycamResname("NGA", (3, 6)) == "VVB"
    assert glycamResname("GAL", (3,)) == "3LB"
    assert glycamResname("XYP", ()) == "0XB"


def test_glycam_resname_fructose():
    from moleculekit.tools.glycans import GLYCAM_UNIT_NAMES, glycamResname

    cases = [((), "0CU"), ((1,), "1CU"), ((3, 6), "VCU")]
    for linked_positions, expected in cases:
        name = glycamResname("FRU", linked_positions)
        assert name == expected
        assert name in GLYCAM_UNIT_NAMES


def test_glycam_resname_uronic_acids():
    from moleculekit.tools.glycans import GLYCAM_UNIT_NAMES, glycamResname

    assert glycamResname("BDP", ()) == "0ZB"  # beta-D-glucuronic acid
    assert glycamResname("GCU", ()) == "0ZA"  # alpha-D-glucuronic acid
    assert glycamResname("GTR", ()) == "0OB"  # beta-D-galacturonic acid
    assert glycamResname("ADA", ()) == "0OA"  # alpha-D-galacturonic acid
    assert glycamResname("IDR", ()) == "0uA"  # alpha-L-iduronic acid
    name = glycamResname("BDP", (4,))
    assert name == "4ZB"
    assert name in GLYCAM_UNIT_NAMES
    for resname, expected in [
        ("BDP", "0ZB"),
        ("GCU", "0ZA"),
        ("GTR", "0OB"),
        ("ADA", "0OA"),
        ("IDR", "0uA"),
    ]:
        name = glycamResname(resname, ())
        assert name == expected
        assert name in GLYCAM_UNIT_NAMES


def test_glycam_resname_unsupported_uronic_acids():
    """GLYCAM 06j ships units for only three uronic acids (glucuronic,
    galacturonic and L-iduronic), each in a single configuration. It has no
    unit for mannuronic acid (PDB ``BEM``) or guluronic acid (PDB ``LGU``),
    so these two stay unmapped in :data:`GLYCAM_SUGARS`. This is a
    force-field coverage gap, not a missing table entry: inventing a letter
    for them would silently build the wrong chemistry.
    """
    from moleculekit.tools.glycans import GLYCAM_SUGARS, glycamResname

    assert "BEM" not in GLYCAM_SUGARS
    assert "LGU" not in GLYCAM_SUGARS
    with pytest.raises(RuntimeError, match="BEM"):
        glycamResname("BEM", ())
    with pytest.raises(RuntimeError, match="LGU"):
        glycamResname("LGU", ())


def test_glycam_resname_unknown_sugar():
    from moleculekit.tools.glycans import glycamResname

    with pytest.raises(RuntimeError, match="XXX"):
        glycamResname("XXX", (4,))


def test_glycam_resname_invalid_linkage():
    from moleculekit.tools.glycans import glycamResname

    # GLYCAM ships no 2-linked GlcNAc (the acetyl occupies position 2)
    with pytest.raises(RuntimeError, match="2YB"):
        glycamResname("NAG", (2,))
    # No letter exists for a {8, 9} combination
    with pytest.raises(RuntimeError, match="SIA"):
        glycamResname("SIA", (8, 9))


def test_linked_positions_roundtrip():
    from moleculekit.tools.glycans import (
        GLYCAM_UNIT_NAMES,
        linkedPositionsFromGlycamResname,
    )

    assert linkedPositionsFromGlycamResname("0YB") == ()
    assert linkedPositionsFromGlycamResname("4YB") == (4,)
    assert linkedPositionsFromGlycamResname("UYB") == (4, 6)
    assert linkedPositionsFromGlycamResname("VMA") == (3, 6)
    assert linkedPositionsFromGlycamResname("0CU") == ()
    assert linkedPositionsFromGlycamResname("1CU") == (1,)
    assert linkedPositionsFromGlycamResname("VCU") == (3, 6)
    # 184 -> 204: adding fructose's 20 furanose units (10 linkage positions,
    # anomers D and U each) grew the table; 204 -> 258: adding the three
    # uronic acid letters (Z, O, u), each with 9 linkage positions and
    # anomers A and B, added 54 more; see the GLYCAM_UNIT_NAMES regeneration
    # recipe in moleculekit/tools/glycans.py.
    assert "UYB" in GLYCAM_UNIT_NAMES and len(GLYCAM_UNIT_NAMES) == 258


def test_sugar_table_consistency():
    from moleculekit.tools.glycans import GLYCAM_SUGARS

    for resname, tmpl in GLYCAM_SUGARS.items():
        # A/B (alpha/beta) for every sugar except fructose's furanose ring,
        # which GLYCAM names D/U instead.
        assert tmpl.anomer in ("A", "B", "D", "U"), resname
        assert tmpl.anomeric_carbon in ("C1", "C2"), resname
        assert tmpl.ring_oxygen in ("O5", "O6"), resname


import os
import numpy as np
from moleculekit.molecule import Molecule

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
GLYCAN_DIR = os.path.join(CURR_DIR, "test_glycans")


def _residue_groups(mol):
    from moleculekit.util import sequenceID

    uq = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))
    return [np.where(uq == u)[0] for u in np.unique(uq)]


def _single_residue_mol(resname, anomeric_carbon, ring_oxygen):
    """Minimal 2-atom molecule for one GLYCAM unit residue, holding just its
    anomeric carbon and ring oxygen 1.4 A apart (a real ring bond distance),
    the only atoms glycamUnitMask's composition gate inspects.
    """
    mol = Molecule().empty(2)
    mol.name[:] = [anomeric_carbon, ring_oxygen]
    mol.element[:] = ["C", "O"]
    mol.resname[:] = resname
    mol.resid[:] = 1
    mol.chain[:] = "X"
    mol.segid[:] = "X"
    mol.insertion[:] = ""
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    mol.coords = coords.reshape(2, 3, 1).astype(Molecule._dtypes["coords"])
    return mol


def test_glycam_unit_mask_table_driven_ring_atoms():
    """Composition-gate regression test for the anomeric-carbon /
    ring-oxygen refactor: glycamUnitMask must recognize a fructose residue
    via C2 + O5, while still recognizing sialic acid via C2 + O6 and an
    ordinary sugar via C1 + O5.
    """
    from moleculekit.tools.glycans import glycamUnitMask

    # fructose: C2 + O5 (the combination that breaks the old binary, since
    # it is neither the C1+O5 nor the C2+O6 branch it hardcoded)
    mol = _single_residue_mol("0CU", "C2", "O5")
    assert glycamUnitMask(mol).all()

    # regression: sialic acid via C2 + O6
    mol = _single_residue_mol("0SA", "C2", "O6")
    assert glycamUnitMask(mol).all()

    # regression: an ordinary sugar via C1 + O5
    mol = _single_residue_mol("0YB", "C1", "O5")
    assert glycamUnitMask(mol).all()


def test_analyze_3ave_branched_nglycan():
    from moleculekit.tools.glycans import analyzeGlycanResidues, glycamResname

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    groups = _residue_groups(mol)
    info = analyzeGlycanResidues(mol, mol.bonds, groups)
    assert len(info) == 8
    codes = sorted(
        glycamResname(str(mol.resname[groups[i][0]]), gi.linked_positions)
        for i, gi in info.items()
    )
    assert codes == sorted(["UYB", "4YB", "VMB", "2MA", "0YB", "2MA", "0YB", "0fA"])
    anchored = [gi for gi in info.values() if gi.anchor_res is not None]
    assert len(anchored) == 1
    assert anchored[0].anchor_atom == "ND2"
    assert not any(gi.free_reducing_end for gi in info.values())


def test_analyze_1cvn_free_reducing_end():
    from moleculekit.tools.glycans import analyzeGlycanResidues, glycamResname

    mol = Molecule(os.path.join(GLYCAN_DIR, "1CVN_frag.pdb"))
    groups = _residue_groups(mol)
    info = analyzeGlycanResidues(mol, mol.bonds, groups)
    assert len(info) == 3
    codes = sorted(
        glycamResname(str(mol.resname[groups[i][0]]), gi.linked_positions)
        for i, gi in info.items()
    )
    assert codes == ["0MA", "0MA", "VMA"]
    free = [gi for gi in info.values() if gi.free_reducing_end]
    assert len(free) == 1 and free[0].linked_positions == (3, 6)


def test_analyze_1g1s_olinked():
    from moleculekit.tools.glycans import analyzeGlycanResidues, glycamResname

    mol = Molecule(os.path.join(GLYCAN_DIR, "1G1S_frag.pdb"))
    groups = _residue_groups(mol)
    info = analyzeGlycanResidues(mol, mol.bonds, groups)
    codes = sorted(
        glycamResname(str(mol.resname[groups[i][0]]), gi.linked_positions)
        for i, gi in info.items()
    )
    assert codes == sorted(["VVB", "WYB", "3LB", "0SA", "0fA", "0LB"])
    anchored = [gi for gi in info.values() if gi.anchor_res is not None]
    assert len(anchored) == 1 and anchored[0].anchor_atom == "OG1"


def test_analyze_unsupported_anchor():
    import pytest
    from moleculekit.tools.glycans import analyzeGlycanResidues

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    # fake a TRP C-mannosylation: relabel the anchor ASN as TRP
    asn = mol.resname == "ASN"
    nd2 = asn & (mol.name == "ND2")
    mol.resname[asn] = "TRP"
    mol.name[nd2] = "CD1"
    groups = _residue_groups(mol)
    with pytest.raises(RuntimeError, match="TRP"):
        analyzeGlycanResidues(mol, mol.bonds, groups)


def test_analyze_unmapped_child_sugar():
    import pytest
    from moleculekit.tools.glycans import analyzeGlycanResidues

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    # NAG5 is linked onto MAN4's O2 (a child reached via a ring oxygen, not
    # an anomeric carbon). Relabel it to a carbohydrate GLYCAM_SUGARS does
    # not cover: a naive implementation could silently drop this branch
    # instead of raising, since only the anomeric-carbon path was an
    # obvious place to check for an unmapped sugar.
    nag5 = (mol.resname == "NAG") & (mol.chain == "C") & (mol.resid == 5)
    mol.resname[nag5] = "XXX"
    groups = _residue_groups(mol)
    with pytest.raises(RuntimeError, match="XXX"):
        analyzeGlycanResidues(mol, mol.bonds, groups)


def test_analyze_unmapped_parent_sugar():
    import pytest
    from moleculekit.tools.glycans import analyzeGlycanResidues

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    # NAG1 is the tree root: nothing bonds into any of its own ring oxygens,
    # so relabeling it cannot trip the ring-oxygen (child) check above.
    # Its children (NAG2 via O4, FUC8 via O6) still see NAG1 as the
    # unresolved partner of their own anomeric carbon, which must be
    # reported as an unmapped sugar, not as an unsupported anchor (NAG1 is
    # not a protein residue at all).
    nag1 = (mol.resname == "NAG") & (mol.chain == "C") & (mol.resid == 1)
    mol.resname[nag1] = "XXX"
    groups = _residue_groups(mol)
    with pytest.raises(RuntimeError) as excinfo:
        analyzeGlycanResidues(mol, mol.bonds, groups)
    message = str(excinfo.value)
    assert "XXX" in message
    assert "supported GLYCAM sugar table" in message
    # must not be misreported as the (unrelated) unsupported-anchor path
    assert "does not support glycosylation of" not in message


def test_analyze_multiple_anomeric_partners():
    import pytest
    from moleculekit.tools.glycans import analyzeGlycanResidues

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    groups = _residue_groups(mol)
    nag1_c1 = int(
        np.where(
            (mol.resname == "NAG")
            & (mol.chain == "C")
            & (mol.resid == 1)
            & (mol.name == "C1")
        )[0][0]
    )
    fuc8_o2 = int(
        np.where(
            (mol.resname == "FUC")
            & (mol.chain == "C")
            & (mol.resid == 8)
            & (mol.name == "O2")
        )[0][0]
    )
    # Build a synthetic bonds array (mol.bonds is untouched) that gives
    # NAG1's anomeric carbon a second partner in addition to ASN's ND2.
    bonds = np.vstack([mol.bonds, [nag1_c1, fuc8_o2]])
    with pytest.raises(RuntimeError, match="at most one"):
        analyzeGlycanResidues(mol, bonds, groups)


def test_glycan_bonds_from_names_3ave():
    from moleculekit.tools.glycans import glycanBondsFromNames

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    # simulate the prepared molecule: GLYCAM resnames, no bonds
    ren = {
        1: "UYB", 2: "4YB", 3: "VMB", 4: "2MA", 5: "0YB", 6: "2MA",
        7: "0YB", 8: "0fA",
    }
    for resid, code in ren.items():
        mol.resname[(mol.chain == "C") & (mol.resid == resid)] = code
    asn = (mol.chain == "A") & (mol.resid == 297)
    mol.resname[asn] = "NLN"
    mol.deleteBonds("all")
    pairs = glycanBondsFromNames(mol)
    named = set()
    for i, j in pairs:
        named.add(tuple(sorted((str(mol.name[i]), str(mol.name[j])))))
    # every derived bond is oxygen/nitrogen to anomeric carbon
    assert len(pairs) == 8  # 7 glycosidic + 1 ND2 anchor
    assert ("C1", "ND2") in named
    assert ("C1", "O4") in named and ("C1", "O6") in named


def test_detect_emits_glycanspec_3ave():
    from moleculekit.tools.nonstandard_residues import (
        GlycanSpec,
        detectNonStandardResidues,
    )

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    specs = detectNonStandardResidues(mol)
    gly = [s for s in specs if isinstance(s, GlycanSpec)]
    assert len(gly) == 8
    assert sorted(s.new_resname for s in gly) == sorted(
        ["UYB", "4YB", "VMB", "2MA", "0YB", "2MA", "0YB", "0fA"]
    )
    stem = [s for s in gly if s.anchor_residue is not None]
    assert len(stem) == 1
    assert stem[0].anchor_new_resname == "NLN" and stem[0].anchor_atom == "ND2"
    assert stem[0].atom_renames == {"C7": "C2N", "O7": "O2N", "C8": "CME"}
    # no other spec types remain: no XX bucket for the ASN, no ligand specs
    assert len(gly) == len(specs)


def test_detect_1r1j_glycanspec():
    # 1R1J: three single-NAG sites. A NAG whose C1 bonds an ASN and carries
    # no other links is 0YB with an NLN anchor. The three ASNs emit no spec
    # of their own (the rename travels on the sugar's anchor fields).
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec,
        GlycanSpec,
        detectNonStandardResidues,
    )

    mol = Molecule(os.path.join(CURR_DIR, "pdb", "1r1j.pdb"))
    specs = detectNonStandardResidues(mol)
    gly = [s for s in specs if isinstance(s, GlycanSpec)]
    assert len(gly) == 3
    assert all(s.new_resname == "0YB" for s in gly)
    assert all(s.anchor_new_resname == "NLN" for s in gly)
    asn_specs = [
        s for s in specs if isinstance(s, ChainResidueSpec) and s.resname == "ASN"
    ]
    assert len(asn_specs) == 0


def test_systemprepare_3ave_glycam_renames():
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(os.path.join(GLYCAN_DIR, "3AVE_frag.pdb"))
    pmol, _ = systemPrepare(mol)
    # chain/segid may be reassigned by preparation; assert on resnames globally
    for code in ("UYB", "4YB", "VMB", "2MA", "0YB", "0fA", "NLN"):
        assert code in pmol.resname, code
    assert "NAG" not in pmol.resname and "BMA" not in pmol.resname
    # NLN keeps exactly one amide hydrogen, named HD21
    nln = pmol.resname == "NLN"
    nd2h = nln & np.isin(pmol.name, ["HD21", "HD22"])
    assert np.sum(nd2h) == 1 and pmol.name[nd2h][0] == "HD21"
    # acetyl heavy atoms renamed
    uyb = pmol.resname == "UYB"
    assert np.sum(uyb & (pmol.name == "C2N")) == 1
    assert np.sum(uyb & (pmol.name == "C7")) == 0


def test_systemprepare_1cvn_roh_split():
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(os.path.join(GLYCAN_DIR, "1CVN_frag.pdb"))
    pmol, _ = systemPrepare(mol)
    assert "ROH" in pmol.resname and "VMA" in pmol.resname
    roh = pmol.resname == "ROH"
    # ROH is its own residue holding the anomeric hydroxyl oxygen
    assert set(pmol.name[roh & (pmol.element != "H")]) == {"O1"}
    vma = pmol.resname == "VMA"
    assert np.sum(vma & (pmol.name == "O1")) == 0
    # ROH shares the sugar's segment and has a unique resid there
    seg = pmol.segid[roh][0]
    roh_resid = pmol.resid[roh][0]
    others = (pmol.segid == seg) & ~roh
    assert roh_resid not in pmol.resid[others]


def test_systemprepare_1g1s_olinked():
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(os.path.join(GLYCAN_DIR, "1G1S_frag.pdb"))
    pmol, _ = systemPrepare(mol)
    for code in ("OLT", "VVB", "WYB", "3LB", "0SA", "0fA", "0LB"):
        assert code in pmol.resname, code
    olt = pmol.resname == "OLT"
    assert np.sum(olt & (pmol.name == "HG1")) == 0
    assert np.sum(olt & (pmol.name == "OG1")) == 1
