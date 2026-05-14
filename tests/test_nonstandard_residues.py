import os
import numpy as np
from moleculekit.molecule import Molecule, UniqueResidueID
from moleculekit.tools._anchor_variants import lookup_anchor_variant
from moleculekit.residues import ORIGINAL_RESIDUE_NAME_TABLE
from moleculekit.tools.nonstandard_residues import (
    detectNonStandardResidues,
    NCAASpec,
    CrosslinkedNCAASpec,
    ScaffoldSpec,
    CovalentLigandSpec,
    LigandSpec,
    CanonicalRenamedSpec,
    _disambiguate_terminus_resnames,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(curr_dir, "test_nonstandard_residues")
QFZ_B_CIF = os.path.join(DATA_DIR, "8QFZ_B.cif")
QU4_A_CIF = os.path.join(DATA_DIR, "8QU4_A.cif")
VBL_PDB = os.path.join(curr_dir, "pdb", "5vbl.pdb")
R1J_PDB = os.path.join(curr_dir, "pdb", "1r1j.pdb")


def _residue_atom_names(mol, resname, resid):
    """Helper: return the set of atom names on a (resname, resid) residue."""
    sel = mol.atomselect(f"resname {resname} and resid {resid}", indexes=True)
    return {str(mol.name[int(i)]) for i in sel}


def _drop_crosslinks(mol, resname, n):
    """Drop ``n`` of the inter-residue bonds connecting the named residue to
    the rest of mol. Returns the modified mol (mutated in place)."""
    target = mol.atomselect(f"resname {resname}", indexes=True)
    keep_idx = []
    dropped = 0
    for i, b in enumerate(mol.bonds):
        is_cross = (b[0] in target) != (b[1] in target)
        if is_cross and dropped < n:
            dropped += 1
            continue
        keep_idx.append(i)
    mol.bonds = mol.bonds[keep_idx]
    mol.bondtype = mol.bondtype[keep_idx]
    return mol


def _make_vbl_with_z_alc_nterminal():
    """Duplicate 5VBL chain A into a new chain Z (offset 100 Å so the
    distance-based peptide-bond fallback doesn't see chain A's C atoms
    as neighbours of chain Z's N atoms) and drop chain-Z residues N-side
    of the duplicated ALC so chain Z's ALC ends up N-terminal."""
    mol = Molecule(VBL_PDB)
    a_resid = int(np.unique(mol.resid[mol.resname == "ALC"])[0])
    dup = mol.copy()
    dup.filter(np.where(dup.chain == "A")[0], _logger=False)
    dup.chain[:] = "Z"
    dup.segid[:] = "Z"
    dup.resid[:] = dup.resid + 1000
    dup.coords += 100.0
    mol.append(dup, collisions=False)
    z_alc_resid = a_resid + 1000
    drop_mask = (mol.chain == "Z") & (mol.resid < z_alc_resid)
    mol.remove(np.where(drop_mask)[0], _logger=False)
    return mol


def _test_anchor_variants_lookup():
    e = lookup_anchor_variant("CYS", "SG")
    assert e is not None
    assert e["variant"] == "CYX"
    assert e["drop_h"] == ["HG"]
    assert e["ff14sb_type"] == "S"

    # Variant resname (CYX) routes via ORIGINAL_RESIDUE_NAME_TABLE back to CYS
    # and yields the same entry.
    assert lookup_anchor_variant("CYX", "SG") == e
    assert ORIGINAL_RESIDUE_NAME_TABLE["CYX"] == "CYS"

    # Unknown anchor: missing entry, not an error.
    assert lookup_anchor_variant("ALA", "CA") is None
    assert lookup_anchor_variant("UNK", "X1") is None


def _test_8qfz_scaffolded_peptide():
    """8QFZ chain B: LFI scaffold thio-ether bonded to three CYS sidechains
    at resids 11, 17, 22. CYS 11 is N-terminal, CYS 22 is C-terminal,
    CYS 17 is mid-chain. Each chain-position bucket gets its own custom
    resname so terminal forms (which carry OXT or H1/H2/H3 in solution)
    don't collapse onto the mid-chain template."""
    mol = Molecule(QFZ_B_CIF)
    n_atoms_before = mol.numAtoms
    resnames_before = sorted(set(str(r) for r in mol.resname))

    specs = detectNonStandardResidues(mol)

    # Detect is non-mutating.
    assert mol.numAtoms == n_atoms_before
    assert sorted(set(str(r) for r in mol.resname)) == resnames_before

    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]

    assert len(scaffolds) == 1 and scaffolds[0].resname == "LFI"
    assert len(renames) == 3
    assert {r.residue.resname for r in renames} == {"CYS"}
    new_names = {r.new_resname for r in renames}
    # Three distinct chain positions -> three distinct rename targets.
    assert len(new_names) == 3, f"expected three distinct renames, got {new_names}"
    for n in new_names:
        assert len(n) == 3 and n.startswith("CY")
    for r in renames:
        assert r.drop_h == ["HG"]

    assert all(isinstance(s, (ScaffoldSpec, CanonicalRenamedSpec)) for s in specs)


def _test_single_anchor_demotes_scaffold_to_covalent_ligand():
    """Strip two of the three CYS-LFI bonds: LFI now has only one anchor
    so it's a CovalentLigandSpec rather than a ScaffoldSpec, and only the
    one remaining CYS gets renamed."""
    mol = _drop_crosslinks(Molecule(QFZ_B_CIF), "LFI", 2)

    specs = detectNonStandardResidues(mol)
    cov = [s for s in specs if isinstance(s, CovalentLigandSpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]

    assert scaffolds == []
    assert len(cov) == 1 and cov[0].resname == "LFI"
    assert len(renames) == 1 and renames[0].residue.resname == "CYS"


def _test_5vbl_ncaas_and_free_ligand():
    """5VBL chain A peptide inhibitor: five chain-resident NCAAs with no
    crosslinks, plus a free OLC ligand. Detector emits NCAASpec entries
    and one LigandSpec."""
    mol = Molecule(VBL_PDB)
    specs = detectNonStandardResidues(mol)

    ncaas = [s for s in specs if isinstance(s, NCAASpec)]
    crosslinked = [s for s in specs if isinstance(s, CrosslinkedNCAASpec)]
    ligands = [s for s in specs if isinstance(s, LigandSpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]
    cov = [s for s in specs if isinstance(s, CovalentLigandSpec)]

    assert crosslinked == [] and renames == [] and scaffolds == [] and cov == []
    assert sorted(s.resname for s in ncaas) == ["200", "ALC", "HRG", "NLE", "OIC"]
    # Resid 17 = "200" is the C-terminal NCAA: no peptide tail bond.
    for s in ncaas:
        if s.resname == "200":
            assert s.is_c_term and not s.is_n_term
        else:
            assert not s.is_n_term and not s.is_c_term

    assert len(ligands) == 1
    assert ligands[0].resname == "OLC" and ligands[0].residue.chain == "B"


def _test_1r1j_covalent_glycosylation():
    """1R1J: three NAG-Asn N-glycosylation sites. Each NAG has one bond to
    an Asn ND2; all three (ASN, ND2, NAG) buckets share the same chain
    position (mid-chain) so the detector emits one CovalentLigandSpec
    per NAG plus three CanonicalRenamedSpec entries that share the same
    new resname.

    The OIR inhibitor in 1R1J is a thiorphan-class non-covalent Zn-chelator:
    its O19 and S26 contact the active-site Zn (PDB ``LINK`` records, loaded
    as bonds) but it has no covalent bond to the protein. Metal-coordination
    contacts must not be counted as covalent crosslinks, so OIR is a
    LigandSpec rather than a ScaffoldSpec."""
    mol = Molecule(R1J_PDB)
    specs = detectNonStandardResidues(mol)

    cov = [s for s in specs if isinstance(s, CovalentLigandSpec) and s.resname == "NAG"]
    asn_renames = [
        s
        for s in specs
        if isinstance(s, CanonicalRenamedSpec) and s.residue.resname == "ASN"
    ]
    assert len(cov) == 3
    assert len(asn_renames) == 3
    new_names = {r.new_resname for r in asn_renames}
    assert len(new_names) == 1, f"expected one shared rename, got {new_names}"
    shared = next(iter(new_names))
    assert len(shared) == 3 and shared.startswith("NL")
    for r in asn_renames:
        rid = int(r.residue.resid)
        names = _residue_atom_names(mol, r.residue.resname, rid)
        assert "ND2" in names
        # The displaced HD22 hydrogen is reported via drop_h, not removed.
        assert r.drop_h == ["HD22"]

    oir = [s for s in specs if s.residue.resname == "OIR"]
    assert len(oir) == 1
    assert isinstance(oir[0], LigandSpec), (
        f"OIR coordinates Zn via O19/S26 but has no covalent partner; "
        f"expected LigandSpec, got {type(oir[0]).__name__}"
    )


def _test_8qu4_ncaa_crosslink():
    """8QU4 chain A stapled peptide: NLE272 + MK8276 are two NCAAs joined
    by a sidechain CE-CE staple. Detector emits a CrosslinkedNCAASpec for
    each; no CanonicalRenamedSpec because neither residue is canonical."""
    mol = Molecule(QU4_A_CIF)
    specs = detectNonStandardResidues(mol)

    crosslinked = [s for s in specs if isinstance(s, CrosslinkedNCAASpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]

    assert renames == [] and scaffolds == []
    assert sorted(s.resname for s in crosslinked) == ["MK8", "NLE"]
    # Both residues are mid-chain (not termini).
    for s in crosslinked:
        assert not s.is_n_term and not s.is_c_term


def _test_scaffold_anchored_on_ncaa_sidechains():
    """A scaffold whose anchors land on NCAA sidechains (rather than
    canonical Cys/Lys/etc.) still emits ScaffoldSpec for the scaffold and
    CrosslinkedNCAASpec for each chain-resident NCAA. No
    CanonicalRenamedSpec because there are no canonical anchors."""
    mol = Molecule(QU4_A_CIF)
    mol.segid[:] = "P"

    nle_ce = int(mol.atomselect("resname NLE and name CE", indexes=True)[0])
    mk8_ce = int(mol.atomselect("resname MK8 and name CE", indexes=True)[0])

    # Replace the existing NLE.CE-MK8.CE staple with two anchor bonds to a
    # synthetic 2-carbon SCF residue.
    keep_idx = [
        i for i, b in enumerate(mol.bonds)
        if {int(b[0]), int(b[1])} != {nle_ce, mk8_ce}
    ]
    mol.bonds = mol.bonds[keep_idx]
    mol.bondtype = mol.bondtype[keep_idx]

    scf = Molecule().empty(2)
    scf.name[:] = ["C1", "C2"]
    scf.element[:] = ["C", "C"]
    scf.resname[:] = "SCF"
    scf.resid[:] = 999
    scf.chain[:] = "A"
    scf.segid[:] = "P"
    scf.record[:] = "HETATM"
    scf.coords = (
        np.tile(mol.coords[nle_ce, :, mol.frame], (2, 1))
        + np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    )[:, :, np.newaxis].astype(np.float32)
    n_before = mol.numAtoms
    mol.append(scf, collisions=False)
    c1, c2 = n_before, n_before + 1
    mol.addBond(c1, c2, "1")
    mol.addBond(nle_ce, c1, "1")
    mol.addBond(mk8_ce, c2, "1")

    specs = detectNonStandardResidues(mol)
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]
    crosslinked = [s for s in specs if isinstance(s, CrosslinkedNCAASpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]

    assert len(scaffolds) == 1 and scaffolds[0].resname == "SCF"
    assert sorted(s.resname for s in crosslinked) == ["MK8", "NLE"]
    assert renames == []


def _test_canonical_to_ncaa_crosslink():
    """A non-peptide bond between a canonical AA and an NCAA emits a
    CrosslinkedNCAASpec for the NCAA plus a CanonicalRenamedSpec for the
    canonical residue (a single-anchor scenario, so no ScaffoldSpec)."""
    mol = Molecule(QU4_A_CIF)
    mol.segid[:] = "P"
    # Rename MK8 -> ALA so the existing CE-CE staple becomes asymmetric.
    mol.resname[mol.resname == "MK8"] = "ALA"

    specs = detectNonStandardResidues(mol)
    crosslinked = [s for s in specs if isinstance(s, CrosslinkedNCAASpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]

    assert len(crosslinked) == 1 and crosslinked[0].resname == "NLE"
    assert len(renames) == 1
    # ALA has no entry in the anchor-variants table so the new resname
    # falls back to the original base with a numeric suffix.
    assert renames[0].residue.resname == "ALA"
    assert renames[0].new_resname.startswith("AL")


def _test_distinct_partner_resnames_get_distinct_renames():
    """Two CYS residues, each bonded to a *different* non-canonical
    residue (different ``partner_resname``), produce two distinct
    bucket keys and two distinct rename targets. CYS residues bonded to
    the *same* partner resname collapse into one shared rename."""
    base = _drop_crosslinks(Molecule(QFZ_B_CIF), "LFI", 2)

    a = base.copy()
    a.chain[:] = "X"
    a.segid[:] = "X"
    b = base.copy()
    b.chain[:] = "Y"
    b.segid[:] = "Y"
    b.resid[:] = b.resid + 100
    # Rename the second copy's scaffold so it has a *different* partner
    # resname (LFI vs SCF), forcing a distinct bucket.
    b.resname[b.resname == "LFI"] = "SCF"
    a.append(b, collisions=False)

    specs = detectNonStandardResidues(a)
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]
    assert len(renames) == 2
    new_names = sorted({r.new_resname for r in renames})
    assert len(new_names) == 2, f"expected two buckets, got {new_names}"
    for n in new_names:
        assert len(n) == 3 and n.startswith("CY")


def _test_ncaa_spec_default_new_resname():
    """new_resname defaults to None on both NCAA spec types and accepts a string when given."""
    rid = UniqueResidueID(
        resname="ALC", chain="A", resid=1, insertion="", segid="A"
    )
    s1 = NCAASpec(resname="ALC", residue=rid, is_n_term=False, is_c_term=False)
    assert s1.new_resname is None
    s2 = NCAASpec(
        resname="ALC",
        residue=rid,
        is_n_term=True,
        is_c_term=False,
        new_resname="NALC",
    )
    assert s2.new_resname == "NALC"

    s3 = CrosslinkedNCAASpec(
        resname="ALC", residue=rid, is_n_term=False, is_c_term=False
    )
    assert s3.new_resname is None
    s4 = CrosslinkedNCAASpec(
        resname="ALC",
        residue=rid,
        is_n_term=False,
        is_c_term=True,
        new_resname="CALC",
    )
    assert s4.new_resname == "CALC"


def _test_disambiguate_single_config_no_rename():
    """One NCAA resname, all instances mid-chain -> no rename."""
    rid_a = UniqueResidueID(
        resname="ALC", chain="A", resid=1, insertion="", segid="A"
    )
    rid_b = UniqueResidueID(
        resname="ALC", chain="A", resid=2, insertion="", segid="A"
    )
    specs = [
        NCAASpec(resname="ALC", residue=rid_a, is_n_term=False, is_c_term=False),
        NCAASpec(resname="ALC", residue=rid_b, is_n_term=False, is_c_term=False),
    ]
    _disambiguate_terminus_resnames(specs)
    assert all(s.new_resname is None for s in specs)


def _test_disambiguate_two_configs_renames_nonmid():
    """Same NCAA resname appearing mid-chain AND N-term: mid keeps the
    original (new_resname=None), N-term gets NALC."""
    rid_mid = UniqueResidueID(
        resname="ALC", chain="A", resid=2, insertion="", segid="A"
    )
    rid_n = UniqueResidueID(
        resname="ALC", chain="A", resid=1, insertion="", segid="A"
    )
    s_mid = NCAASpec(
        resname="ALC", residue=rid_mid, is_n_term=False, is_c_term=False
    )
    s_n = NCAASpec(
        resname="ALC", residue=rid_n, is_n_term=True, is_c_term=False
    )
    _disambiguate_terminus_resnames([s_mid, s_n])
    assert s_mid.new_resname is None
    assert s_n.new_resname == "NALC"


def _test_disambiguate_three_configs():
    """Mid + N + C-term all present -> mid stays, N -> NALC, C -> CALC."""
    def _spec(i, n, c):
        return NCAASpec(
            resname="ALC",
            residue=UniqueResidueID(
                resname="ALC", chain="A", resid=i, insertion="", segid="A"
            ),
            is_n_term=n,
            is_c_term=c,
        )

    s_mid = _spec(2, False, False)
    s_n = _spec(1, True, False)
    s_c = _spec(3, False, True)
    _disambiguate_terminus_resnames([s_mid, s_n, s_c])
    assert s_mid.new_resname is None
    assert s_n.new_resname == "NALC"
    assert s_c.new_resname == "CALC"


def _test_disambiguate_both_terminus_uses_b_prefix():
    """A single-residue chain (both is_n_term and is_c_term True) coexisting
    with a mid-chain instance of the same resname uses the B prefix."""
    rid_mid = UniqueResidueID(
        resname="ALC", chain="A", resid=2, insertion="", segid="A"
    )
    rid_both = UniqueResidueID(
        resname="ALC", chain="B", resid=1, insertion="", segid="B"
    )
    s_mid = NCAASpec(
        resname="ALC", residue=rid_mid, is_n_term=False, is_c_term=False
    )
    s_both = NCAASpec(
        resname="ALC", residue=rid_both, is_n_term=True, is_c_term=True
    )
    _disambiguate_terminus_resnames([s_mid, s_both])
    assert s_mid.new_resname is None
    assert s_both.new_resname == "BALC"


def _test_disambiguate_applies_to_crosslinked_ncaa():
    """The disambiguator treats CrosslinkedNCAASpec the same as NCAASpec."""
    rid_mid = UniqueResidueID(
        resname="MK8", chain="A", resid=2, insertion="", segid="A"
    )
    rid_n = UniqueResidueID(
        resname="MK8", chain="A", resid=1, insertion="", segid="A"
    )
    s_mid = CrosslinkedNCAASpec(
        resname="MK8", residue=rid_mid, is_n_term=False, is_c_term=False
    )
    s_n = CrosslinkedNCAASpec(
        resname="MK8", residue=rid_n, is_n_term=True, is_c_term=False
    )
    _disambiguate_terminus_resnames([s_mid, s_n])
    assert s_mid.new_resname is None
    assert s_n.new_resname == "NMK8"


def _test_disambiguate_independent_groups():
    """Two different NCAA resnames, each with a single config, both
    stay un-renamed. Disambiguation is per-resname, not global."""
    s_alc = NCAASpec(
        resname="ALC",
        residue=UniqueResidueID(
            resname="ALC", chain="A", resid=1, insertion="", segid="A"
        ),
        is_n_term=False,
        is_c_term=False,
    )
    s_hrg = NCAASpec(
        resname="HRG",
        residue=UniqueResidueID(
            resname="HRG", chain="A", resid=2, insertion="", segid="A"
        ),
        is_n_term=True,
        is_c_term=False,
    )
    _disambiguate_terminus_resnames([s_alc, s_hrg])
    assert s_alc.new_resname is None
    assert s_hrg.new_resname is None


def _test_disambiguate_4char_input_raises():
    """If disambiguation is required AND the input resname is 4+ chars,
    the prefixed name would exceed the 4-char AMBER prepi unit-name limit."""
    import pytest as _pytest

    s_mid = NCAASpec(
        resname="ABCD",
        residue=UniqueResidueID(
            resname="ABCD", chain="A", resid=1, insertion="", segid="A"
        ),
        is_n_term=False,
        is_c_term=False,
    )
    s_n = NCAASpec(
        resname="ABCD",
        residue=UniqueResidueID(
            resname="ABCD", chain="A", resid=2, insertion="", segid="A"
        ),
        is_n_term=True,
        is_c_term=False,
    )
    with _pytest.raises(RuntimeError, match="ABCD"):
        _disambiguate_terminus_resnames([s_mid, s_n])


def _test_disambiguate_ignores_other_spec_types():
    """The disambiguator only touches NCAASpec / CrosslinkedNCAASpec.
    Other specs (LigandSpec, ScaffoldSpec, etc.) are left untouched."""
    rid = UniqueResidueID(
        resname="OLC", chain="B", resid=1, insertion="", segid="B"
    )
    other = LigandSpec(resname="OLC", residue=rid)
    s_mid = NCAASpec(
        resname="ALC",
        residue=UniqueResidueID(
            resname="ALC", chain="A", resid=1, insertion="", segid="A"
        ),
        is_n_term=False,
        is_c_term=False,
    )
    _disambiguate_terminus_resnames([other, s_mid])
    assert other.resname == "OLC"
    assert s_mid.new_resname is None


def _test_detect_no_rename_when_all_one_config():
    """5VBL has 200/ALC/HRG/NLE/OIC each appearing exactly once and the
    one C-terminal residue is 200 (no other resname appears at the C
    terminus). After Task 3 the detector still emits new_resname=None
    for all of them - no disambiguation needed."""
    mol = Molecule(VBL_PDB)
    specs = detectNonStandardResidues(mol)
    ncaas = [s for s in specs if isinstance(s, NCAASpec)]
    assert len(ncaas) >= 5
    for s in ncaas:
        assert s.new_resname is None, (
            f"{s.resname} unexpectedly got new_resname={s.new_resname!r}"
        )


def _test_detect_renames_when_resname_shared_across_configs():
    """Take 5VBL, duplicate its peptide chain into chain Z with a resid
    offset, then drop the chain-Z residues that sit N-side of the
    duplicated ALC so chain Z's ALC becomes N-terminal. Now ALC appears
    both mid-chain (chain A) and N-terminal (chain Z): the detector
    emits new_resname=NALC for the N-term spec and None for the
    mid-chain spec."""
    mol = _make_vbl_with_z_alc_nterminal()

    specs = detectNonStandardResidues(mol)
    alc_specs = [
        s
        for s in specs
        if isinstance(s, NCAASpec) and s.residue.resname == "ALC"
    ]
    assert len(alc_specs) == 2, (
        f"expected 2 ALC specs (chain A mid + chain Z N-term), got "
        f"{len(alc_specs)}"
    )

    by_chain = {str(s.residue.chain): s for s in alc_specs}
    assert by_chain["A"].is_n_term is False and by_chain["A"].is_c_term is False
    assert by_chain["A"].new_resname is None
    assert by_chain["Z"].is_n_term is True
    assert by_chain["Z"].new_resname == "NALC"


def _test_detect_unaffected_groups_left_alone():
    """When ALC needs disambiguation but HRG does not, only ALC specs
    get new_resname set."""
    mol = _make_vbl_with_z_alc_nterminal()

    specs = detectNonStandardResidues(mol)
    hrg_specs = [
        s
        for s in specs
        if isinstance(s, NCAASpec) and s.residue.resname == "HRG"
    ]
    for s in hrg_specs:
        assert s.new_resname is None, (
            f"HRG spec on chain {s.residue.chain} unexpectedly renamed to "
            f"{s.new_resname!r}; only ALC should be disambiguated."
        )


def _test_apply_detect_spec_renames_for_ncaa():
    """The rename helper sets mol.resname to spec.new_resname when set
    on an NCAASpec; other residues with the same input resname are
    untouched."""
    from moleculekit.tools.preparation import _apply_detect_spec_renames

    mol = Molecule(VBL_PDB)
    alc_resids = np.unique(mol.resid[mol.resname == "ALC"])
    target_resid = int(alc_resids[0])
    alc_mask = (mol.resname == "ALC") & (mol.resid == target_resid)
    target_chain = str(mol.chain[np.where(alc_mask)[0][0]])
    target_segid = str(mol.segid[np.where(alc_mask)[0][0]])
    target_insertion = str(mol.insertion[np.where(alc_mask)[0][0]])

    spec = NCAASpec(
        resname="ALC",
        residue=UniqueResidueID(
            resname="ALC",
            chain=target_chain,
            resid=target_resid,
            insertion=target_insertion,
            segid=target_segid,
        ),
        is_n_term=True,
        is_c_term=False,
        new_resname="NALC",
    )

    _apply_detect_spec_renames(mol, [spec])

    target_mask = (
        (mol.segid == target_segid)
        & (mol.chain == target_chain)
        & (mol.resid == target_resid)
        & (mol.insertion == target_insertion)
    )
    assert (mol.resname[target_mask] == "NALC").all()
    other_alc = (mol.resname == "ALC") & ~target_mask
    if other_alc.any():
        assert (mol.resname[other_alc] == "ALC").all()


def _test_apply_detect_spec_renames_skips_none():
    """new_resname=None leaves mol.resname untouched."""
    from moleculekit.tools.preparation import _apply_detect_spec_renames

    mol = Molecule(VBL_PDB)
    before = mol.resname.copy()

    alc_resid = int(np.unique(mol.resid[mol.resname == "ALC"])[0])
    spec = NCAASpec(
        resname="ALC",
        residue=UniqueResidueID(
            resname="ALC", chain="A", resid=alc_resid, insertion="", segid="A"
        ),
        is_n_term=False,
        is_c_term=False,
        new_resname=None,
    )
    _apply_detect_spec_renames(mol, [spec])
    assert (mol.resname == before).all()


def _test_apply_detect_spec_renames_canonical_drops_h():
    """For CanonicalRenamedSpec the helper applies the rename AND drops
    listed sidechain hydrogens, matching the existing systemPrepare
    behavior."""
    from moleculekit.tools.preparation import _apply_detect_spec_renames

    mol = Molecule().empty(4)
    mol.name[:] = ["CA", "CB", "SG", "HG"]
    mol.element[:] = ["C", "C", "S", "H"]
    mol.resname[:] = "CYS"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.insertion[:] = ""
    mol.coords = np.zeros((4, 3, 1), dtype=np.float32)

    spec = CanonicalRenamedSpec(
        residue=UniqueResidueID(
            resname="CYS", chain="A", resid=1, insertion="", segid="A"
        ),
        new_resname="CY1",
        drop_h=["HG"],
    )
    _apply_detect_spec_renames(mol, [spec])
    assert (mol.resname == "CY1").all()
    assert "HG" not in set(mol.name)
    assert mol.numAtoms == 3
