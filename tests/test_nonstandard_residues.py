import os
import numpy as np
from moleculekit.molecule import Molecule, UniqueResidueID
from moleculekit.tools._anchor_variants import lookup_anchor
from moleculekit.residues import ORIGINAL_RESIDUE_NAME_TABLE
from moleculekit.tools.nonstandard_residues import (
    detectNonStandardResidues,
    ChainResidueSpec,
    ScaffoldSpec,
    CovalentLigandSpec,
    LigandSpec,
    PROTEIN_RESNAMES,
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


def test_anchor_variants_lookup():
    # Retained as a smoke-test; test_lookup_anchor gives the canonical
    # post-Task-2 assertions.  Updated to use the new lookup_anchor API.
    e = lookup_anchor("CYS", "SG")
    assert e is not None
    assert e["ff_variant"] == "CYX"
    assert e["drop_h"] == ["HG"]
    assert e["ff14sb_type"] == "S"

    # Variant resname (CYX) routes via ORIGINAL_RESIDUE_NAME_TABLE back to CYS
    # and yields the same entry.
    assert lookup_anchor("CYX", "SG") == e
    assert ORIGINAL_RESIDUE_NAME_TABLE["CYX"] == "CYS"

    # Unknown anchor: missing entry, not an error.
    assert lookup_anchor("ALA", "CA") is None
    assert lookup_anchor("UNK", "X1") is None


def test_1u5u_heme_iron_tyr_coordination():
    """1U5U: Fe in HEM is axially coordinated by TYR353-OH (chains A and B).
    The bcif reader stores this as an 'mc' bond. detectNonStandardResidues
    treats it as a non-peptide partner: HEM becomes a CovalentLigandSpec
    (not a free LigandSpec) and TYR353 gets a ChainResidueSpec with
    anchor_atom='OH', because the Tyr's protonation state (Tyr-O-) changes
    when coordinated and needs a custom prepi.

    Coordinations to standalone ion residues (3PTB's Ca2+) and to water
    (HEM-Fe...HOH) are still skipped via the ion/water residue-name filter."""
    mol = Molecule("1u5u")
    specs = detectNonStandardResidues(mol)

    tyrs = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "TYR"
    ]
    hems = [s for s in specs if isinstance(s, CovalentLigandSpec) and s.resname == "HEM"]

    # One TYR353 per chain (A, B) gets flagged, anchored on OH.
    assert len(tyrs) == 2
    assert {t.residue.chain for t in tyrs} == {"A", "B"}
    assert all(t.anchor_atom == "OH" and t.residue.resid == 353 for t in tyrs)
    # Both TYR353s land in the same bucket (same canonical/anchor/partner)
    # so they share one custom resname.
    assert len({t.new_resname for t in tyrs}) == 1

    # HEM is demoted from free LigandSpec to CovalentLigandSpec.
    assert len(hems) == 2
    assert {h.residue.chain for h in hems} == {"A", "B"}
    assert not any(
        isinstance(s, LigandSpec) and s.resname == "HEM" for s in specs
    )


def test_3ptb_calcium_coordination_skips_ion_residue():
    """3PTB: Ca2+ ion coordinated by 4 protein O + 2 waters. The ion lives
    in its own residue (resname 'CA'), which is in _ION_RESNAMES, so every
    Ca-O 'mc' bond is filtered out by detectNonStandardResidues. Only the
    BEN ligand surfaces as a spec."""
    mol = Molecule("3ptb")
    specs = detectNonStandardResidues(mol)
    assert len(specs) == 1
    assert isinstance(specs[0], LigandSpec)
    assert specs[0].resname == "BEN"


def test_1m63_free_iron_coordination_skipped():
    """1M63: the calcineurin Fe/Zn binuclear centre has Fe coordinated by
    Asp90 / His92 / Asp118. The iron is a standalone single-atom residue
    named 'FE', which is not in _ION_RESNAMES but is recognised as a free
    metal ion from its element-symbol resname (_METAL_ION_RESNAMES). Its
    coordination bonds are therefore skipped, exactly like the Zn2+ and Ca2+
    ions, so detection neither crashes on the Asp-Fe bond nor classifies the
    ion as a ligand to parameterize. The only specs are the cyclosporin
    NCAAs (two cyclic copies of DAL/ABA/SAR/MLE/MVA/BMT)."""
    mol = Molecule("1M63")
    specs = detectNonStandardResidues(mol)

    # No free metal ion surfaces as a spec.
    assert not any(
        s.resname in ("FE", "ZN", "CA") for s in specs
    ), "a free metal ion was classified as a non-standard residue"

    # The cyclosporin NCAAs are all chain-resident specs (no covalent / ligand
    # / scaffold spurious classifications from the metal-coordination bonds).
    assert all(isinstance(s, ChainResidueSpec) for s in specs)
    from collections import Counter
    counts = Counter(s.resname for s in specs)
    assert counts == {
        "MLE": 8, "DAL": 2, "MVA": 2, "BMT": 2, "ABA": 2, "SAR": 2
    }, counts


def _backbone_n_acyl_isopeptide(acceptor_resname):
    """An ``acceptor_resname`` residue whose BACKBONE amide N is acylated by
    the side-chain gamma-carbonyl (CD) of a glutamate donor - the generic
    backbone-N isopeptide seen in 6S6Y's poly-gamma-glutamate methanofuran
    tails (there the acceptor is GLU). Explicit bonds; geometry placed so no
    spurious backbone peptide contacts are guessed."""
    acc_names = ["N", "CA", "C", "O", "CB"]
    acc_elem = ["N", "C", "C", "O", "C"]
    don_names = ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"]
    don_elem = ["N", "C", "C", "O", "C", "C", "C", "O", "O"]
    na = len(acc_names)
    mol = Molecule().empty(na + len(don_names))
    mol.name[:] = acc_names + don_names
    mol.element[:] = acc_elem + don_elem
    mol.resname[:] = [acceptor_resname] * na + ["GLU"] * len(don_names)
    mol.resid[:] = [1] * na + [2] * len(don_names)
    mol.chain[:] = "A"
    mol.segid[:] = "P0"
    mol.record[:] = "ATOM"
    acc_xyz = [[-1.33, 0, 0], [-2.5, 0.5, 0], [-3.5, 0, 0], [-3.5, -1, 0], [-2.5, 2, 0]]
    don_xyz = [[5, -1, 0], [4.5, 0, 0], [5.5, 1, 0], [5.5, 2, 0], [3, 0, 0],
               [1.5, 0, 0], [0, 0, 0], [0.6, 1, 0], [0.6, -1, 0]]  # CD (idx 6) at origin
    mol.coords = np.array(acc_xyz + don_xyz, dtype=np.float32).reshape(-1, 3, 1)
    acc_intra = [(0, 1), (1, 2), (2, 3), (1, 4)]
    don_intra = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (6, 7), (6, 8)]
    bonds = acc_intra + [(a + na, b + na) for a, b in don_intra] + [(0, na + 6)]
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def test_detect_backbone_n_isopeptide_is_generic():
    """A canonical residue whose backbone N is acylated by an external
    side-chain carbonyl (a gamma-glutamyl isopeptide, 6S6Y's
    poly-gamma-glutamate) must be recognised, not crash with 'unrecognized
    bond ... <res>-N'. The backbone N is the same atom in every amino acid, so
    this works for ANY canonical residue, not only GLU. The acceptor's anchor
    is its backbone N; the glutamate donor keeps its side-chain CD anchor."""
    for acceptor in ("GLU", "ALA", "LEU"):
        mol = _backbone_n_acyl_isopeptide(acceptor)
        specs = detectNonStandardResidues(mol)
        by_resid = {
            s.residue.resid: s for s in specs if isinstance(s, ChainResidueSpec)
        }
        assert set(by_resid) == {1, 2}, (acceptor, [type(s).__name__ for s in specs])
        assert by_resid[1].resname == acceptor, acceptor
        assert by_resid[1].anchor_atom == "N", (acceptor, by_resid[1].anchor_atom)
        assert by_resid[2].anchor_atom == "CD", (acceptor, by_resid[2].anchor_atom)


def test_backbone_n_isopeptide_is_not_n_terminal():
    """A backbone N acylated by an inter-residue bond (peptide OR isopeptide) is
    NOT a free terminus. The acceptor's N is bonded to the donor's CD, so its
    is_n_term must be False (its C is unbonded, so is_c_term stays True). 1FJM's
    DAM relies on this: its is_n_term must reflect the FGA gamma-glutamyl bond on
    its backbone N, not just standard N-C peptide bonds."""
    mol = _backbone_n_acyl_isopeptide("GLU")
    specs = detectNonStandardResidues(mol)
    acceptor = next(
        s for s in specs if isinstance(s, ChainResidueSpec) and s.residue.resid == 1
    )
    assert acceptor.is_n_term is False, "backbone N is isopeptide-bonded, not free"
    assert acceptor.is_c_term is True, "backbone C is unbonded (free terminus)"


def test_8qfz_scaffolded_peptide():
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
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]

    assert len(scaffolds) == 1 and scaffolds[0].resname == "LFI"
    assert len(renames) == 3
    assert {r.resname for r in renames} == {"CYS"}
    new_names = {r.new_resname for r in renames}
    # Three distinct chain positions -> three distinct rename targets.
    assert len(new_names) == 3, f"expected three distinct renames, got {new_names}"
    for n in new_names:
        assert len(n) == 3 and n.startswith("X")

    assert all(isinstance(s, (ScaffoldSpec, ChainResidueSpec)) for s in specs)


def test_single_anchor_demotes_scaffold_to_covalent_ligand():
    """Strip two of the three CYS-LFI bonds: LFI now has only one anchor
    so it's a CovalentLigandSpec rather than a ScaffoldSpec, and only the
    one remaining CYS gets renamed."""
    mol = _drop_crosslinks(Molecule(QFZ_B_CIF), "LFI", 2)

    specs = detectNonStandardResidues(mol)
    cov = [s for s in specs if isinstance(s, CovalentLigandSpec)]
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]

    assert scaffolds == []
    assert len(cov) == 1 and cov[0].resname == "LFI"
    assert len(renames) == 1 and renames[0].resname == "CYS"


def test_5vbl_ncaas_and_free_ligand():
    """5VBL chain A peptide inhibitor: five chain-resident NCAAs with no
    crosslinks, plus a free OLC ligand. Detector emits ChainResidueSpec entries
    for NCAAs, ChainResidueSpec renames for GLU/LYS isopeptide, and one LigandSpec."""
    mol = Molecule(VBL_PDB)
    specs = detectNonStandardResidues(mol)

    ncaas = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname not in PROTEIN_RESNAMES
        and s.anchor_atom is None
    ]
    crosslinked = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname not in PROTEIN_RESNAMES
        and s.anchor_atom is not None
    ]
    ligands = [s for s in specs if isinstance(s, LigandSpec)]
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]
    cov = [s for s in specs if isinstance(s, CovalentLigandSpec)]

    assert crosslinked == [] and scaffolds == [] and cov == []
    # 5VBL's GLU 10 - LYS 13 isopeptide produces 2 renames; chain B has 4 CYS
    # disulfide renames (CYX); total renames >= 2.
    glu_lys_renames = [r for r in renames if r.resname in ("GLU", "LYS")]
    assert len(glu_lys_renames) == 2
    assert {r.resname for r in glu_lys_renames} == {"GLU", "LYS"}

    assert sorted(s.resname for s in ncaas) == ["200", "ALC", "HRG", "NLE", "OIC"]
    # Resid 17 = "200" is the C-terminal NCAA: no peptide tail bond.
    for s in ncaas:
        if s.resname == "200":
            assert s.is_c_term and not s.is_n_term
        else:
            assert not s.is_n_term and not s.is_c_term

    assert len(ligands) == 1
    assert ligands[0].resname == "OLC" and ligands[0].residue.chain == "B"


def test_1r1j_covalent_glycosylation():
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
        if isinstance(s, ChainResidueSpec) and s.resname == "ASN"
        and s.new_resname is not None
    ]
    assert len(cov) == 3
    assert len(asn_renames) == 3
    new_names = {r.new_resname for r in asn_renames}
    assert len(new_names) == 1, f"expected one shared rename, got {new_names}"
    shared = next(iter(new_names))
    assert len(shared) == 3 and shared.startswith("X")
    for r in asn_renames:
        rid = int(r.residue.resid)
        names = _residue_atom_names(mol, r.resname, rid)
        assert "ND2" in names

    oir = [s for s in specs if s.residue.resname == "OIR"]
    assert len(oir) == 1
    assert isinstance(oir[0], LigandSpec), (
        f"OIR coordinates Zn via O19/S26 but has no covalent partner; "
        f"expected LigandSpec, got {type(oir[0]).__name__}"
    )


def test_8qu4_ncaa_crosslink():
    """8QU4 chain A stapled peptide: NLE272 + MK8276 are two NCAAs joined
    by a sidechain CE-CE staple. Detector emits a CrosslinkedNCAASpec for
    each; no CanonicalRenamedSpec because neither residue is canonical."""
    mol = Molecule(QU4_A_CIF)
    specs = detectNonStandardResidues(mol)

    crosslinked = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname not in PROTEIN_RESNAMES
        and s.anchor_atom is not None
    ]
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]

    assert renames == [] and scaffolds == []
    assert sorted(s.resname for s in crosslinked) == ["MK8", "NLE"]
    # Both residues are mid-chain (not termini).
    for s in crosslinked:
        assert not s.is_n_term and not s.is_c_term


def test_scaffold_anchored_on_ncaa_sidechains():
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
    crosslinked = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname not in PROTEIN_RESNAMES
        and s.anchor_atom is not None
    ]
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]

    assert len(scaffolds) == 1 and scaffolds[0].resname == "SCF"
    assert sorted(s.resname for s in crosslinked) == ["MK8", "NLE"]
    assert renames == []


def test_canonical_to_ncaa_crosslink():
    """A non-peptide bond between a canonical AA (CYS-SG) and an NCAA emits a
    ChainResidueSpec for the NCAA plus a ChainResidueSpec rename (CYX) for the
    canonical residue (a single-anchor scenario, so no ScaffoldSpec)."""
    mol = Molecule(QU4_A_CIF)
    mol.segid[:] = "P"
    # Rename MK8 -> CYS and relabel the crosslink atom CE -> SG so the
    # existing CE-CE staple becomes an asymmetric NLE.CE -- CYS.SG bond.
    mk8_mask = mol.resname == "MK8"
    mol.resname[mk8_mask] = "CYS"
    # Rename the CE atom on the new CYS to SG (CYS anchors at SG).
    ce_mask = mk8_mask & (mol.name == "CE")
    mol.name[ce_mask] = "SG"

    specs = detectNonStandardResidues(mol)
    crosslinked = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname not in PROTEIN_RESNAMES
        and s.anchor_atom is not None
    ]
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]

    assert len(crosslinked) == 1 and crosslinked[0].resname == "NLE"
    assert len(renames) == 1
    # CYS-SG bonded to an NCAA (not CYS/CYX) gets an auto-generated XX# rename
    # (CYX is only assigned for CYS-SG <-> CYS-SG disulfide bonds).
    assert renames[0].resname == "CYS"
    assert renames[0].anchor_atom == "SG"
    assert renames[0].new_resname is not None and renames[0].new_resname.startswith("X")


def test_distinct_partner_resnames_get_distinct_renames():
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
    renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname in PROTEIN_RESNAMES
    ]
    assert len(renames) == 2
    new_names = sorted({r.new_resname for r in renames})
    assert len(new_names) == 2, f"expected two buckets, got {new_names}"
    for n in new_names:
        assert len(n) == 3 and n.startswith("X")


def test_ncaa_spec_default_new_resname():
    """new_resname defaults to None on ChainResidueSpec and accepts a string when given."""
    rid = UniqueResidueID(
        resname="ALC", chain="A", resid=1, insertion="", segid="A"
    )
    s1 = ChainResidueSpec(resname="ALC", residue=rid, is_n_term=False, is_c_term=False)
    assert s1.new_resname is None
    s2 = ChainResidueSpec(
        resname="ALC",
        residue=rid,
        is_n_term=True,
        is_c_term=False,
        new_resname="NALC",
    )
    assert s2.new_resname == "NALC"

    s3 = ChainResidueSpec(
        resname="ALC", residue=rid, is_n_term=False, is_c_term=False,
        anchor_atom="CE",
    )
    assert s3.new_resname is None
    s4 = ChainResidueSpec(
        resname="ALC",
        residue=rid,
        is_n_term=False,
        is_c_term=True,
        anchor_atom="CE",
        new_resname="CALC",
    )
    assert s4.new_resname == "CALC"


def test_disambiguate_single_config_no_rename():
    """One NCAA resname, all instances mid-chain -> no rename."""
    rid_a = UniqueResidueID(
        resname="ALC", chain="A", resid=1, insertion="", segid="A"
    )
    rid_b = UniqueResidueID(
        resname="ALC", chain="A", resid=2, insertion="", segid="A"
    )
    specs = [
        ChainResidueSpec(resname="ALC", residue=rid_a, is_n_term=False, is_c_term=False),
        ChainResidueSpec(resname="ALC", residue=rid_b, is_n_term=False, is_c_term=False),
    ]
    _disambiguate_terminus_resnames(specs)
    assert all(s.new_resname is None for s in specs)


def test_disambiguate_two_configs_renames_nonmid():
    """Same NCAA resname appearing mid-chain AND N-term: mid keeps the
    original (new_resname=None), N-term gets NALC."""
    rid_mid = UniqueResidueID(
        resname="ALC", chain="A", resid=2, insertion="", segid="A"
    )
    rid_n = UniqueResidueID(
        resname="ALC", chain="A", resid=1, insertion="", segid="A"
    )
    s_mid = ChainResidueSpec(
        resname="ALC", residue=rid_mid, is_n_term=False, is_c_term=False
    )
    s_n = ChainResidueSpec(
        resname="ALC", residue=rid_n, is_n_term=True, is_c_term=False
    )
    _disambiguate_terminus_resnames([s_mid, s_n])
    assert s_mid.new_resname is None
    assert s_n.new_resname == "NALC"


def test_disambiguate_three_configs():
    """Mid + N + C-term all present -> mid stays, N -> NALC, C -> CALC."""
    def _spec(i, n, c):
        return ChainResidueSpec(
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


def test_disambiguate_both_terminus_uses_b_prefix():
    """A single-residue chain (both is_n_term and is_c_term True) coexisting
    with a mid-chain instance of the same resname uses the B prefix."""
    rid_mid = UniqueResidueID(
        resname="ALC", chain="A", resid=2, insertion="", segid="A"
    )
    rid_both = UniqueResidueID(
        resname="ALC", chain="B", resid=1, insertion="", segid="B"
    )
    s_mid = ChainResidueSpec(
        resname="ALC", residue=rid_mid, is_n_term=False, is_c_term=False
    )
    s_both = ChainResidueSpec(
        resname="ALC", residue=rid_both, is_n_term=True, is_c_term=True
    )
    _disambiguate_terminus_resnames([s_mid, s_both])
    assert s_mid.new_resname is None
    assert s_both.new_resname == "BALC"


def test_disambiguate_applies_to_crosslinked_ncaa():
    """The disambiguator treats crosslinked-NCAA ChainResidueSpec the same as plain NCAA."""
    rid_mid = UniqueResidueID(
        resname="MK8", chain="A", resid=2, insertion="", segid="A"
    )
    rid_n = UniqueResidueID(
        resname="MK8", chain="A", resid=1, insertion="", segid="A"
    )
    s_mid = ChainResidueSpec(
        resname="MK8", residue=rid_mid, is_n_term=False, is_c_term=False,
        anchor_atom="CE",
    )
    s_n = ChainResidueSpec(
        resname="MK8", residue=rid_n, is_n_term=True, is_c_term=False,
        anchor_atom="CE",
    )
    _disambiguate_terminus_resnames([s_mid, s_n])
    assert s_mid.new_resname is None
    assert s_n.new_resname == "NMK8"


def test_disambiguate_independent_groups():
    """Two different NCAA resnames, each with a single config, both
    stay un-renamed. Disambiguation is per-resname, not global."""
    s_alc = ChainResidueSpec(
        resname="ALC",
        residue=UniqueResidueID(
            resname="ALC", chain="A", resid=1, insertion="", segid="A"
        ),
        is_n_term=False,
        is_c_term=False,
    )
    s_hrg = ChainResidueSpec(
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


def test_disambiguate_4char_input_raises():
    """If disambiguation is required AND the input resname is 4+ chars,
    the prefixed name would exceed the 4-char AMBER prepi unit-name limit."""
    import pytest as _pytest

    s_mid = ChainResidueSpec(
        resname="ABCD",
        residue=UniqueResidueID(
            resname="ABCD", chain="A", resid=1, insertion="", segid="A"
        ),
        is_n_term=False,
        is_c_term=False,
    )
    s_n = ChainResidueSpec(
        resname="ABCD",
        residue=UniqueResidueID(
            resname="ABCD", chain="A", resid=2, insertion="", segid="A"
        ),
        is_n_term=True,
        is_c_term=False,
    )
    with _pytest.raises(RuntimeError, match="ABCD"):
        _disambiguate_terminus_resnames([s_mid, s_n])


def test_disambiguate_ignores_other_spec_types():
    """The disambiguator only touches ChainResidueSpec entries with non-protein resname.
    Other specs (LigandSpec, ScaffoldSpec, etc.) are left untouched."""
    rid = UniqueResidueID(
        resname="OLC", chain="B", resid=1, insertion="", segid="B"
    )
    other = LigandSpec(resname="OLC", residue=rid)
    s_mid = ChainResidueSpec(
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


def test_detect_no_rename_when_all_one_config():
    """5VBL has 200/ALC/HRG/NLE/OIC each appearing exactly once and the
    one C-terminal residue is 200 (no other resname appears at the C
    terminus). After Task 3 the detector still emits new_resname=None
    for all of them - no disambiguation needed."""
    mol = Molecule(VBL_PDB)
    specs = detectNonStandardResidues(mol)
    ncaas = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname not in PROTEIN_RESNAMES
    ]
    assert len(ncaas) >= 5
    for s in ncaas:
        assert s.new_resname is None, (
            f"{s.resname} unexpectedly got new_resname={s.new_resname!r}"
        )


def test_detect_renames_when_resname_shared_across_configs():
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
        if isinstance(s, ChainResidueSpec) and s.resname == "ALC"
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


def test_detect_unaffected_groups_left_alone():
    """When ALC needs disambiguation but HRG does not, only ALC specs
    get new_resname set."""
    mol = _make_vbl_with_z_alc_nterminal()

    specs = detectNonStandardResidues(mol)
    hrg_specs = [
        s
        for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "HRG"
    ]
    for s in hrg_specs:
        assert s.new_resname is None, (
            f"HRG spec on chain {s.residue.chain} unexpectedly renamed to "
            f"{s.new_resname!r}; only ALC should be disambiguated."
        )


def test_apply_detect_spec_renames_for_ncaa():
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

    spec = ChainResidueSpec(
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


def test_apply_detect_spec_renames_skips_none():
    """new_resname=None leaves mol.resname untouched."""
    from moleculekit.tools.preparation import _apply_detect_spec_renames

    mol = Molecule(VBL_PDB)
    before = mol.resname.copy()

    alc_resid = int(np.unique(mol.resid[mol.resname == "ALC"])[0])
    spec = ChainResidueSpec(
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


def test_apply_detect_spec_renames_just_renames():
    """With H-drop moved upstream to _template_renamed_canonical_residues,
    _apply_detect_spec_renames is now a rename-only safety net."""
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

    spec = ChainResidueSpec(
        resname="CYS",
        residue=UniqueResidueID(
            resname="CYS", chain="A", resid=1, insertion="", segid="A"
        ),
        new_resname="CY1",
    )
    _apply_detect_spec_renames(mol, [spec])
    assert (mol.resname == "CY1").all()
    assert "HG" in set(mol.name)  # NOT dropped here anymore
    assert mol.numAtoms == 4


def test_anchor_table_covers_all_anchors():
    """ANCHOR_TABLE is the single source of truth for per-anchor facts.
    Covers every existing ANCHOR_VARIANTS anchor (CYS-SG, ASN-ND2,
    LYS-NZ, TYR-OH, HIS-ND1/NE2, SER-OG, THR-OG1) plus the new
    canonical-canonical amide-carbonyl anchors (GLU-CD, ASP-CG,
    ASN-CG, GLN-CD)."""
    from moleculekit.tools._anchor_variants import ANCHOR_TABLE

    expected = {
        ("CYS", "SG"), ("LYS", "NZ"), ("TYR", "OH"),
        ("HIS", "ND1"), ("HIS", "NE2"),
        ("SER", "OG"), ("THR", "OG1"),
        ("ASN", "ND2"),
        ("GLU", "CD"), ("ASP", "CG"), ("ASN", "CG"), ("GLN", "CD"),
    }
    assert set(ANCHOR_TABLE) == expected


def test_lookup_anchor():
    """lookup_anchor returns the table entry for known anchors and None
    for unknown ones. Variant resnames (CYX, LYN) route back to the
    base via ORIGINAL_RESIDUE_NAME_TABLE."""
    from moleculekit.tools._anchor_variants import lookup_anchor

    entry = lookup_anchor("CYS", "SG")
    assert entry is not None
    assert entry["displaced_heavy"] == ()
    assert entry["ff_variant"] == "CYX"
    assert entry["drop_h"] == ["HG"]
    assert entry["ff14sb_type"] == "S"

    glu = lookup_anchor("GLU", "CD")
    assert glu["displaced_heavy"] == ("OE2",)
    assert glu["smiles_variant"] == "GLU"
    assert glu["ff_variant"] is None

    assert lookup_anchor("CYX", "SG") == entry
    assert lookup_anchor("LYN", "NZ") is not None

    assert lookup_anchor("MET", "SD") is None
    assert lookup_anchor("ALA", "CB") is None


def test_canonical_anchor_smiles():
    """canonical_anchor_smiles returns RESIDUE_SMILES[smiles_variant].
    Picks LYN for LYS NZ; HID/HIE for HIS; canonical resname otherwise."""
    from moleculekit.residues import RESIDUE_SMILES
    from moleculekit.tools._anchor_variants import canonical_anchor_smiles

    assert canonical_anchor_smiles("LYS", "NZ") == RESIDUE_SMILES["LYN"]
    assert canonical_anchor_smiles("HIS", "ND1") == RESIDUE_SMILES["HID"]
    assert canonical_anchor_smiles("HIS", "NE2") == RESIDUE_SMILES["HIE"]
    assert canonical_anchor_smiles("GLU", "CD") == RESIDUE_SMILES["GLU"]
    assert canonical_anchor_smiles("ASN", "CG") == RESIDUE_SMILES["ASN"]
    assert canonical_anchor_smiles("CYS", "SG") == RESIDUE_SMILES["CYS"]
    assert canonical_anchor_smiles("CYX", "SG") == RESIDUE_SMILES["CYS"]
    assert canonical_anchor_smiles("LYN", "NZ") == RESIDUE_SMILES["LYN"]
    import pytest
    with pytest.raises(ValueError, match=r"MET.*SD"):
        canonical_anchor_smiles("MET", "SD")


def test_5vbl_glu_lys_isopeptide_emits_chain_residue_specs():
    """5VBL has an isopeptide bond between GLU A 10 CD and LYS A 13 NZ.
    The detector emits a ChainResidueSpec for each end with auto-
    generated X## new_resnames."""
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec, detectNonStandardResidues,
    )

    mol = Molecule(VBL_PDB)
    n_atoms_before = mol.numAtoms
    specs = detectNonStandardResidues(mol)
    assert mol.numAtoms == n_atoms_before  # non-mutating

    canonical_crosslinks = [
        s for s in specs
        if isinstance(s, ChainResidueSpec)
        and s.resname in ("GLU", "LYS")
    ]
    by_resname = {s.resname: s for s in canonical_crosslinks}
    assert set(by_resname) == {"GLU", "LYS"}
    glu, lys = by_resname["GLU"], by_resname["LYS"]
    assert glu.residue.resid == 10 and lys.residue.resid == 13
    assert glu.new_resname and glu.new_resname.startswith("X")
    assert lys.new_resname and lys.new_resname.startswith("X")
    assert glu.new_resname != lys.new_resname
    assert glu.anchor_atom == "CD"
    assert lys.anchor_atom == "NZ"


def test_cyx_disulfide_emits_chain_residue_specs():
    """Two CYS-SG <-> CYS-SG residues both get new_resname='CYX' (shared,
    no auto-generated XX# suffix)."""
    cys2 = Molecule().empty(12)
    cys2.name[:] = ["N", "CA", "C", "O", "CB", "SG"] * 2
    cys2.element[:] = ["N", "C", "C", "O", "C", "S"] * 2
    cys2.resname[:] = "CYS"
    cys2.resid[:] = [1] * 6 + [2] * 6
    cys2.chain[:] = "A"
    cys2.segid[:] = "A"
    cys2.insertion[:] = ""
    cys2.coords = np.zeros((12, 3, 1), dtype=np.float32)
    cys2.bonds = np.array([[5, 11]], dtype=np.int64)
    cys2.bondtype = np.array(["1"], dtype=object)

    specs = detectNonStandardResidues(cys2)
    cys_specs = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "CYS"
    ]
    assert len(cys_specs) == 2
    assert {s.new_resname for s in cys_specs} == {"CYX"}
    assert all(s.anchor_atom == "SG" for s in cys_specs)


def test_8qfz_three_cys_distinct_buckets():
    """8QFZ has 3 CYS bonded to LFI at distinct chain positions (N-term,
    mid, C-term). Each gets its own auto-generated X## name because the
    bucket key (resname, anchor, partner, n_term, c_term) differs."""
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec, detectNonStandardResidues,
    )
    mol = Molecule(QFZ_B_CIF)
    specs = detectNonStandardResidues(mol)
    cys_specs = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "CYS"
    ]
    assert len(cys_specs) == 3
    new_names = {s.new_resname for s in cys_specs}
    assert len(new_names) == 3
    for n in new_names:
        assert n.startswith("X") and len(n) == 3


def test_1r1j_three_asn_share_bucket():
    """1R1J: three ASN-ND2-NAG glycosylation sites at distinct mid-chain
    positions share one X## new_resname because they share bucket key
    (ASN, ND2, NAG, False, False)."""
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec, detectNonStandardResidues,
    )
    mol = Molecule(R1J_PDB)
    specs = detectNonStandardResidues(mol)
    asn_specs = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "ASN"
        and s.new_resname is not None
    ]
    assert len(asn_specs) == 3
    new_names = {s.new_resname for s in asn_specs}
    assert len(new_names) == 1, (
        f"expected three ASNs to share one new_resname; got {new_names}"
    )


def test_unknown_canonical_anchor_raises():
    """A canonical AA with a non-peptide bond at an anchor not in
    ANCHOR_TABLE raises clearly."""
    from moleculekit.tools.nonstandard_residues import detectNonStandardResidues

    # Two MET residues with an SD-SD bond (no anchor entry for MET-SD).
    mol = Molecule().empty(16)
    mol.name[:] = ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"] * 2
    mol.element[:] = ["N", "C", "C", "O", "C", "C", "S", "C"] * 2
    mol.resname[:] = "MET"
    mol.resid[:] = [1] * 8 + [2] * 8
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.insertion[:] = ""
    mol.coords = np.zeros((16, 3, 1), dtype=np.float32)
    mol.bonds = np.array([[6, 14]], dtype=np.int64)
    mol.bondtype = np.array(["1"], dtype=object)

    import pytest
    with pytest.raises(RuntimeError, match=r"MET.*SD"):
        detectNonStandardResidues(mol)


def test_bonds_to_waters_are_ignored():
    """Bonds between a canonical residue and a water (e.g. a PDB LINK
    record to a coordinating water) must be skipped: waters are not
    covalent partners, so they must neither trigger an
    ``Unsupported canonical-sidechain crosslink anchor`` error nor get
    classified as covalent ligands."""
    mol = Molecule().empty(5)
    mol.name[:] = ["N", "CA", "C", "O", "OW"]
    mol.element[:] = ["N", "C", "C", "O", "O"]
    mol.resname[:] = ["HID", "HID", "HID", "HID", "WAT"]
    mol.resid[:] = [1, 1, 1, 1, 2]
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.insertion[:] = ""
    mol.coords = np.zeros((5, 3, 1), dtype=np.float32)
    # HID backbone O (idx 3) "bonded" to water O (idx 4) - the spurious
    # contact the detector used to flag as an unknown HID-O anchor.
    mol.bonds = np.array([[3, 4]], dtype=np.int64)
    mol.bondtype = np.array(["1"], dtype=object)

    specs = detectNonStandardResidues(mol)
    assert specs == []


def test_guessed_backbone_anchor_is_ignored():
    """A bond *guessed* from coordinates that lands on a canonical AA's
    backbone O (an atom that never forms a real crosslink) must be treated
    as a spurious close contact and ignored, not raised on. This is the
    common modelled-structure case: no input bonds, slightly-off geometry."""
    mol = Molecule().empty(5)
    # ASN backbone N-CA-C-O, plus a free LIG atom placed ~1.2 A from ASN's O.
    mol.name[:] = ["N", "CA", "C", "O", "C1"]
    mol.element[:] = ["N", "C", "C", "O", "C"]
    mol.resname[:] = ["ASN", "ASN", "ASN", "ASN", "LIG"]
    mol.resid[:] = [1, 1, 1, 1, 2]
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.insertion[:] = ""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # N
            [1.5, 0.0, 0.0],  # CA
            [2.0, 1.3, 0.0],  # C
            [3.2, 1.3, 0.0],  # O  (C-O ~1.2)
            [4.4, 1.3, 0.0],  # C1 (O-C1 ~1.2 -> guessed bond to backbone O)
        ],
        dtype=np.float32,
    )
    mol.coords = coords.reshape(5, 3, 1)
    # No mol.bonds: the detector guesses them, and the only inter-residue
    # contact is the spurious ASN-O <-> LIG-C1 one.

    specs = detectNonStandardResidues(mol)

    # ASN is left alone (no ChainResidueSpec); LIG stays a free ligand.
    assert not any(isinstance(s, ChainResidueSpec) for s in specs)
    assert [type(s) for s in specs] == [LigandSpec]
    assert specs[0].resname == "LIG"


def test_guess_bonds_false_skips_guessing():
    """guess_bonds=False must skip distance-based bond guessing entirely:
    a bondless input whose geometry would otherwise guess a sidechain
    crosslink at an unknown anchor raises by default, but not when guessing
    is disabled."""
    import pytest

    # MET-SD ~1.2 A from a free LIG atom; SD is a sidechain atom with no
    # ANCHOR_TABLE entry, so a guessed SD bond is a genuine "unknown anchor".
    mol = Molecule().empty(3)
    mol.name[:] = ["CA", "SD", "C1"]
    mol.element[:] = ["C", "S", "C"]
    mol.resname[:] = ["MET", "MET", "LIG"]
    mol.resid[:] = [1, 1, 2]
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.insertion[:] = ""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # CA
            [1.81, 0.0, 0.0],  # SD (CA-SD ~1.81)
            [3.0, 0.0, 0.0],  # C1 (SD-C1 ~1.19 -> guessed bond to MET SD)
        ],
        dtype=np.float32,
    )
    mol.coords = coords.reshape(3, 3, 1)

    # Default guesses the SD bond and rejects the unknown anchor; the message
    # makes clear the bonds were guessed.
    with pytest.raises(RuntimeError, match=r"MET.*SD.*guess"):
        detectNonStandardResidues(mol)

    # With guessing disabled there are no bonds, so no crosslink is seen.
    specs = detectNonStandardResidues(mol, guess_bonds=False)
    assert not any(isinstance(s, ChainResidueSpec) for s in specs)
    assert [type(s) for s in specs] == [LigandSpec]


def test_template_renamed_canonical_residues_5vbl():
    """Calling _template_renamed_canonical_residues on 5VBL specs
    renames GLU 10 to XX#, LYS 13 to XX#, drops OE2 on GLU, places
    exactly 1 H on LYS NZ (secondary amide), and preserves the CD-NZ
    bond."""
    from moleculekit.tools.preparation import (
        _template_renamed_canonical_residues,
    )
    from moleculekit.tools.nonstandard_residues import (
        detectNonStandardResidues,
    )

    mol = Molecule(VBL_PDB)
    specs = detectNonStandardResidues(mol)

    mol_out = mol.copy()
    _template_renamed_canonical_residues(mol_out, specs)

    # GLU 10 renamed to XX#; OE2 dropped (heavy-atom leaving group).
    glu_spec = next(
        s for s in specs
        if s.resname == "GLU" and s.residue.resid == 10
    )
    lys_spec = next(
        s for s in specs
        if s.resname == "LYS" and s.residue.resid == 13
    )
    glu_names = _residue_atom_names(mol_out, glu_spec.new_resname, 10)
    lys_names = _residue_atom_names(mol_out, lys_spec.new_resname, 13)
    assert "CD" in glu_names
    assert "OE1" in glu_names
    assert "OE2" not in glu_names
    assert "NZ" in lys_names

    # LYS NZ has exactly 1 H.
    nz_new = int(mol_out.atomselect(
        f"resname {lys_spec.new_resname} and resid 13 and name NZ",
        indexes=True,
    )[0])
    nz_h = [
        nb for nb in mol_out.getNeighbors(nz_new)
        if mol_out.element[int(nb)] == "H"
    ]
    assert len(nz_h) == 1, (
        f"expected 1 H on isopeptide NZ, got "
        f"{[mol_out.name[int(i)] for i in nz_h]}"
    )

    # CD-NZ bond preserved.
    cd_new = int(mol_out.atomselect(
        f"resname {glu_spec.new_resname} and resid 10 and name CD",
        indexes=True,
    )[0])
    pair = sorted([cd_new, nz_new])
    assert any(
        sorted([int(b[0]), int(b[1])]) == pair for b in mol_out.bonds
    ), "CD-NZ bond lost during pre-templating"


def test_template_renamed_canonical_residues_known_variant_skips_templating():
    """A spec renaming CYS to CYX (disulfide) gets the rename but NOT a
    SMILES re-template, since PDB2PQR understands CYX natively."""
    from moleculekit.tools.preparation import (
        _template_renamed_canonical_residues,
    )
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec, detectNonStandardResidues,
    )

    cys2 = Molecule().empty(12)
    cys2.name[:] = ["N", "CA", "C", "O", "CB", "SG"] * 2
    cys2.element[:] = ["N", "C", "C", "O", "C", "S"] * 2
    cys2.resname[:] = "CYS"
    cys2.resid[:] = [1] * 6 + [2] * 6
    cys2.chain[:] = "A"
    cys2.segid[:] = "A"
    cys2.insertion[:] = ""
    cys2.coords = np.zeros((12, 3, 1), dtype=np.float32)
    cys2.bonds = np.array([[5, 11]], dtype=np.int64)
    cys2.bondtype = np.array(["1"], dtype=object)

    specs = detectNonStandardResidues(cys2)
    _template_renamed_canonical_residues(cys2, specs)
    assert all(cys2.resname == "CYX")
    assert cys2.numAtoms == 12  # no atoms added (CYX skips templating)


RKP_PDB = os.path.join(curr_dir, "pdb", "3rkp.pdb")


def test_3rkp_asn_lys_pilin_isopeptide():
    """3RKP has multiple LYS NZ - ASN CG pilin isopeptides. The detector
    emits ChainResidueSpec for each end with auto-generated XX## names."""
    mol = Molecule(RKP_PDB)
    specs = detectNonStandardResidues(mol)
    asn_specs = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "ASN"
        and s.new_resname is not None
    ]
    lys_specs = [
        s for s in specs
        if isinstance(s, ChainResidueSpec) and s.resname == "LYS"
        and s.new_resname is not None
    ]
    assert len(asn_specs) >= 1
    assert len(lys_specs) >= 1
    # ASN side bonds via CG (the amide carbonyl carbon).
    for s in asn_specs:
        assert s.anchor_atom == "CG"
    # LYS side bonds via NZ.
    for s in lys_specs:
        assert s.anchor_atom == "NZ"


def test_systemprepare_5vbl_glu_lys_isopeptide_end_to_end():
    """systemPrepare on 5VBL with the canonical-canonical GLU-LYS
    isopeptide: renames GLU 10 / LYS 13, runs PDB2PQR with the
    pre-templated mol, returns a prepared mol where CD-NZ is still
    bonded and NZ has the expected (single) H of an amide nitrogen."""
    try:
        from moleculekit.tools.preparation import systemPrepare
    except ImportError:
        import pytest
        pytest.skip("pdb2pqr not available")

    mol = Molecule(VBL_PDB)
    # 5VBL has NCAAs (HRG, ALC, NLE, OIC, 200, OLC) which must be
    # pre-templated before systemPrepare. Otherwise PDB2PQR adds Hs
    # that aren't covered by ``_restore_termini_bonds`` (the crystal
    # input has no Hs, so capture has nothing to restore for the
    # sidechain Hs) and ``_assert_specs_bonded`` flags them.
    mol.remove("element H", _logger=False)
    smiles = {
        "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",
        "ALC": "C1CCC(CC1)C[C@@H](C=O)N",
        "HRG": "C(CCNC(=N)N)C[C@@H](C=O)N",
        "NLE": "CCCC[C@@H](C=O)N",
        "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C=O",
        "OLC": "CCCCCCCC(=O)OC[C@H](O)CO",
    }
    for resname, smi in smiles.items():
        if (mol.resname == resname).any():
            mol.templateResidueFromSmiles(
                f'resname "{resname}"', smi, addHs=True, _logger=False
            )
    specs = detectNonStandardResidues(mol)
    prepared, _ = systemPrepare(mol, detect_specs=specs, verbose=False)

    glu = next(
        s for s in specs
        if s.resname == "GLU" and s.residue.resid == 10
    )
    lys = next(
        s for s in specs
        if s.resname == "LYS" and s.residue.resid == 13
    )
    assert any(prepared.resname == glu.new_resname)
    assert any(prepared.resname == lys.new_resname)
    cd_mask = (
        (prepared.resname == glu.new_resname)
        & (prepared.resid == 10)
        & (prepared.name == "CD")
    )
    nz_mask = (
        (prepared.resname == lys.new_resname)
        & (prepared.resid == 13)
        & (prepared.name == "NZ")
    )
    cd = int(np.where(cd_mask)[0][0])
    nz = int(np.where(nz_mask)[0][0])
    pair = sorted([cd, nz])
    assert any(
        sorted([int(b[0]), int(b[1])]) == pair for b in prepared.bonds
    ), "CD-NZ bond lost during systemPrepare"
    nz_h = [
        nb for nb in prepared.getNeighbors(nz)
        if prepared.element[int(nb)] == "H"
    ]
    assert len(nz_h) == 1


def test_systemprepare_8qfz_canonical_to_ncaa_end_to_end():
    """8QFZ: three CYS-SG-LFI thioethers. After systemPrepare, each
    renamed CYS still has SG with no HG."""
    try:
        from moleculekit.tools.preparation import systemPrepare
    except ImportError:
        import pytest
        pytest.skip("pdb2pqr not available")

    mol = Molecule(QFZ_B_CIF)
    specs = detectNonStandardResidues(mol)
    prepared, _ = systemPrepare(mol, detect_specs=specs, verbose=False)
    cys_renames = [
        s for s in specs
        if isinstance(s, ChainResidueSpec)
        and s.resname == "CYS"
        and s.new_resname
    ]
    assert len(cys_renames) == 3
    for spec in cys_renames:
        rid = spec.residue
        names = _residue_atom_names(prepared, spec.new_resname, int(rid.resid))
        assert "SG" in names
        assert "HG" not in names
