import os
import numpy as np
from moleculekit.molecule import Molecule
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
    """8QFZ chain B: LFI scaffold thio-ether bonded to three CYS sidechains.
    All three CYS share the (CYS, SG, LFI) bucket so they collapse onto
    the same custom 3-char resname (one shared parameterization). The
    fixture has no explicit HG hydrogens so no atoms are dropped; the
    rename still happens."""
    mol = Molecule(QFZ_B_CIF)
    n_atoms_before = mol.numAtoms
    assert (mol.resname == "CYS").sum() > 0

    specs = detectNonStandardResidues(mol)

    # All three CYS residues collapse to one bucket -> one new resname.
    assert (mol.resname == "CYS").sum() == 0
    assert mol.numAtoms == n_atoms_before

    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]

    assert len(scaffolds) == 1 and scaffolds[0].resname == "LFI"
    assert len(renames) == 3
    assert {r.original_resname for r in renames} == {"CYS"}
    new_names = {r.new_resname for r in renames}
    assert len(new_names) == 1, f"expected one shared rename, got {new_names}"
    new_resname = next(iter(new_names))
    assert len(new_resname) == 3 and new_resname.startswith("CY")
    # The renamed residue resname in mol must match the spec.
    assert (mol.resname == new_resname).sum() > 0
    for r in renames:
        assert r.residue.resname == new_resname

    # No other spec types for this fixture.
    assert all(isinstance(s, (ScaffoldSpec, CanonicalRenamedSpec)) for s in specs)


def _test_h_drop_for_canonical_anchor():
    """Insert an HG atom into one of the LFI-anchored cysteines (right
    after that residue's existing atoms so it ends up in the same residue
    group), then verify detect removes it as part of the rename."""
    mol = Molecule(QFZ_B_CIF)
    cys_resid = int(sorted(set(mol.resid[mol.resname == "CYS"]))[0])
    cys_atom_idxs = mol.atomselect(f"resname CYS and resid {cys_resid}", indexes=True)
    sg_idx = int(
        mol.atomselect(f"resname CYS and resid {cys_resid} and name SG", indexes=True)[
            0
        ]
    )
    insert_at = int(cys_atom_idxs[-1]) + 1

    hg = Molecule().empty(1)
    hg.name[:] = ["HG"]
    hg.element[:] = ["H"]
    hg.resname[:] = "CYS"
    hg.resid[:] = cys_resid
    hg.chain[:] = mol.chain[sg_idx]
    hg.segid[:] = mol.segid[sg_idx]
    hg.coords = (mol.coords[sg_idx, :, mol.frame] + np.array([1.0, 0.0, 0.0]))[
        np.newaxis, :, np.newaxis
    ].astype(np.float32)
    hg.record[:] = "ATOM"
    n_before = mol.numAtoms
    mol.insert(hg, insert_at)
    # SG's index is unchanged because we inserted after it; HG sits at insert_at.
    mol.addBond(sg_idx, insert_at, "1")

    detectNonStandardResidues(mol)
    # The injected HG must be gone after detect.
    assert mol.numAtoms == n_before
    assert (mol.name == "HG").sum() == 0


def _test_min_anchors_threshold_emits_covalent_ligand():
    """Strip two of the three CYS-LFI bonds: LFI now has only one anchor
    so it's a CovalentLigandSpec rather than a ScaffoldSpec, and only the
    one remaining CYS gets renamed."""
    mol = Molecule(QFZ_B_CIF)
    keep = []
    dropped = 0
    lfi = mol.atomselect("resname LFI", indexes=True)
    for b in mol.bonds:
        is_cross = (b[0] in lfi) != (b[1] in lfi)
        if is_cross and dropped < 2:
            dropped += 1
            continue
        keep.append(b)
    mol.bonds = np.asarray(keep, dtype=np.uint32)
    mol.bondtype = mol.bondtype[: len(keep)]

    specs = detectNonStandardResidues(mol)
    cov = [s for s in specs if isinstance(s, CovalentLigandSpec)]
    renames = [s for s in specs if isinstance(s, CanonicalRenamedSpec)]
    scaffolds = [s for s in specs if isinstance(s, ScaffoldSpec)]

    assert scaffolds == []
    assert len(cov) == 1 and cov[0].resname == "LFI"
    assert len(renames) == 1 and renames[0].original_resname == "CYS"


def _test_5vbl_ncaas_and_free_ligand():
    """5VBL chain A peptide inhibitor: five chain-resident NCAAs with no
    crosslinks, plus a free OLC ligand. Detector emits NCAASpec entries
    and one LigandSpec; mol is unchanged."""
    mol = Molecule(VBL_PDB)
    n_atoms_before = mol.numAtoms
    resnames_before = sorted(set(str(r) for r in mol.resname))

    specs = detectNonStandardResidues(mol)

    # No canonical anchors in this fixture, so no mutations.
    assert mol.numAtoms == n_atoms_before
    assert sorted(set(str(r) for r in mol.resname)) == resnames_before

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
    an Asn ND2; all three (ASN, ND2, NAG) buckets are identical so the
    detector emits one CovalentLigandSpec per NAG plus three
    CanonicalRenamedSpec entries that all share the same new resname."""
    mol = Molecule(R1J_PDB)
    specs = detectNonStandardResidues(mol)

    cov = [s for s in specs if isinstance(s, CovalentLigandSpec) and s.resname == "NAG"]
    asn_renames = [
        s
        for s in specs
        if isinstance(s, CanonicalRenamedSpec) and s.original_resname == "ASN"
    ]
    assert len(cov) == 3
    assert len(asn_renames) == 3
    new_names = {r.new_resname for r in asn_renames}
    assert len(new_names) == 1, f"expected one shared rename, got {new_names}"
    shared = next(iter(new_names))
    assert len(shared) == 3 and shared.startswith("NL")
    for r in asn_renames:
        rid = int(r.residue.resid)
        names = _residue_atom_names(mol, shared, rid)
        assert "ND2" in names


def _test_8qu4_ncaa_crosslink():
    """8QU4 chain A stapled peptide: NLE272 + MK8276 are two NCAAs joined
    by a sidechain CE-CE staple. Detector emits a CrosslinkedNCAASpec for
    each; no CanonicalRenamedSpec because neither residue is canonical."""
    mol = Molecule(QU4_A_CIF)
    n_atoms_before = mol.numAtoms
    specs = detectNonStandardResidues(mol)

    # No canonical anchors -> no mol mutation.
    assert mol.numAtoms == n_atoms_before

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
    keep = []
    for b in mol.bonds:
        if {int(b[0]), int(b[1])} == {nle_ce, mk8_ce}:
            continue
        keep.append(b)
    mol.bonds = np.asarray(keep, dtype=np.uint32)
    mol.bondtype = mol.bondtype[: len(keep)]

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
    # ALA has no entry in ANCHOR_VARIANTS so the new resname falls back to
    # the original base with a numeric suffix.
    assert renames[0].original_resname == "ALA"
    assert renames[0].new_resname.startswith("ALA") or renames[
        0
    ].new_resname.startswith("AL")


def _test_distinct_partner_resnames_get_distinct_renames():
    """Two CYS residues, each bonded to a *different* non-canonical
    residue (different ``partner_resname``), produce two distinct
    bucket keys and two distinct rename targets. CYS residues bonded to
    the *same* partner resname collapse into one shared rename."""
    base = Molecule(QFZ_B_CIF)

    # Trim to one CYS-LFI anchor (drop two of the three).
    keep = []
    dropped = 0
    lfi_idx = base.atomselect("resname LFI", indexes=True)
    for bnd in base.bonds:
        is_cross = (bnd[0] in lfi_idx) != (bnd[1] in lfi_idx)
        if is_cross and dropped < 2:
            dropped += 1
            continue
        keep.append(bnd)
    base.bonds = np.asarray(keep, dtype=np.uint32)
    base.bondtype = base.bondtype[: len(keep)]

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
