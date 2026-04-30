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
    new resname."""
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
