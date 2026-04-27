import os
import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.tools._anchor_variants import lookup_anchor_variant
from moleculekit.residues import ORIGINAL_RESIDUE_NAME_TABLE
from moleculekit.tools.nonstandard_residues import (
    detectNonStandardResidues,
    forceProtonationFromSpecs,
    custombondsFromSpecs,
    ScaffoldedPeptideSpec,
    CofactorSpec,
    ModelAtom,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(curr_dir, "test_nonstandard_residues")
QFZ_B_CIF = os.path.join(DATA_DIR, "8QFZ_B.cif")


def _test_anchor_variants_lookup():
    # Direct hit on the canonical resname.
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


def _test_8qfz_chain_b_scaffolded_peptide(tmp_path):
    mol = Molecule(QFZ_B_CIF)

    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )

    # Chain B has water (HOH, canonical, skipped) plus the LFI scaffold,
    # which should be the only spec returned.
    assert len(specs) == 1, [s.resname for s in specs]
    spec = specs[0]
    assert isinstance(spec, ScaffoldedPeptideSpec)
    assert spec.category == "scaffolded_peptide"
    assert spec.resname == "LFI"
    assert int(spec.residue.resid) == 101

    anchors = spec.anchors
    assert len(anchors) == 3
    anchor_resids = sorted(int(a.anchor_atom.resid) for a in anchors)
    assert anchor_resids == [11, 17, 22]
    for a in anchors:
        assert a.anchor_atom.name == "SG"
        assert a.anchor_atom.resname == "CYS"
        assert a.scaffold_atom.name in {"C10", "C11", "C12"}
        # The UniqueAtomIDs must round-trip against the input molecule:
        # selecting must hit one atom, and that atom's name/resid must match.
        idx = int(a.anchor_atom.selectAtom(mol))
        assert mol.name[idx] == "SG"
        assert int(mol.resid[idx]) == int(a.anchor_atom.resid)

    # Model compound was written and is loadable.
    cif_path = spec.model_compound_cif
    assert os.path.isfile(cif_path)
    model = Molecule(cif_path)
    # 18 LFI heavy atoms + 3 anchors x 5 stub atoms each = 33.
    assert model.numAtoms == 33

    # atom_map covers every atom in the model and uses the ModelAtom dataclass.
    atom_map = spec.model_atom_map
    assert set(atom_map.keys()) == {str(n) for n in model.name}
    assert all(isinstance(v, ModelAtom) for v in atom_map.values())
    n_stub = sum(1 for v in atom_map.values() if v.role == "stub")
    n_scaffold = sum(1 for v in atom_map.values() if v.role == "scaffold")
    assert n_scaffold == 18
    assert n_stub == 15  # 3 anchors x 5 atoms

    # Stub atoms must carry a canonical-FF type so the junction-frcmod splitter
    # can rewrite GAFF2 names. For Cys SG anchors we expect "S", "2C", "H1".
    stub_ff_types = {v.ff_type for v in atom_map.values() if v.role == "stub"}
    assert stub_ff_types == {"S", "2C", "H1"}
    # Scaffold atoms have no canonical-FF type (they keep their GAFF2 type).
    assert all(v.ff_type is None for v in atom_map.values() if v.role == "scaffold")


def _test_force_protonation_from_specs(tmp_path):
    mol = Molecule(QFZ_B_CIF)
    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )

    fp = forceProtonationFromSpecs(specs)
    assert len(fp) == 3
    for sel, variant in fp:
        assert variant == "CYX"
        # Each selection must resolve to one residue (>=1 atom is fine; PDB2PQR
        # collapses to residue identity).
        n = mol.atomselect(sel).sum()
        assert n > 0


def _test_custombonds_from_specs(tmp_path):
    mol = Molecule(QFZ_B_CIF)
    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )

    cb = custombondsFromSpecs(specs)
    assert len(cb) == 3
    for sel1, sel2 in cb:
        # Each side must point to exactly one atom in the molecule.
        n1 = mol.atomselect(sel1).sum()
        n2 = mol.atomselect(sel2).sum()
        assert n1 == 1
        assert n2 == 1


def _test_specs_without_writing_models():
    mol = Molecule(QFZ_B_CIF)
    specs = detectNonStandardResidues(mol, write_models=False, include_known=True)
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ScaffoldedPeptideSpec)
    assert spec.model_compound_cif is None
    assert spec.model_atom_map is None
    # Anchor records are still populated even without model-writing.
    assert len(spec.anchors) == 3


def _test_min_anchors_filter(tmp_path):
    """A scaffold with only 1 covalent bond to a canonical residue should be a
    cofactor, not a scaffolded peptide."""
    mol = Molecule(QFZ_B_CIF)
    # Drop two of the three CYS-LFI bonds so LFI only has one anchor.
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

    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )
    assert not any(isinstance(s, ScaffoldedPeptideSpec) for s in specs)
    assert any(isinstance(s, CofactorSpec) and s.resname == "LFI" for s in specs)
