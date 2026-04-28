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
    NCAASpec,
    CovalentLigandSpec,
    LigandSpec,
    CrosslinkSpec,
    ModelAtom,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(curr_dir, "test_nonstandard_residues")
QFZ_B_CIF = os.path.join(DATA_DIR, "8QFZ_B.cif")
QU4_A_CIF = os.path.join(DATA_DIR, "8QU4_A.cif")
VBL_PDB = os.path.join(curr_dir, "pdb", "5vbl.pdb")
R1J_PDB = os.path.join(curr_dir, "pdb", "1r1j.pdb")


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
    """A scaffold with only 1 covalent bond to a canonical residue falls below
    the scaffolded-peptide threshold and should classify as
    :class:`CovalentLigandSpec`."""
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
    lfi_specs = [s for s in specs if s.resname == "LFI"]
    assert len(lfi_specs) == 1
    assert isinstance(lfi_specs[0], CovalentLigandSpec)
    assert lfi_specs[0].anchor.anchor_atom.resname == "CYS"
    assert lfi_specs[0].anchor.anchor_atom.name == "SG"


def _test_5vbl_ncaas_and_free_ligand(tmp_path):
    """5VBL: chain A is a peptide inhibitor with five non-canonical amino
    acids in the polymer chain (HRG, ALC, OIC, NLE, 200), and chain B
    carries OLC (oleic acid) as a free, non-covalently bound ligand."""
    mol = Molecule(VBL_PDB)
    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )

    ncaas = [s for s in specs if isinstance(s, NCAASpec)]
    ligands = [s for s in specs if isinstance(s, LigandSpec)]
    assert not any(isinstance(s, ScaffoldedPeptideSpec) for s in specs)
    assert not any(isinstance(s, CovalentLigandSpec) for s in specs)

    # NCAAs: chain-embedded non-canonical amino acids.
    assert sorted(s.resname for s in ncaas) == ["200", "ALC", "HRG", "NLE", "OIC"]
    for s in ncaas:
        assert s.head_atom == "N" and s.tail_atom == "C"
        # 200 is the C-terminal NCAA (resid 17): no C-tail bond.
        if s.resname == "200":
            assert s.is_c_term and not s.is_n_term
        else:
            assert not s.is_n_term and not s.is_c_term
        # Each NCAA writes a bare-residue model CIF.
        m = Molecule(s.model_compound_cif)
        assert m.numAtoms > 0
        assert all(rn == s.resname for rn in m.resname)

    # Free ligand: OLC, no covalent bonds.
    assert len(ligands) == 1
    assert ligands[0].resname == "OLC" and ligands[0].residue.chain == "B"
    assert os.path.isfile(ligands[0].model_compound_cif)

    # Helpers must not emit anything for NCAAs or free ligands.
    assert forceProtonationFromSpecs(specs) == []
    assert custombondsFromSpecs(specs) == []


def _test_1r1j_covalent_glycosylation(tmp_path):
    """1R1J is a glycoprotein with three NAG residues each covalently
    attached to a different Asn ND2 (N-glycosylation), plus a free OIR
    ligand. This exercises :class:`CovalentLigandSpec` and the
    ``custombondsFromSpecs`` helper for single-anchor covalent bonds."""
    mol = Molecule(R1J_PDB)
    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )

    nags = [s for s in specs if isinstance(s, CovalentLigandSpec) and s.resname == "NAG"]
    free = [s for s in specs if isinstance(s, LigandSpec)]
    assert not any(isinstance(s, ScaffoldedPeptideSpec) for s in specs)
    assert not any(isinstance(s, NCAASpec) for s in specs)

    # Three NAGs, each bonded ASN.ND2 <-> NAG.C1.
    assert len(nags) == 3
    asn_resids = sorted(int(s.anchor.anchor_atom.resid) for s in nags)
    assert asn_resids == [144, 324, 627]
    for s in nags:
        assert s.anchor.anchor_atom.resname == "ASN"
        assert s.anchor.anchor_atom.name == "ND2"
        assert s.anchor.scaffold_atom.name == "C1"
        # Round-trip the UniqueAtomIDs against the input molecule.
        anchor_idx = int(s.anchor.anchor_atom.selectAtom(mol))
        scaffold_idx = int(s.anchor.scaffold_atom.selectAtom(mol))
        assert mol.name[anchor_idx] == "ND2"
        assert mol.name[scaffold_idx] == "C1"
        assert mol.resname[scaffold_idx] == "NAG"

    # Free ligand: OIR.
    assert len(free) == 1
    assert free[0].resname == "OIR"

    # ANCHOR_VARIANTS has an (ASN, ND2) -> NLN entry for N-glycosylation;
    # forceProtonationFromSpecs emits one rename per glycosylated Asn.
    fp = forceProtonationFromSpecs(specs)
    assert len(fp) == 3
    assert all(variant == "NLN" for _, variant in fp)

    # Three custombonds must be emitted: one per NAG-Asn glycosidic bond.
    cb = custombondsFromSpecs(specs)
    assert len(cb) == 3
    for sel1, sel2 in cb:
        assert mol.atomselect(sel1).sum() == 1
        assert mol.atomselect(sel2).sum() == 1


def _test_8qu4_stapled_peptide(tmp_path):
    """Pattern-B stapled peptide: PDB 8QU4 chain A is a 13-mer NF-Y-derived
    peptide with an i, i+4 hydrocarbon staple between two non-canonical amino
    acids (NLE at resid 272 and MK8 at resid 276), capped with ACE and NH2.
    The staple bond NLE.CE - MK8.CE is the only non-peptide inter-residue
    bond and must be reported as a :class:`CrosslinkSpec`. The ACE
    and NH2 caps have AMBER parameters bundled and are not flagged."""
    mol = Molecule(QU4_A_CIF)
    specs = detectNonStandardResidues(
        mol, outdir=str(tmp_path), write_models=True, include_known=True
    )

    # Exactly two NCAAs (the stapled residues) and one crosslink. ACE / NH2
    # caps must not be flagged.
    ncaas = [s for s in specs if isinstance(s, NCAASpec)]
    crosslinks = [s for s in specs if isinstance(s, CrosslinkSpec)]
    assert sorted(s.resname for s in ncaas) == ["MK8", "NLE"]
    assert len(crosslinks) == 1
    assert len(specs) == 3, [type(s).__name__ for s in specs]

    # Crosslink endpoints: NLE272.CE <-> MK8276.CE.
    cl = crosslinks[0]
    pair_resnames = {cl.atom_a.resname, cl.atom_b.resname}
    pair_resids = {int(cl.atom_a.resid), int(cl.atom_b.resid)}
    assert pair_resnames == {"NLE", "MK8"}
    assert pair_resids == {272, 276}
    assert cl.atom_a.name == "CE" and cl.atom_b.name == "CE"

    # UniqueAtomIDs round-trip against the input.
    for uid in (cl.atom_a, cl.atom_b):
        idx = int(uid.selectAtom(mol))
        assert mol.name[idx] == "CE"
        assert int(mol.resid[idx]) == int(uid.resid)

    # custombondsFromSpecs must emit exactly one entry for the staple bond
    # (no other anchor-bearing specs in this fixture).
    cb = custombondsFromSpecs(specs)
    assert len(cb) == 1
    sel1, sel2 = cb[0]
    assert mol.atomselect(sel1).sum() == 1
    assert mol.atomselect(sel2).sum() == 1

    # No scaffolded peptide, no covalent ligand, no free ligand for this fixture.
    assert not any(isinstance(s, ScaffoldedPeptideSpec) for s in specs)
    assert not any(isinstance(s, CovalentLigandSpec) for s in specs)
    assert not any(isinstance(s, LigandSpec) for s in specs)
