import os
import pytest
from moleculekit.molecule import Molecule, mol_equal

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_templateResidueFromSmiles():
    def _cmp(mol, ref_mol):
        assert mol_equal(
            mol,
            ref_mol,
            checkFields=Molecule._all_fields,
            exceptFields=["crystalinfo", "fileloc"],
            uqBonds=True,
            fieldPrecision={"coords": 1e-3},
        )

    testdir = os.path.join(curr_dir, "test_molecule", "test_templating")
    start_file = os.path.join(testdir, "BEN.pdb")

    sel = "resname BEN"
    smiles = "[NH2+]=C(N)c1ccccc1"
    ref_mol = Molecule(os.path.join(testdir, "BEN_pH7.4.cif"))

    mol = Molecule(start_file)
    mol.templateResidueFromSmiles(sel, smiles, addHs=False, guessBonds=True)
    _cmp(mol, ref_mol.copy(sel="noh"))

    # With addHs=True
    mol = Molecule(start_file)
    mol.templateResidueFromSmiles(sel, smiles, addHs=True, guessBonds=True)
    _cmp(mol, ref_mol)

    # Template inside the original complex
    mol = Molecule("3ptb")
    mol.templateResidueFromSmiles(sel, smiles, addHs=True, guessBonds=True)
    _cmp(mol, Molecule(os.path.join(testdir, "3PTB_BEN_pH7.4.cif")))

    # With RCSB incorrect SMILES
    smiles = "NC(=N)c1ccccc1"
    ref_mol = Molecule(os.path.join(testdir, "BEN_RCSB.cif"))

    mol = Molecule(start_file)
    mol.templateResidueFromSmiles(sel, smiles, addHs=False, guessBonds=True)
    _cmp(mol, ref_mol.copy(sel="noh"))

    # With addHs=True
    mol = Molecule(start_file)
    mol.templateResidueFromSmiles(sel, smiles, addHs=True, guessBonds=True)
    _cmp(mol, ref_mol)

    # Test 1STP BTN Biotin molecule templating
    start_file = os.path.join(testdir, "1STP_BTN.pdb")
    sel = "resname BTN"
    smiles = "C1[C@H]2[C@@H]([C@@H](S1)CCCCC(=O)O)NC(=O)N2"
    ref_mol = Molecule(os.path.join(testdir, "1STP_BTN.cif"))

    mol = Molecule(start_file)
    mol.templateResidueFromSmiles(sel, smiles, addHs=True, guessBonds=True)
    _cmp(mol, ref_mol)


@pytest.mark.parametrize("file", ("1STP_BTN.cif", "BEN_pH7.4.cif"))
def _test_toRDKitMol(file):
    testdir = os.path.join(curr_dir, "test_molecule", "test_templating")
    mol = Molecule(os.path.join(testdir, file))
    rmol = mol.toRDKitMol()

    # Roundtrip conversion test
    mol2 = Molecule.fromRDKitMol(rmol)
    assert mol_equal(
        mol,
        mol2,
        checkFields=Molecule._all_fields,
        exceptFields=["crystalinfo", "fileloc", "chain", "segid", "resid"],
        uqBonds=True,
    )


def _test_templatingNonStandardResidues():
    test_home = os.path.join(
        curr_dir, "test_systemprepare", "test-nonstandard-residues"
    )
    mol = Molecule(os.path.join(test_home, "1A4W", "1A4W.pdb"))
    mol.templateResidueFromSmiles(
        "resname TYS",
        "c1cc(ccc1C[C@@H](CO)N)OS(=O)(=O)O",
        addHs=True,
        guessBonds=True,
        onlyOnAtoms="not backbone",
    )

    refmol = Molecule(
        os.path.join(curr_dir, "test_molecule", "test_templating", "TYS_templated.cif")
    )
    assert mol_equal(
        mol.copy(sel="resname TYS"),
        refmol,
        checkFields=Molecule._all_fields,
        exceptFields=["crystalinfo", "fileloc", "serial"],
        uqBonds=True,
        fieldPrecision={"coords": 1e-3},
    )


def _test_extend_residue_from_smiles():
    from moleculekit.molecule import Molecule
    import numpy as np

    mol = Molecule("3ptb")
    mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    mol.remove("resname BEN and name H6")
    mol.extendResidueFromSmiles(
        sel="resname BEN",
        extension_smiles="*C(C)(C)C",
        target_atom_sel="resname BEN and name N1",
    )

    ben = mol.copy(sel="resname BEN")
    ref_bondtype = [
        "2",
        "1",
        "1",
        "1",
        "1",
        "2",
        "1",
        "1",
        "1",
        "2",
        "1",
        "1",
        "2",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
    ]
    assert ben.numAtoms == 30
    assert ben.numBonds == 30
    assert np.all(ben.bondtype == ref_bondtype)

    mol2 = Molecule("3ptb")
    mol2.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    mol2.extendResidueFromSmiles(
        sel="resname BEN",
        extension_smiles="*C(C)(C)C",
        target_atom_sel="resname BEN and name H6",
    )
    assert mol_equal(mol, mol2, checkFields=Molecule._all_fields)


def _test_extend_residue_from_smiles_double_bond():
    from moleculekit.molecule import Molecule
    import numpy as np

    mol = Molecule("3ptb")
    mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    mol.remove("resname BEN and name H6 H7")
    mol.extendResidueFromSmiles(
        sel="resname BEN",
        extension_smiles="*=O",
        target_atom_sel="resname BEN and name N1",
    )

    ben = mol.copy(sel="resname BEN")
    ref_bondtype = [
        "2",
        "1",
        "1",
        "1",
        "1",
        "2",
        "1",
        "1",
        "1",
        "2",
        "1",
        "1",
        "2",
        "1",
        "1",
        "1",
        "2",
    ]
    assert ben.numAtoms == 17
    assert ben.numBonds == 17
    assert np.all(ben.bondtype == ref_bondtype)
