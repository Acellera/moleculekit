import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_crystalpacking_asymmetric_unit():
    from os.path import join
    from moleculekit.molecule import Molecule, mol_equal
    from moleculekit.tools.crystalpacking import generateCrystalPacking

    mol = generateCrystalPacking("2hhb")
    refmol = Molecule(join(curr_dir, "test_crystalpacking", "2hhb_packing.pdb"))
    assert mol_equal(mol, refmol, fieldPrecision={"coords": 1e-3})


def _test_crystalpacking_biological_unit():
    from os.path import join
    from moleculekit.molecule import Molecule, mol_equal
    from moleculekit.tools.crystalpacking import generateCrystalPacking

    mol = generateCrystalPacking("1out")
    refmol = Molecule(join(curr_dir, "test_crystalpacking", "1out_packing.pdb"))
    assert mol_equal(mol, refmol, fieldPrecision={"coords": 1e-3})
