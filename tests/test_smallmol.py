import pytest
import os
import numpy as np
from moleculekit.smallmol.smallmolcdp import cdp_installed
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.molecule import Molecule
from glob import glob

curr_dir = os.path.dirname(os.path.abspath(__file__))

BENZAMIDINE_N_ATOMS = 18
BENZAMIDINE_BONDTYPES = [
    "ar",
    "ar",
    "1",
    "ar",
    "1",
    "ar",
    "1",
    "ar",
    "1",
    "ar",
    "1",
    "1",
    "2",
    "1",
    "1",
    "1",
    "1",
    "1",
]
BENZAMIDINE_BOND_ATOMS = [
    [0, 1],
    [0, 5],
    [0, 6],
    [1, 2],
    [1, 7],
    [2, 3],
    [2, 8],
    [3, 4],
    [3, 9],
    [4, 5],
    [4, 10],
    [5, 11],
    [6, 12],
    [6, 13],
    [12, 16],
    [12, 17],
    [13, 14],
    [13, 15],
]

LIGAND_N_ATOMS = 64

SMILE_SMI = "c1ccccc1O"
SMILE_N_ATOMS = 13


PHENOL_ELEMENT_IDX_1 = "C"
PHENOL_ELEMENT_NEIGHBORS_OX = [5, 12]
PHENOL_BTYPES_OX = [1, 1]

CHIRAL_SMI = "C[C@H](Cl)F"
CHIRAL_DETAILS = [("C1", "S")]

FOUNDBOND_SMI = "C=CN"


@pytest.mark.parametrize(
    "smiles",
    ((0, "Oc1c(cccc3)c3nc2ccncc12", 3), (1, "CN=c1nc[nH]cc1", 3), (2, "CC=CO", 2)),
)
def _test_tautomers(smiles):
    from moleculekit.util import file_diff
    from moleculekit.smallmol.smallmol import SmallMol
    import tempfile

    i, smiles, n_tauts = smiles
    mol = SmallMol(smiles, fixHs=False)
    tauts, scores = mol.getTautomers(canonical=False, genConformers=False)
    assert len(tauts) == n_tauts
    with tempfile.TemporaryDirectory() as tmpdir:
        newfile = os.path.join(tmpdir, "tautomers.sdf")
        tauts.writeSdf(os.path.join(tmpdir, "tautomers.sdf"))
        reffile = os.path.join(curr_dir, "test_smallmol", f"tautomer_results_{i}.sdf")
        file_diff(newfile, reffile)


@pytest.mark.skipif(not cdp_installed, reason="CDPKit not installed. Skipping test.")
def _test_smallmolcdp():
    from moleculekit.smallmol.smallmolcdp import SmallMolCDP

    sm = SmallMolCDP(os.path.join(curr_dir, "test_smallmol", "Imatinib.sdf"))
    sm.generateConformers(10)
    assert sm.coords.shape == (68, 3, 10)

    mol = sm.toMolecule()
    assert mol.coords.shape == (68, 3, 10)
    assert mol.bonds.shape == (72, 2)
    assert mol.bondtype.shape == (72,)
    assert np.sum(mol.bondtype == "2") == 13


def _test_loadMol2file():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    n_atoms = sm.numAtoms
    assert (
        n_atoms == BENZAMIDINE_N_ATOMS
    ), f"Atoms not correctly loaded. Expected: {BENZAMIDINE_N_ATOMS}; Now: {n_atoms}"


def _test_loadPdbfile():
    pdbfile = os.path.join(curr_dir, "test_smallmol", "ligand.pdb")
    sm = SmallMol(pdbfile)
    n_atoms = sm.numAtoms
    assert (
        n_atoms == LIGAND_N_ATOMS
    ), f"Atoms not correctly loaded. Expected: {LIGAND_N_ATOMS}; Now: {n_atoms}"


def _test_loadSmile():
    smi = SMILE_SMI
    sm = SmallMol(smi)
    n_atoms = sm.numAtoms
    assert (
        n_atoms == SMILE_N_ATOMS
    ), f"Atoms not correctly loaded. Expected: {SMILE_N_ATOMS}; Now: {n_atoms}"


def _test_getAtoms():
    smi = SMILE_SMI
    sm = SmallMol(smi)
    element_idx_1 = sm.get("element", "idx 1")[0]
    neighbors_element_O = sm.get("neighbors", "element O")[0]
    btypes_element_O = sm.get("bondtype", "element O", convertType=False)[0]

    assert (
        element_idx_1 == PHENOL_ELEMENT_IDX_1
    ), f"Element of the first atom does not correspond. Expect: {PHENOL_ELEMENT_IDX_1}; Now: {element_idx_1}"

    assert (
        neighbors_element_O == PHENOL_ELEMENT_NEIGHBORS_OX
    ), f"Neighbors atoms of the oxygen atom do not correspond. Expected: {PHENOL_ELEMENT_NEIGHBORS_OX}; Now: {neighbors_element_O}"

    assert btypes_element_O == PHENOL_BTYPES_OX, (
        "Bondtypes of the oxygen atom do not correspond:"
        "Expected: {}; Now: {}".format(btypes_element_O, PHENOL_BTYPES_OX)
    )


def _test_isChiral():
    smi = CHIRAL_SMI
    sm = SmallMol(smi, sanitize=True)
    ischiral, details = sm.isChiral(returnDetails=True)
    assert (
        details == CHIRAL_DETAILS
    ), f"chiral atom does not match.Expected: {CHIRAL_DETAILS}; Now: {details}"


def _test_foundBond():
    smi = FOUNDBOND_SMI
    sm = SmallMol(smi)
    isbond_0_N = sm.foundBondBetween("idx 0", "element N")
    isbond_0_1_single = sm.foundBondBetween("idx 0", "idx 1", bondtype=1)
    isbond_0_1_double, _ = sm.foundBondBetween("idx 0", "idx 1", bondtype=2)

    assert not isbond_0_N, "Bond between atom 0 and any nitrogens should not be present"
    assert not isbond_0_1_single, "Bond between atom 0 1 should not be single"
    assert isbond_0_1_double, "Bond between atom 0 1 should  be double"


def _test_generateConformers():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    current_conformer = sm.numFrames
    sm.generateConformers(num_confs=10, append=False)
    n_conformers = sm.numFrames

    assert (
        n_conformers >= current_conformer
    ), "The conformer generation should provide at least the same amount of conformers"


def _test_writeGenerateAndWriteConformers(tmp_path):
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    sm.generateConformers(num_confs=10, append=False)
    tmpfname = os.path.join(tmp_path, "benzamidine.sdf")
    tmpdir = os.path.dirname(tmpfname)
    sm.write(tmpfname, merge=False)
    direxists = os.path.isdir(tmpdir)
    n_files = len(glob(os.path.join(tmpdir, "*.sdf")))
    assert direxists, "The directory where to store the conformations was not created"
    assert n_files >= 1, "No conformations were written. At least one should be present"


def _test_removeGenerateConformer():
    molsmile = SMILE_SMI
    sm = SmallMol(molsmile)
    sm.generateConformers(num_confs=10, append=False)
    n_confs = sm.numFrames
    sm.dropFrames([0])
    n_confs_del = sm.numFrames
    sm.dropFrames()
    n_confs_zero = sm.numFrames

    assert (
        n_confs_del == n_confs - 1
    ), "The number of conformations after the deletion was not reduced by exactly one unit"

    assert (
        n_confs_zero == 0
    ), "The number of conformations after the deletion was not reduced to 0"


def _test_convertToMolecule():
    from moleculekit.molecule import mol_equal

    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    mol = sm.toMolecule()
    assert mol_equal(sm, mol, exceptFields=["serial", "name"], _logger=False)


@pytest.mark.parametrize("molfile", ["BEN_model.sdf", "BEN_pH7_4.sdf", "indole.mol2"])
def _test_convertFromMolecule(molfile):
    from moleculekit.molecule import mol_equal

    mol = Molecule(os.path.join(curr_dir, "test_smallmol", molfile))
    mol.resid[:] = 1
    sm = SmallMol(mol)
    assert mol_equal(sm, mol, exceptFields=["serial", "name"], _logger=False)


def _test_getBonds():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))

    assert (
        sm._bonds.tolist() == BENZAMIDINE_BOND_ATOMS
    ), "The atoms in bonds are not the same of the reference"

    assert (
        sm._bondtype.tolist() == BENZAMIDINE_BONDTYPES
    ), "The bonds type are not the same of the reference"


def _test_repr():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    _ = str(sm)


def _test_toSMILES():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    assert (
        sm.toSMILES() == "NC(=[NH2+])C1=CC=CC=C1"
    ), f"Failed with SMILES: {sm.toSMILES()}"


def _test_toSMARTS():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    assert (
        sm.toSMARTS() == "[#6]1(:[#6]:[#6]:[#6]:[#6]:[#6]:1)-[#6](=[#7+])-[#7]"
    ), f"Failed with SMARTS: {sm.toSMARTS()}"


def _test_align():
    from moleculekit.util import rotationMatrix
    import numpy as np

    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    mol = sm.toMolecule()
    mol.rotateBy(rotationMatrix([1, 0, 0], 3.14))
    sm.align(mol)

    assert (
        np.abs(sm._coords) - np.abs(mol.coords)
    ).max()  # I need to do the abs of the coords since it's a symmetrical molecule


def _test_copy():
    sm = SmallMol(os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"))
    sm_copy = sm.copy()
    coords = sm.get("coords")
    coords_copy = sm_copy.get("coords")
    assert np.array_equal(coords, coords_copy)

    # Ensure no hydrogens are added in the copy method
    sm = SmallMol(
        os.path.join(curr_dir, "test_smallmol", "benzamidine.sdf"),
        removeHs=True,
        fixHs=False,
    )
    sm_copy = sm.copy()
    coords = sm.get("coords")
    coords_copy = sm_copy.get("coords")
    assert np.array_equal(coords, coords_copy)


# def _test_getRCSBLigandByLigname():
#     from moleculekit.smallmol.util import getRCSBLigandByLigname
#     sm = getRCSBLigandByLigname('BEN')
