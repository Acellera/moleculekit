from moleculekit.molecule import Molecule, mol_equal
from glob import glob
import numpy as np
import pytest
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_pdb():
    for f in glob(os.path.join(curr_dir, "test_readers", "*.pdb")):
        _ = Molecule(f)
    for f in glob(os.path.join(curr_dir, "pdb", "*.pdb")):
        _ = Molecule(f)


def _test_pdbqt():
    _ = Molecule(os.path.join(curr_dir, "test_readers", "3ptb.pdbqt"))


def _test_psf():
    _ = Molecule(os.path.join(curr_dir, "test_readers", "4RWS", "structure.psf"))


def _test_xtc():
    _ = Molecule(os.path.join(curr_dir, "test_readers", "4RWS", "traj.xtc"))


def _test_combine_topo_traj():
    testfolder = os.path.join(curr_dir, "test_readers", "4RWS")
    mol = Molecule(os.path.join(testfolder, "structure.psf"))
    mol.read(os.path.join(testfolder, "traj.xtc"))


def _test_prmtop():
    testfolder = os.path.join(curr_dir, "test_readers", "3AM6")
    _ = Molecule(os.path.join(testfolder, "structure.prmtop"))


def _test_crd():
    testfolder = os.path.join(curr_dir, "test_readers", "3AM6")
    _ = Molecule(os.path.join(testfolder, "structure.crd"))


def _test_mol2():
    testfolder = os.path.join(curr_dir, "test_readers", "3L5E")
    _ = Molecule(os.path.join(testfolder, "protein.mol2"))
    _ = Molecule(os.path.join(testfolder, "ligand.mol2"))


def _test_sdf():
    import pickle

    sdf_file = os.path.join(curr_dir, "test_readers", "benzamidine-3PTB-pH7.sdf")
    ref_file = os.path.join(curr_dir, "test_readers", "benzamidine-3PTB-pH7.pkl")
    with open(ref_file, "rb") as f:
        ref_data = pickle.load(f)

    mol = Molecule(sdf_file)
    assert mol.numAtoms == 18
    assert np.all(mol.name == mol.element)
    assert np.array_equal(mol.element, ref_data["element"])
    assert np.array_equal(mol.coords, ref_data["coords"])
    assert np.array_equal(mol.formalcharge, ref_data["formalcharge"])
    assert np.array_equal(mol.bonds, ref_data["bonds"])
    assert np.array_equal(mol.bondtype, ref_data["bondtype"])

    sdf_file = os.path.join(curr_dir, "test_readers", "S98_ideal.sdf")
    ref_file = os.path.join(curr_dir, "test_readers", "S98_ideal.pkl")
    with open(ref_file, "rb") as f:
        ref_data = pickle.load(f)

    mol = Molecule(sdf_file)
    assert mol.numAtoms == 34
    assert np.all(mol.name == mol.element)
    assert np.array_equal(mol.element, ref_data["element"])
    assert np.array_equal(mol.coords, ref_data["coords"])
    assert np.array_equal(mol.formalcharge, ref_data["formalcharge"])
    assert np.array_equal(mol.bonds, ref_data["bonds"])
    assert np.array_equal(mol.bondtype, ref_data["bondtype"])


def _test_gjf():
    testfolder = os.path.join(curr_dir, "test_readers", "3L5E")
    _ = Molecule(os.path.join(testfolder, "ligand.gjf"))


def _test_xyz():
    testfolder = os.path.join(curr_dir, "test_readers", "3L5E")
    _ = Molecule(os.path.join(testfolder, "ligand.xyz"))


def _test_MDTRAJTOPOread():
    testfolder = os.path.join(curr_dir, "test_readers", "3L5E")
    _ = Molecule(os.path.join(testfolder, "ligand.mol2"))


def _test_mae():
    for f in glob(os.path.join(curr_dir, "test_readers", "*.mae")):
        _ = Molecule(f)


def _test_append_trajectories():
    testfolder = os.path.join(curr_dir, "test_readers", "CMYBKIX")
    mol = Molecule(os.path.join(testfolder, "filtered.pdb"))
    mol.read(glob(os.path.join(testfolder, "*.xtc")))


def _test_missing_crystal_info():
    _ = Molecule(os.path.join(curr_dir, "test_readers", "weird-cryst.pdb"))


def _test_missing_occu_beta():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "opm_missing_occu_beta.pdb"))
    assert np.all(mol.occupancy[:2141] != 0)
    assert np.all(mol.occupancy[2141:] == 0)
    assert np.all(mol.beta[:2141] != 0)
    assert np.all(mol.beta[2141:] == 0)


def _test_dcd():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "dcd", "1kdx_0.pdb"))
    mol.read(os.path.join(curr_dir, "test_readers", "dcd", "1kdx.dcd"))
    assert mol.coords.shape == (1809, 3, 17)


def _test_dcd_into_prmtop():
    mol = Molecule(
        os.path.join(curr_dir, "test_readers", "dialanine", "structure.prmtop")
    )
    mol.read(os.path.join(curr_dir, "test_readers", "dialanine", "traj.dcd"))
    assert mol.numFrames == 2


def _test_dcd_frames():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "dcd", "1kdx_0.pdb"))
    mol.read(os.path.join(curr_dir, "test_readers", "dcd", "1kdx.dcd"))
    tmpcoo = mol.coords.copy()
    mol.read([os.path.join(curr_dir, "test_readers", "dcd", "1kdx.dcd")], frames=[8])
    assert np.array_equal(
        tmpcoo[:, :, 8], np.squeeze(mol.coords)
    ), "Specific frame reading not working"


def _test_xtc_frames():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "4RWS", "structure.pdb"))
    mol.read(os.path.join(curr_dir, "test_readers", "4RWS", "traj.xtc"))
    tmpcoo = mol.coords.copy()
    mol.read([os.path.join(curr_dir, "test_readers", "4RWS", "traj.xtc")], frames=[1])
    assert np.array_equal(
        tmpcoo[:, :, 1], np.squeeze(mol.coords)
    ), "Specific frame reading not working"


def _test_acemd3_xtc_fstep():
    from moleculekit.util import tempname

    mol = Molecule(
        os.path.join(curr_dir, "test_readers", "aladipep_traj_4fs_100ps.xtc")
    )
    refstep = np.array(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
        ]
    )
    reftime = np.array(
        [
            40.0,
            80.0,
            120.0,
            160.0,
            200.0,
            240.0,
            280.0,
            320.0,
            360.0,
            400.0,
            440.0,
            480.0,
            520.0,
            560.0,
            600.0,
            640.0,
            680.0,
            720.0,
            760.0,
            800.0,
        ],
        dtype=np.float32,
    )
    assert np.array_equal(mol.step, refstep)
    assert np.allclose(mol.time, reftime)
    assert abs(mol.fstep - 4e-5) < 1e-5

    # Test that XTC writing doesn't mess up the times
    tmpfile = tempname(suffix=".xtc")
    mol.write(tmpfile)

    mol2 = Molecule(tmpfile)
    assert mol.fstep == mol2.fstep
    assert np.array_equal(mol.time, mol2.time)


def _test_gromacs_top():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "gromacs.top"))
    assert np.array_equal(mol.name, ["C1", "O2", "N3", "H4", "H5", "N6", "H7", "H8"])
    assert np.array_equal(mol.atomtype, ["C", "O", "NT", "H", "H", "NT", "H", "H"])
    assert np.all(mol.resid == 1)
    assert np.all(mol.resname == "UREA")
    assert np.array_equal(
        mol.charge,
        np.array(
            [0.683, -0.683, -0.622, 0.346, 0.276, -0.622, 0.346, 0.276], np.float32
        ),
    )
    assert np.array_equal(
        mol.bonds, [[2, 3], [2, 4], [5, 6], [5, 7], [0, 1], [0, 2], [0, 5]]
    )
    assert np.array_equal(
        mol.angles,
        [
            [0, 2, 3],
            [0, 2, 4],
            [3, 2, 4],
            [0, 5, 6],
            [0, 5, 7],
            [6, 5, 7],
            [1, 0, 2],
            [1, 0, 5],
            [2, 0, 5],
        ],
    )
    assert np.array_equal(
        mol.dihedrals,
        [
            [1, 0, 2, 3],
            [5, 0, 2, 3],
            [1, 0, 2, 4],
            [5, 0, 2, 4],
            [1, 0, 5, 6],
            [2, 0, 5, 6],
            [1, 0, 5, 7],
            [2, 0, 5, 7],
            [2, 3, 4, 0],
            [5, 6, 7, 0],
            [0, 2, 5, 1],
        ],
    )
    assert len(mol.impropers) == 0


def _test_mmcif_single_frame():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "1ffk.cif"))
    assert mol.numAtoms == 64281
    assert mol.numFrames == 1


def _test_mmcif_multi_frame():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "1j8k.cif"))
    assert mol.numAtoms == 1402
    assert mol.numFrames == 20


def _test_mmcif_ligand():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "URF.cif"))
    assert mol.numAtoms == 12, mol.numAtoms
    assert mol.numFrames == 1

    mol = Molecule(os.path.join(curr_dir, "test_readers", "BEN.cif"))
    assert mol.numAtoms == 17, mol.numAtoms
    assert mol.numFrames == 1

    mol = Molecule(os.path.join(curr_dir, "test_readers", "33X.cif"))
    assert mol.numAtoms == 16, mol.numAtoms
    assert mol.numFrames == 1


def _test_multiple_file_fileloc():
    mol = Molecule(
        os.path.join(curr_dir, "test_readers", "multi-traj", "structure.pdb")
    )
    mol.read(
        glob(os.path.join(curr_dir, "test_readers", "multi-traj", "data", "*", "*.xtc"))
    )
    # Try to vstack fileloc. This will fail with wrong fileloc shape
    fileloc = np.vstack(mol.fileloc)
    assert fileloc.shape == (12, 2)
    print("Correct fileloc shape with multiple file reading.")


def _test_topo_overwriting():
    # Testing overwriting of topology fields
    mol = Molecule(os.path.join(curr_dir, "test_readers", "multi-topo", "mol.psf"))
    atomtypes = mol.atomtype.copy()
    charges = mol.charge.copy()
    coords = np.array(
        [
            [[0.0], [0.0], [-0.17]],
            [[0.007], [1.21], [0.523]],
            [[0.0], [0.0], [-1.643]],
            [[-0.741], [-0.864], [-2.296]],
        ],
        dtype=np.float32,
    )

    mol.read(os.path.join(curr_dir, "test_readers", "multi-topo", "mol.pdb"))
    assert np.array_equal(mol.atomtype, atomtypes)
    assert np.array_equal(mol.charge, charges)
    assert np.array_equal(mol.coords, coords)
    print("Merging of topology fields works")


def _test_integer_resnames():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "errors.pdb"))
    assert np.unique(mol.resname) == "007"


def _test_star_indexes():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "errors.pdb"))
    assert np.all(mol.serial == np.arange(1, mol.numAtoms + 1))


def _test_xsc():
    mol1 = Molecule(os.path.join(curr_dir, "test_readers", "test1.xsc"))
    ref1box = np.array([[81.67313], [75.81903], [75.02757]], dtype=np.float32)
    ref1step = np.array([2])
    assert np.array_equal(mol1.box, ref1box)
    assert np.array_equal(mol1.step, ref1step)

    mol2 = Molecule(os.path.join(curr_dir, "test_readers", "test2.xsc"))
    ref2box = np.array([[41.0926], [35.99448], [63.51968]], dtype=np.float32)
    ref2step = np.array([18])
    assert np.array_equal(mol2.box, ref2box)
    assert np.array_equal(mol2.step, ref2step)

    # Test reading xsc into existing molecule
    mol1 = Molecule("3ptb")
    mol1.read(os.path.join(curr_dir, "test_readers", "test1.xsc"))
    assert np.array_equal(mol1.box, ref1box)
    assert np.array_equal(mol1.step, ref1step)


def _test_pdb_element_guessing():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "errors.pdb"))
    refelem = np.array(
        [
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "N",
            "Na",
            "N",
            "H",
            "H",
            "C",
            "Cl",
            "Ca",
        ],
        dtype=object,
    )
    assert np.array_equal(mol.element, refelem)

    mol = Molecule(os.path.join(curr_dir, "test_readers", "dialanine_solute.pdb"))
    refelem = np.array(
        [
            "H",
            "C",
            "H",
            "H",
            "C",
            "O",
            "N",
            "H",
            "C",
            "H",
            "C",
            "H",
            "H",
            "H",
            "C",
            "O",
            "N",
            "H",
            "C",
            "H",
            "H",
            "H",
        ],
        dtype=object,
    )
    assert np.array_equal(mol.element, refelem)

    mol = Molecule(os.path.join(curr_dir, "test_readers", "cl_na_element.pdb"))
    refelem = np.array(["Cl", "Na"], dtype=object)
    assert np.array_equal(mol.element, refelem)

    mol = Molecule(os.path.join(curr_dir, "test_readers", "dummy_atoms.mol2"))
    refelem = np.array(
        [
            "Cd",
            "Co",
            "Ca",
            "Xe",
            "Rb",
            "Lu",
            "Ga",
            "Ba",
            "Cs",
            "Ho",
            "Pb",
            "Sr",
            "Yb",
            "Y",
        ],
        dtype=object,
    )
    assert np.array_equal(
        mol.element, refelem
    ), f"Failed guessing {refelem}, got {mol.element}"


def _test_pdb_charges():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "errors.pdb"))
    refcharge = np.array(
        [-1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32
    )
    assert np.array_equal(mol.formalcharge, refcharge)


def _test_prepi():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "benzamidine.prepi"))
    assert np.array_equal(
        mol.name,
        [
            "N",
            "H18",
            "H17",
            "C7",
            "N14",
            "H15",
            "H16",
            "C",
            "C6",
            "H12",
            "C5",
            "H11",
            "C4",
            "H10",
            "C3",
            "H9",
            "C2",
            "H",
        ],
    )
    assert np.array_equal(
        mol.atomtype,
        [
            "NT",
            "H",
            "H",
            "CM",
            "NT",
            "H",
            "H",
            "CA",
            "CA",
            "HA",
            "CA",
            "HA",
            "CA",
            "HA",
            "CA",
            "HA",
            "CA",
            "HA",
        ],
    )
    assert np.array_equal(
        mol.charge,
        np.array(
            [
                -0.4691,
                0.3267,
                0.3267,
                0.5224,
                -0.4532,
                0.3267,
                0.3267,
                -0.2619,
                -0.082,
                0.149,
                -0.116,
                0.17,
                -0.058,
                0.17,
                -0.116,
                0.17,
                -0.082,
                0.149,
            ],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        mol.impropers,
        np.array(
            [
                [7, 0, 3, 4],
                [8, 16, 7, 3],
                [7, 10, 8, 9],
                [8, 12, 10, 11],
                [10, 14, 12, 13],
                [12, 16, 14, 15],
                [7, 14, 16, 17],
            ]
        ),
    )


def _test_rtf():
    mol = Molecule(os.path.join(curr_dir, "test_readers", "mol.rtf"))
    assert np.array_equal(
        mol.element,
        [
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "N",
            "N",
            "H",
            "H",
            "H",
            "H",
        ],
    )
    assert np.array_equal(
        mol.name,
        [
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "H8",
            "H9",
            "H10",
            "H11",
            "H12",
            "N13",
            "N14",
            "H15",
            "H16",
            "H17",
            "H18",
        ],
    )
    assert np.array_equal(
        mol.atomtype,
        [
            "C261",
            "C261",
            "C261",
            "C261",
            "C261",
            "C261",
            "C2N2",
            "HG61",
            "HG61",
            "HG61",
            "HG61",
            "HG61",
            "N2P1",
            "N2P1",
            "HGP2",
            "HGP2",
            "HGP2",
            "HGP2",
        ],
    )
    assert np.array_equal(
        mol.charge,
        np.array(
            [
                0.062728,
                -0.046778,
                -0.061366,
                -0.062216,
                -0.061366,
                -0.046778,
                0.270171,
                0.063213,
                0.062295,
                0.062269,
                0.062295,
                0.063213,
                -0.28704,
                -0.28704,
                0.3016,
                0.3016,
                0.3016,
                0.3016,
            ],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        mol.masses,
        np.array(
            [
                12.011,
                12.011,
                12.011,
                12.011,
                12.011,
                12.011,
                12.011,
                1.008,
                1.008,
                1.008,
                1.008,
                1.008,
                14.007,
                14.007,
                1.008,
                1.008,
                1.008,
                1.008,
            ],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        mol.bonds,
        np.array(
            [
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
                [16, 12],
                [17, 12],
                [14, 13],
                [15, 13],
            ]
        ),
    )
    assert np.array_equal(mol.impropers, np.array([[6, 0, 12, 13]]))


def _test_sdf2():
    from moleculekit.readers import sdf_generator

    sdf_file = os.path.join(curr_dir, "test_readers", "fda_drugs_light.sdf")
    mol = Molecule(sdf_file)
    assert mol.numAtoms == 41
    mol = Molecule(sdf_file, mol_idx=99)
    assert mol.numAtoms == 13

    with pytest.raises(RuntimeError):
        Molecule(filename=sdf_file, mol_idx=100)

    gen = sdf_generator(sdf_file)
    ref_n_atoms = [41, 12, 33, 41]
    k = 0
    for mol in gen:
        if k >= len(ref_n_atoms):
            break
        assert mol.numAtoms == ref_n_atoms[k]
        k += 1


def _test_broken_pdbs():
    from glob import glob

    for ff in glob(os.path.join(curr_dir, "test_readers", "broken-pdbs") + "/*.pdb"):
        mol = Molecule(ff)
        assert mol.numAtoms > 0


def _test_netcdf():
    from glob import glob

    for ff in glob(os.path.join(curr_dir, "test_readers", "netcdf") + "/*.nc"):
        mol = Molecule(ff)
        assert mol.numAtoms > 0 and mol.numFrames > 0


def _test_trr():
    from glob import glob

    for ff in glob(os.path.join(curr_dir, "test_readers", "trr") + "/*.trr"):
        mol = Molecule(ff)
        assert mol.coords.shape == (22, 3, 501)


def _test_binpos():
    from glob import glob

    for ff in glob(os.path.join(curr_dir, "test_readers", "trr") + "/*.binpos"):
        mol = Molecule(ff)
        assert mol.coords.shape == (2269, 3, 1)


@pytest.mark.parametrize("pdbid", ["3ptb", "3hyd", "6a5j", "5vbl", "7q5b"])
def _test_bcif_cif(pdbid):
    ciffile = os.path.join(curr_dir, "pdb", f"{pdbid.lower()}.cif")
    bciffile = os.path.join(curr_dir, "pdb", f"{pdbid.lower()}.bcif.gz")
    mol1 = Molecule(ciffile)
    mol2 = Molecule(bciffile)
    assert mol_equal(
        mol1,
        mol2,
        checkFields=Molecule._all_fields,
        exceptFields=["fileloc"],
    )


@pytest.mark.parametrize("pdbid", ["3ptb", "3hyd", "6a5j", "5vbl", "7q5b"])
def _test_bcif_pdb(pdbid):
    from moleculekit.molecule import calculateUniqueBonds

    ciffile = os.path.join(curr_dir, "pdb", f"{pdbid.lower()}.bcif.gz")
    pdbfile = os.path.join(curr_dir, "pdb", f"{pdbid.lower()}.pdb")
    mol1 = Molecule(pdbfile)
    mol2 = Molecule(ciffile, covalentonly=False)

    # Comparing just the intersection of bonds of mol1 and 2 because PDB has much fewer
    keep_bonds = []
    mol1bonds = [tuple(x) for x in mol1.bonds]
    for i, bb in enumerate(mol2.bonds):
        if tuple(bb) in mol1bonds or tuple(bb[::-1]) in mol1bonds:
            keep_bonds.append(i)
    if len(keep_bonds):
        keep_bonds = np.array(keep_bonds)
        mol2.bonds = mol2.bonds[keep_bonds]
        mol2.bondtype = mol2.bondtype[keep_bonds]
        mol2.bonds, mol2.bondtype = calculateUniqueBonds(mol2.bonds, mol2.bondtype)
    else:
        mol2.bonds = np.zeros((0, 2), dtype=np.uint32)
        mol2.bondtype = np.zeros((0,), dtype=object)

    extra_except = []
    if pdbid == "6a5j":
        # The CIF file is missing crystal info
        extra_except = ["box", "boxangles"]

    assert mol_equal(
        mol1,
        mol2,
        checkFields=Molecule._all_fields,
        exceptFields=[
            "fileloc",
            "crystalinfo",
            "segid",
            "serial",
            "bondtype",
        ]
        + extra_except,
    )


def _test_inpcrd():
    import tempfile

    mol = Molecule(
        os.path.join(curr_dir, "test_readers", "dialanine", "structure.prmtop")
    )
    mol.read(os.path.join(curr_dir, "test_readers", "dialanine", "traj.dcd"))
    mol.dropFrames(keep=mol.numFrames - 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test.inpcrd")
        mol.write(filename)
        mol2 = Molecule(filename)
        assert mol_equal(mol, mol2, checkFields=["coords", "box", "boxangles"])
