from moleculekit.molecule import Molecule, mol_equal
from moleculekit.writers import _WRITERS
import numpy as np
import pytest
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("filetype", ["xtc", "dcd", "trr"])
def _test_trajectory_writers_roundtrip(tmp_path, filetype):
    natoms = 7
    nframes = 13

    mol = Molecule().empty(natoms)
    mol.coords = np.random.rand(natoms, 3, nframes).astype(np.float32) * 10
    mol.box = np.random.rand(3, nframes).astype(np.float32) * 10

    # The unitcell conversions fail if the angles are not realistic (not sure the exact conditions)
    mol.boxangles = np.array([[120, 120, 90]], dtype=np.float32).T
    mol.boxangles = np.tile(mol.boxangles, (1, nframes))

    mol.time = np.arange(nframes).astype(np.float32)
    mol.step = np.arange(nframes).astype(np.int32)

    mol.write(os.path.join(tmp_path, f"output.{filetype}"))

    mol2 = Molecule(os.path.join(tmp_path, f"output.{filetype}"))
    assert np.allclose(mol.coords, mol2.coords)
    assert np.allclose(mol.box, mol2.box)
    assert np.allclose(mol.boxangles, mol2.boxangles)


def _setupmol():
    from moleculekit.molecule import calculateUniqueBonds

    mol = Molecule(os.path.join(curr_dir, "test_writers", "filtered.psf"))
    mol.read(os.path.join(curr_dir, "test_writers", "filtered.pdb"))
    mol.coords = np.tile(mol.coords, (1, 1, 2))
    mol.filter("protein and resid 1 to 20")
    mol.boxangles = np.ones((3, 2), dtype=np.float32) * 90
    mol.box = np.ones((3, 2), dtype=np.float32) * 15
    mol.step = np.arange(2)
    mol.time = np.arange(2) * 1e5
    mol.fileloc = [mol.fileloc[0], mol.fileloc[0]]
    mol.bondtype[:] = "1"
    mol.bonds, mol.bondtype = calculateUniqueBonds(mol.bonds, mol.bondtype)
    return mol


_MOL = _setupmol()


@pytest.mark.parametrize("ext", list(_WRITERS.keys()))
def _test_writers(ext):
    from moleculekit.util import tempname

    if ext == "mmtf":
        pytest.skip("Not supported in tests due to deprecation")

    # Skip file-comparing binary filetypes
    # TODO: Remove SDF. Currently skipping it due to date in second line
    skipcomparison = (
        "ncrst",
        "rst7",
        "dcd",
        "h5",
        "nc",
        "netcdf",
        "ncdf",
        "xyz.gz",
        "xyz",
    )

    tmpfile = tempname(suffix="." + ext)
    if ext == "pdbqt":
        mol = _MOL.copy()
        mol.atomtype[:] = "NA"
        mol.write(tmpfile)
    elif ext == "mol2":
        _MOL.write(tmpfile, sel="resid 1")
    else:
        _MOL.write(tmpfile)
    if ext in skipcomparison:
        return

    reffile = os.path.join(curr_dir, "test_writers", "mol." + ext)

    try:
        with open(tmpfile, "r") as f:
            filelines = f.readlines()
            if ext == "sdf":
                filelines = filelines[2:]
    except UnicodeDecodeError:
        print(f"Could not compare file {reffile} due to not being unicode")
        return

    print("Testing file", reffile, tmpfile)
    if ext == "json":  # The precision is too high to compare files directly
        assert mol_equal(
            _MOL,
            Molecule(tmpfile),
            checkFields=Molecule._all_fields,
            exceptFields=("fileloc"),
        )
    else:
        with open(reffile, "r") as f:
            reflines = f.readlines()
            if ext == "sdf":
                reflines = reflines[2:]

        assert filelines == reflines, f"Failed comparison of {reffile} {tmpfile}"


def _test_sdf_writer():
    from moleculekit.molecule import Molecule
    from moleculekit.util import tempname

    reffile = os.path.join(curr_dir, "test_writers", "mol_bromium_out.sdf")
    mol = Molecule(os.path.join(curr_dir, "test_writers", "mol_bromium.sdf"))
    tmpfile = tempname(suffix=".sdf")
    mol.write(tmpfile)

    with open(tmpfile, "r") as f:
        filelines = f.readlines()[2:]
    with open(reffile, "r") as f:
        reflines = f.readlines()[2:]

    assert filelines == reflines, f"Failed comparison of {reffile} {tmpfile}"


def _test_psf_writer():
    from moleculekit.molecule import Molecule
    import tempfile

    # This ensures the right masses are written into the psf file from the elements

    reffile = os.path.join(curr_dir, "test_writers", "villin.psf")
    mol = Molecule(os.path.join(curr_dir, "test_writers", "villin.pdb"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "villin.psf")
        mol.write(tmpfile)

        with open(tmpfile, "r") as f:
            filelines = f.readlines()
        with open(reffile, "r") as f:
            reflines = f.readlines()

        assert filelines == reflines, f"Failed comparison of {reffile} {tmpfile}"


def _test_cif_mol2_atom_renaming():
    from moleculekit.molecule import Molecule
    import tempfile

    # This ensures the right masses are written into the psf file from the elements

    reffile1 = os.path.join(curr_dir, "test_writers", "BEN_ideal.cif")
    reffile2 = os.path.join(curr_dir, "test_writers", "BEN_ideal.mol2")
    mol = Molecule(os.path.join(curr_dir, "test_writers", "BEN_ideal.sdf"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "BEN_ideal.cif")
        mol.write(tmpfile)

        with open(tmpfile, "r") as f:
            filelines = f.readlines()
        with open(reffile1, "r") as f:
            reflines = f.readlines()

        assert filelines == reflines, f"Failed comparison of {reffile1} {tmpfile}"

        tmpfile = os.path.join(tmpdir, "BEN_ideal.mol2")
        mol.write(tmpfile)

        with open(tmpfile, "r") as f:
            filelines = f.readlines()
        with open(reffile2, "r") as f:
            reflines = f.readlines()

        assert filelines == reflines, f"Failed comparison of {reffile2} {tmpfile}"


@pytest.mark.parametrize("ext", ("netcdf", "trr", "binpos", "dcd", "xyz", "xyz.gz"))
def _test_traj_writers(ext):
    from moleculekit.molecule import Molecule
    import tempfile

    mol = Molecule(os.path.join(curr_dir, "test_readers", "1N09", "structure.prmtop"))
    mol.read(os.path.join(curr_dir, "test_readers", "1N09", "output.dcd"))

    with tempfile.TemporaryDirectory() as tmpdir:
        mol.write(os.path.join(tmpdir, f"output.{ext}"))
        molc = Molecule(os.path.join(tmpdir, f"output.{ext}"))

    if ext == "binpos":
        assert np.allclose(mol.coords, molc.coords, atol=1e-6)
    elif ext in ("xyz", "xyz.gz"):
        assert mol_equal(
            mol,
            molc,
            checkFields=["element", "coords"],
            fieldPrecision={"coords": 2e-5},
        )
    else:
        assert mol_equal(
            mol,
            molc,
            checkFields=Molecule._traj_fields,
            exceptFields=("fileloc"),
            fieldPrecision={"coords": 3e-6, "box": 3e-6},
        )


def _test_cif_roundtrip():
    from moleculekit.molecule import Molecule, mol_equal
    import tempfile

    mol = Molecule(os.path.join(curr_dir, "test_writers", "triala_capped.cif"))
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "triala_capped.cif")
        mol.write(outfile)
        mol2 = Molecule(outfile)
        assert mol_equal(
            mol,
            mol2,
            checkFields=Molecule._all_fields,
            uqBonds=True,
            exceptFields=("fileloc"),
        )


@pytest.mark.parametrize("ext", ["xsc", "trr", "dcd", "netcdf", "inpcrd"])
def _test_boxangle_writing(ext):
    from moleculekit.molecule import Molecule
    import tempfile

    angles = [[90, 90, 90], [45, 28, 17]]
    mol = Molecule().empty(10)
    mol.coords = np.zeros((10, 3, 1), dtype=np.float32)
    mol.time = np.array([0], dtype=np.float32)
    mol.step = np.array([0], dtype=np.int32)
    mol.box = np.array([[25], [11], [8]], dtype=np.float32)

    for ang in angles:
        mol.boxangles = np.array([[ang[0]], [ang[1]], [ang[2]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            mol.write(os.path.join(tmpdir, f"test.{ext}"))
            mol2 = Molecule(os.path.join(tmpdir, f"test.{ext}"))

            assert np.allclose(mol.box, mol2.box, atol=1e-5)
            assert np.allclose(mol.boxangles, mol2.boxangles, atol=1e-5)

    # Test with multiple frames as well
    mol.coords = np.tile(mol.coords, (1, 1, 2))
    mol.time = np.tile(mol.time, 2)
    mol.step = np.tile(mol.step, 2)
    for ang in angles:
        mol.boxangles = np.array([[ang[0]], [ang[1]], [ang[2]]], dtype=np.float32)
        molc = mol.copy()
        molc.boxangles = np.tile(mol.boxangles, (1, 2))
        molc.box = np.tile(mol.box, (1, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            molc.write(os.path.join(tmpdir, f"test.{ext}"))
            mol2 = Molecule(os.path.join(tmpdir, f"test.{ext}"))

            assert np.allclose(molc.box, mol2.box, atol=1e-5)
            assert np.allclose(molc.boxangles, mol2.boxangles, atol=1e-5)


def _test_non_square_box():
    from moleculekit.molecule import Molecule
    import tempfile

    datadir = os.path.join(curr_dir, "test_readers", "dodecahedral_box")
    mol = Molecule(os.path.join(datadir, "3ptb_dodecahedron.pdb"))
    mol.read(os.path.join(datadir, "output.xtc"))

    assert np.allclose(mol.boxangles[0, :], 120, atol=1e-2)
    assert np.allclose(mol.boxangles[1, :], 120, atol=1e-2)
    assert np.allclose(mol.boxangles[2, :], 90, atol=1e-2)
    refbox = np.array(
        [[71.419, 69.688385], [71.419, 69.688385], [71.419, 69.688385]],
        dtype=np.float32,
    )
    assert np.allclose(mol.box[:, :2], refbox, atol=1e-2)

    with tempfile.TemporaryDirectory() as tmpdir:
        mol.write(os.path.join(tmpdir, "3ptb_dodecahedron.xtc"))
        mol2 = Molecule(os.path.join(tmpdir, "3ptb_dodecahedron.xtc"))

    assert np.allclose(mol.box, mol2.box, atol=1e-2)
    assert np.allclose(mol.boxangles, mol2.boxangles, atol=1e-2)
