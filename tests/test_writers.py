from moleculekit.molecule import Molecule, mol_equal
from moleculekit.writers import _WRITERS
import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("filetype", ["xtc", "dcd", "trr"])
def test_trajectory_writers_roundtrip(tmp_path, filetype):
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
def test_writers(ext):
    from moleculekit.util import tempname

    if ext == "mmtf":
        pytest.skip("Not supported in tests due to deprecation")
    if ext == "h5":
        pytest.skip("Requires mdtraj and extra 'tables' package")

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


def test_sdf_writer():
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


def test_psf_writer():
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


def test_cif_mol2_atom_renaming(tmp_path):
    from moleculekit.molecule import Molecule

    # This ensures the right masses are written into the psf file from the elements

    reffile1 = os.path.join(curr_dir, "test_writers", "BEN_ideal.cif")
    reffile2 = os.path.join(curr_dir, "test_writers", "BEN_ideal.mol2")
    mol = Molecule(os.path.join(curr_dir, "test_writers", "BEN_ideal.sdf"))

    tmpfile = os.path.join(tmp_path, "BEN_ideal.cif")
    mol.write(tmpfile)

    with open(tmpfile, "r") as f:
        filelines = f.readlines()
    with open(reffile1, "r") as f:
        reflines = f.readlines()

    assert filelines == reflines, f"Failed comparison of {reffile1} {tmpfile}"

    tmpfile = os.path.join(tmp_path, "BEN_ideal.mol2")
    mol.write(tmpfile)

    with open(tmpfile, "r") as f:
        filelines = f.readlines()
    with open(reffile2, "r") as f:
        reflines = f.readlines()

    assert filelines == reflines, f"Failed comparison of {reffile2} {tmpfile}"


@pytest.mark.parametrize(
    "ext", ("xtc", "netcdf", "trr", "binpos", "dcd", "xyz", "xyz.gz")
)
@pytest.mark.parametrize("maxtime", [1e9, 1e15])
def test_traj_writers(ext, maxtime):
    from moleculekit.molecule import Molecule
    import tempfile

    mol = Molecule(os.path.join(curr_dir, "test_readers", "1N09", "structure.prmtop"))
    mol.read(os.path.join(curr_dir, "test_readers", "1N09", "output.dcd"))
    # 1e9 fs = 1us. Test if the trajectories can write steps of 100ps over 1us trajectories
    trajfreq = 25000
    timestep = 4
    timefreq = trajfreq * timestep
    mol.time[:] = np.arange(
        maxtime,
        maxtime + mol.numFrames * timefreq,
        timefreq,
        dtype=Molecule._dtypes["time"],
    )
    mol.step[:] = np.arange(
        maxtime / timestep,
        maxtime / timestep + (mol.numFrames * trajfreq),
        trajfreq,
        dtype=Molecule._dtypes["step"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mol.write(os.path.join(tmpdir, f"output.{ext}"))
        molc = Molecule(os.path.join(tmpdir, f"output.{ext}"))

    if maxtime > 1e10:
        mol.time -= (mol.step[0] - trajfreq) * timestep
        mol.step[:] = range(1, mol.numFrames + 1)

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
        coor_prec = 3e-6 if ext != "xtc" else 1e-2
        assert abs(mol.fstep - molc.fstep) < 1e-2
        assert mol_equal(
            mol,
            molc,
            checkFields=Molecule._traj_fields,
            exceptFields=("fileloc"),
            fieldPrecision={"coords": coor_prec, "box": 3e-6, "time": 1e-3},
        )


def test_cif_roundtrip():
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


def _two_lys_disagreeing_on_H3(tmp_path):
    """Two LYS residues that use the atom name ``H3`` for different atoms, which
    is what ``systemPrepare`` produces: a plain N-terminal LYS whose ``H3`` is a
    terminal amine hydrogen, and a crosslinked LYS re-templated from SMILES whose
    hydrogens carry rdkit's generic names, so its ``H3`` sits on ``CB``."""
    from moleculekit.molecule import Molecule

    names = ["N", "CA", "CB", "H3"]
    mol = Molecule().empty(10)
    # A second resname keeps this a normal structure file rather than a
    # single-component definition, where the template is exact by construction.
    mol.name[:] = names * 2 + ["N", "CA"]
    mol.element[:] = ["N", "C", "C", "H"] * 2 + ["N", "C"]
    mol.resname[:] = ["LYS"] * 8 + ["ALA"] * 2
    mol.resid[:] = [1] * 4 + [13] * 4 + [14] * 2
    mol.chain[:] = "A"
    mol.segid[:] = "P0"
    mol.record[:] = "ATOM"
    coords = np.zeros((10, 3), np.float32)
    for i in range(2):
        coords[i * 4 + 0] = [0.0, 3.0 * i, 0.0]  # N
        coords[i * 4 + 1] = [1.5, 3.0 * i, 0.0]  # CA
        coords[i * 4 + 2] = [2.2, 3.0 * i + 1.0, 0.0]  # CB
    coords[3] = [-0.9, 0.4, 0.0]  # LYS 1  H3 on N
    coords[7] = [2.9, 4.6, 0.0]  # LYS 13 H3 on CB
    coords[8] = [0.0, 9.0, 0.0]
    coords[9] = [1.5, 9.0, 0.0]
    mol.coords = coords.reshape(10, 3, 1)
    # N-H3 on LYS 1, CB-H3 on LYS 13: the same name, different partners.
    mol.bonds = np.array(
        [[0, 1], [1, 2], [0, 3], [4, 5], [5, 6], [6, 7], [8, 9]], dtype=np.uint32
    )
    mol.bondtype = np.array(["1"] * 7, dtype=object)
    return mol


def test_cif_conflicting_resname_bonds_roundtrip_exactly(tmp_path):
    """A per-resname ``chem_comp_bond`` template cannot express two instances of
    a resname that disagree about an atom name. Such a resname must fall back to
    per-instance ``struct_conn`` records so the bonds round-trip exactly, while
    resnames whose instances agree keep the compact template."""
    from moleculekit.molecule import Molecule

    mol = _two_lys_disagreeing_on_H3(tmp_path)

    out = str(tmp_path / "conflict.cif")
    mol.write(out)

    text = open(out).read()
    templated = {
        line.split()[0]
        for line in text.splitlines()
        if line.strip() and not line.startswith(("_", "#", "loop_", "data_"))
    }
    assert "LYS" not in templated, "conflicting LYS must not be templated"

    back = Molecule(out)
    assert back.numBonds == mol.numBonds
    before = {
        tuple(sorted((str(mol.name[int(a)]), str(mol.name[int(b)]))))
        for a, b in mol.bonds
    }
    after = {
        tuple(sorted((str(back.name[int(a)]), str(back.name[int(b)]))))
        for a, b in back.bonds
    }
    assert before == after
    deg = np.zeros(back.numAtoms, int)
    for a, b in back.bonds:
        deg[int(a)] += 1
        deg[int(b)] += 1
    h3 = [i for i in range(back.numAtoms) if str(back.name[i]) == "H3"]
    assert all(deg[i] == 1 for i in h3), "H3 must keep exactly one bond per instance"


def test_cif_agreeing_resname_uses_chem_comp_bond(tmp_path):
    """Two instances of a resname carrying the same bonds under the same names
    are compressed into one template, and still round-trip exactly."""
    from moleculekit.molecule import Molecule

    mol = _two_lys_disagreeing_on_H3(tmp_path)
    # Move LYS 13's H3 onto N, so both LYS instances now agree.
    h3_13 = np.where((mol.resid == 13) & (mol.name == "H3"))[0]
    n_13 = np.where((mol.resid == 13) & (mol.name == "N"))[0]
    mol.bonds[5] = [n_13[0], h3_13[0]]

    out = str(tmp_path / "agree.cif")
    mol.write(out)
    text = open(out).read()
    assert "chem_comp_bond" in text
    templated = {
        line.split()[0]
        for line in text.splitlines()
        if line.strip() and not line.startswith(("_", "#", "loop_", "data_"))
    }
    assert "LYS" in templated

    back = Molecule(out)
    assert back.numBonds == mol.numBonds


def test_cif_single_component_still_uses_chem_comp_bond(tmp_path):
    """A one-component file has a single instance by construction, so the
    template is exact and stays the compact representation the PDB deposits."""
    from moleculekit.molecule import Molecule

    mol = Molecule().empty(2)
    mol.name[:] = ["C1", "C2"]
    mol.element[:] = ["C", "C"]
    mol.resname[:] = "LIG"
    mol.resid[:] = 1
    mol.record[:] = "HETATM"
    mol.coords = np.array([[[0.0]], [[1.5]]], dtype=np.float32).repeat(3, axis=1)
    mol.coords = np.zeros((2, 3, 1), dtype=np.float32)
    mol.coords[1, 0, 0] = 1.5
    mol.bonds = np.array([[0, 1]], dtype=np.uint32)
    mol.bondtype = np.array(["1"], dtype=object)

    out = str(tmp_path / "lig.cif")
    mol.write(out)
    text = open(out).read()
    assert "chem_comp_bond" in text
    assert Molecule(out).numBonds == 1


@pytest.mark.parametrize("ext", ["xsc", "trr", "dcd", "netcdf", "inpcrd"])
def test_boxangle_writing(ext):
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


def test_non_square_box():
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


def test_label_seq_ids_separate_insertion_coded_residues():
    """A residue and its insertion-coded partner must not share a position.

    ``label_seq_id`` is a residue's position in its entity and has to be
    unique within a chain, so writing resid into it collapsed 184 and 184A
    into one residue: readers merged their atoms and the polymer trace broke,
    which drew trypsin's three insertion sites as half a residue each.
    """
    from moleculekit.writers import _label_seq_ids

    mol = Molecule().empty(6)
    mol.chain[:] = "A"
    mol.resid[:] = [183, 184, 184, 185, 186, 187]
    mol.insertion[:] = ["", "", "A", "", "", ""]

    labels = _label_seq_ids(mol)
    assert len(set(labels)) == 6, f"positions are not unique: {labels}"
    # Author numbering is followed until the insertion forces it apart.
    assert list(labels) == [183, 184, 185, 186, 187, 188]


def test_label_seq_ids_are_the_resids_without_insertion_codes():
    """The common case must be written exactly as before."""
    from moleculekit.writers import _label_seq_ids

    mol = Molecule().empty(5)
    mol.chain[:] = ["A", "A", "A", "B", "B"]
    mol.resid[:] = [10, 11, 12, 4, 5]
    mol.insertion[:] = ""

    assert list(_label_seq_ids(mol)) == [10, 11, 12, 4, 5]
