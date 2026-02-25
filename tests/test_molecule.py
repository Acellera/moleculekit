from moleculekit.molecule import (
    Molecule,
    mol_equal,
    UniqueResidueID,
    UniqueAtomID,
    getBondedGroups,
)
import pytest
import numpy as np
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))

TRAJMOL = Molecule(os.path.join(curr_dir, "test_molecule", "3ptb_filtered.pdb"))
TRAJMOL.read(os.path.join(curr_dir, "test_molecule", "3ptb_traj.xtc"))

TRAJMOLLIG = TRAJMOL.copy()
_ = TRAJMOLLIG.filter("resname MOL")

MOL3PTB = Molecule("3PTB")


def _test_trajReadingAppending():
    # Testing trajectory reading and appending
    ref = Molecule(os.path.join(curr_dir, "test_molecule", "3ptb_filtered.pdb"))
    xtcfile = os.path.join(curr_dir, "test_molecule", "3ptb_traj.xtc")
    ref.read(xtcfile)
    assert ref.coords.shape == (4507, 3, 200)
    ref.read(xtcfile, append=True)
    assert ref.coords.shape == (4507, 3, 400)
    ref.read([xtcfile, xtcfile, xtcfile])
    assert ref.coords.shape == (4507, 3, 600)


def _test_guessBonds():
    # Checking bonds
    ref = TRAJMOL.copy()
    ref.coords = np.atleast_3d(ref.coords[:, :, 0])
    len1 = len(ref._guessBonds())
    ref.coords = np.array(ref.coords, dtype=np.float32)
    len3 = len(ref._guessBonds())
    assert len1 == 4562
    assert len3 == 4562


def _test_setDihedral():
    # Testing dihedral setting
    mol = Molecule("2HBB")
    quad = [124, 125, 132, 133]
    mol.setDihedral(quad, np.deg2rad(-90), guessBonds=True)
    angle = mol.getDihedral(quad)
    assert np.abs(np.deg2rad(-90) - angle) < 1e-3


def _test_updateBondsAnglesDihedrals():
    # Testing updating of bonds, dihedrals and angles after filtering
    mol = Molecule(os.path.join(curr_dir, "test_molecule", "a1e.prmtop"))
    mol.read(os.path.join(curr_dir, "test_molecule", "a1e.pdb"))
    _ = mol.filter("not water")
    bb, bt, di, im, an = np.load(
        os.path.join(
            curr_dir, "test_molecule", "updatebondsanglesdihedrals_nowater.npy"
        ),
        allow_pickle=True,
    )
    assert np.array_equal(bb, mol.bonds)
    assert np.array_equal(bt, mol.bondtype)
    assert np.array_equal(di, mol.dihedrals)
    assert np.array_equal(im, mol.impropers)
    assert np.array_equal(an, mol.angles)
    _ = mol.filter("not index 8 18")
    bb, bt, di, im, an = np.load(
        os.path.join(
            curr_dir,
            "test_molecule",
            "updatebondsanglesdihedrals_remove8_18.npy",
        ),
        allow_pickle=True,
    )
    assert np.array_equal(bb, mol.bonds)
    assert np.array_equal(bt, mol.bondtype)
    assert np.array_equal(di, mol.dihedrals)
    assert np.array_equal(im, mol.impropers)
    assert np.array_equal(an, mol.angles)


def _test_appendingBondsBondtypes():
    # Testing appending of bonds and bondtypes
    mol = MOL3PTB.copy()
    # TODO do not use parameterize data
    lig = Molecule(os.path.join(curr_dir, "test_molecule", "h2o2.mol2"))
    assert mol.bonds.shape[0] == len(
        mol.bondtype
    )  # Checking that Molecule fills in empty bondtypes
    newmol = Molecule()
    newmol.append(lig)
    newmol.append(mol)
    assert newmol.bonds.shape[0] == (mol.bonds.shape[0] + lig.bonds.shape[0])
    assert newmol.bonds.shape[0] == len(newmol.bondtype)


def _test_uniqueAtomID():
    mol = MOL3PTB.copy()
    uqid = UniqueAtomID.fromMolecule(mol, "resid 20 and name CA")
    assert uqid.selectAtom(mol) == 24
    mol.remove("resid 19")
    assert uqid.selectAtom(mol) == 20
    a1 = UniqueAtomID.fromMolecule(mol, "resid 20 and name CA")
    a2 = UniqueAtomID.fromMolecule(mol, idx=20)
    assert a1 == a2


def _test_uniqueResidueID():
    mol = MOL3PTB.copy()
    uqid = UniqueResidueID.fromMolecule(mol, "resid 20")
    assert np.array_equal(
        uqid.selectAtoms(mol),
        np.array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]),
    )
    mol.remove("resid 19")
    assert np.array_equal(
        uqid.selectAtoms(mol),
        np.array([19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
    )
    r1 = UniqueResidueID.fromMolecule(mol, "resid 20 and name CA")
    r2 = UniqueResidueID.fromMolecule(mol, "resid 20 and name CB")
    r3 = UniqueResidueID.fromMolecule(mol, "resid 21 and name CA")
    assert r1 == r2
    assert r2 != r3


def _test_selfalign():
    # Checking bonds
    mol = TRAJMOLLIG.copy()
    mol.align("noh")

    refcoords = np.load(
        os.path.join(curr_dir, "test_molecule", "test-selfalign-mol.npy"),
        allow_pickle=True,
    )

    assert np.allclose(mol.coords, refcoords, atol=1e-3)


def _test_alignToReference():
    # Checking bonds
    mol = TRAJMOLLIG.copy()

    mol2 = mol.copy()
    mol2.dropFrames(keep=3)  # Keep a random frame
    _ = mol2.filter(
        "noh"
    )  # Remove some atoms to check aligning molecules with different numAtoms

    mol.align("noh", refmol=mol2)

    refcoords = np.load(
        os.path.join(curr_dir, "test_molecule", "test-align-refmol.npy"),
        allow_pickle=True,
    )

    assert np.allclose(mol.coords, refcoords, atol=1e-3)
    assert np.allclose(
        mol.coords[mol.atomselect("noh"), :, 3], mol2.coords[:, :, 0], atol=1e-3
    )


def _test_alignToReferenceMatchingFrames():
    # Checking bonds
    mol = TRAJMOLLIG.copy()

    mol2 = mol.copy()
    mol2.coords = np.roll(mol.coords, 3, axis=2)

    mol.align("noh", refmol=mol2, matchingframes=True)

    refcoords = np.load(
        os.path.join(curr_dir, "test_molecule", "test-align-refmol-matchingframes.npy"),
        allow_pickle=True,
    )

    assert np.allclose(mol.coords, refcoords, atol=1e-3)


def _test_alignToReferenceSpecificFrames():
    # Checking bonds
    mol = TRAJMOLLIG.copy()

    mol2 = mol.copy()
    mol2.dropFrames(keep=3)  # Keep a random frame
    _ = mol2.filter(
        "noh"
    )  # Remove some atoms to check aligning molecules with different numAtoms

    originalcoords = mol.coords.copy()

    mol.align("noh", refmol=mol2, frames=[0, 1, 2, 3])

    refcoords = np.load(
        os.path.join(curr_dir, "test_molecule", "test-align-refmol-selectedframes.npy"),
        allow_pickle=True,
    )

    assert np.allclose(originalcoords[:, :, 4:], mol.coords[:, :, 4:], atol=1e-3)
    assert np.allclose(mol.coords, refcoords, atol=1e-3)


def _test_reorderAtoms():
    mol = Molecule()
    _ = mol.empty(8)
    mol.name[:] = ["C1", "C3", "H", "O", "S", "N", "H3", "H1"]
    randcoords = np.random.rand(8, 3, 1).astype(np.float32)
    mol.coords = randcoords.copy()
    mol.bonds = np.array([[0, 3], [1, 4], [7, 2]])
    mol.bondtype = np.array(["un", "1", "2"], dtype=object)
    mol.angles = np.array([[0, 3, 2], [1, 4, 6], [7, 2, 5]])
    mol.dihedrals = np.array([[0, 3, 1, 2], [1, 4, 2, 7], [7, 2, 1, 0]])
    mol.impropers = mol.dihedrals.copy()
    neworder = [1, 2, 4, 3, 0, 7, 5, 6]
    mol.reorderAtoms(neworder)
    assert np.array_equal(mol.name, ["C3", "H", "S", "O", "C1", "H1", "N", "H3"])
    assert np.array_equal(mol.bonds, np.array([[4, 3], [0, 2], [5, 1]]))
    assert np.array_equal(mol.bondtype, ["un", "1", "2"])
    assert np.array_equal(mol.angles, np.array([[4, 3, 1], [0, 2, 7], [5, 1, 6]]))
    assert np.array_equal(
        mol.dihedrals, np.array([[4, 3, 0, 1], [0, 2, 1, 5], [5, 1, 0, 4]])
    )
    assert np.array_equal(
        mol.impropers, np.array([[4, 3, 0, 1], [0, 2, 1, 5], [5, 1, 0, 4]])
    )
    assert np.array_equal(mol.coords, randcoords[neworder])


def _test_sequence():
    seq, seqatms = MOL3PTB.getSequence(return_idx=True)
    refseq = "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
    assert seq["A"] == refseq

    # Ensure that the returned indexes only belong to a single residue
    for indexes in seqatms["A"]:
        assert len(np.unique(MOL3PTB.resname[indexes])) == 1
        assert len(np.unique(MOL3PTB.resid[indexes])) == 1

    seq = MOL3PTB.getSequence(sel="resid 16 to 50")
    assert seq == {"A": "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQ"}

    mol = Molecule("1lkk")
    seq, idx = mol.getSequence(return_idx=True)
    assert seq == {
        "A": "LEPEPWFFKNLSRKDAERQLLAPGNTHGSFLIRESESTAGSFSLSVRDFDQNQGEVVKHYKIRNLDNGGFYISPRITFPGLHELVRHYTNASDGLCTRLSRPCQT",
        "B": "XEEI",
    }

    refidx = np.array(
        [
            1688,
            1689,
            1690,
            1691,
            1692,
            1693,
            1694,
            1695,
            1696,
            1697,
            1698,
            1699,
            1700,
            1701,
            1702,
        ]
    )
    assert np.array_equal(idx["B"][1], refidx)
    seq = mol.getSequence(dict_key="segid")
    assert seq == {
        "1": "LEPEPWFFKNLSRKDAERQLLAPGNTHGSFLIRESESTAGSFSLSVRDFDQNQGEVVKHYKIRNLDNGGFYISPRITFPGLHELVRHYTNASDGLCTRLSRPCQT",
        "2": "XEEI",
    }
    assert mol.getSequence(one_letter=False)["B"] == ["PTR", "GLU", "GLU", "ILE"]


def _test_appendFrames():
    trajmol = TRAJMOL.copy()
    nframes = trajmol.numFrames
    trajmol.appendFrames(trajmol)
    assert trajmol.numFrames == (2 * nframes)


def _test_renumberResidues():
    mol = MOL3PTB.copy()
    _ = mol.renumberResidues(returnMapping=True)
    refres = np.load(
        os.path.join(curr_dir, "test_molecule", "renumberedresidues.npy"),
        allow_pickle=True,
    )
    assert np.array_equal(mol.resid, refres)


def _test_str_repr():
    assert len(MOL3PTB.__str__()) != 0
    assert len(MOL3PTB.__repr__()) != 0


def _test_mol_equal():
    assert mol_equal(MOL3PTB, MOL3PTB)
    assert not mol_equal(MOL3PTB, TRAJMOL)


def _test_mol_equal_precision():
    mol2 = MOL3PTB.copy()
    mol2.coords += 0.001
    assert mol_equal(MOL3PTB, mol2, fieldPrecision={"coords": 1e-2})
    assert not mol_equal(MOL3PTB, mol2, fieldPrecision={"coords": 1e-4})


def _test_append_collision_to_empty_mol():
    mol = Molecule()
    mol1 = Molecule("3ptb")
    mol.append(mol1, collisions=True)

    mol = Molecule()
    mol.append(mol1)


def _test_append_collisions():
    mol = Molecule("3ptb")
    ben = mol.copy()
    ben.filter("resname BEN")
    ben2 = ben.copy()
    mol.filter("protein")

    # Removes protein residues that are within 6A of BEN
    ben.append(mol, collisions=True, coldist=6)
    assert ben.numAtoms == 1469

    # When specifying a removesel you don't remove atoms which are not specified
    ben2.append(mol, collisions=True, coldist=6, removesel="water")
    assert ben2.numAtoms == 1638


def _test_split_append_insert_trajectory():
    lig = TRAJMOL.copy()
    lig.filter("resname MOL")
    rest = TRAJMOL.copy()
    rest.filter("not resname MOL")

    insertidx = np.where(TRAJMOL.resname == "MOL")[0][0]

    newmol = Molecule()
    newmol.append(rest)
    newmol.insert(lig, insertidx)
    assert mol_equal(TRAJMOL, newmol)


def _test_append_inverse_collisions():
    from moleculekit.molecule import Molecule

    import numpy as np

    mol = Molecule(os.path.join(curr_dir, "test_molecule", "memb_with_water.pdb"))
    n_waters = int(np.sum(mol.resname == "TIP3") / 3)
    n_lipids = np.sum(mol.resname == "POPC")
    popc_n_atoms = 134
    acetone = Molecule(os.path.join(curr_dir, "test_molecule", "ACT.cif"))

    water_index = 46331
    acetone.center()
    acetone.moveBy(mol.coords[water_index])

    mol.append(acetone, collisions=True, invertcollisions=True)
    # Check that all atoms exist
    assert np.sum(mol.resname == "ACT") == acetone.numAtoms
    # Check that only waters were deleted
    assert np.sum(mol.resname == "TIP3") % 3 == 0
    new_n_waters = int(np.sum(mol.resname == "TIP3") / 3)
    new_n_lipids = np.sum(mol.resname == "POPC")
    assert new_n_waters == (n_waters - 4)  # Removed 4 waters
    assert new_n_lipids == n_lipids

    acetone.center()
    acetone.moveBy(mol.coords[58189])
    mol.insert(acetone, 58189, collisions=True, invertcollisions=True)
    # Check that all atoms exist
    assert np.sum(mol.resname == "ACT") == 2 * acetone.numAtoms
    # Check that only waters were deleted
    assert np.sum(mol.resname == "TIP3") % 3 == 0
    new_n_waters = int(np.sum(mol.resname == "TIP3") / 3)
    new_n_lipids = np.sum(mol.resname == "POPC")
    assert new_n_waters == (n_waters - 6)  # Removed total 6 waters
    assert new_n_lipids == n_lipids

    # Try to insert now in the lipids
    acetone.center()
    acetone.moveBy(mol.coords[7700])
    mol.insert(acetone, 7700, collisions=True, invertcollisions=True)
    # Check that all atoms exist
    assert np.sum(mol.resname == "ACT") == 3 * acetone.numAtoms
    # Check that only waters were deleted
    assert np.sum(mol.resname == "TIP3") % 3 == 0
    new_n_waters = int(np.sum(mol.resname == "TIP3") / 3)
    new_n_lipids = np.sum(mol.resname == "POPC")
    assert new_n_waters == (n_waters - 6)  # No waters were killed
    assert (n_lipids - new_n_lipids) == 1 * popc_n_atoms  # 1 lipid removed


def _test_advanced_copy():
    trajmol = Molecule(os.path.join(curr_dir, "test_wrapping", "structure.prmtop"))
    trajmol.read(os.path.join(curr_dir, "test_wrapping", "output.xtc"))

    traj2 = trajmol.copy(frames=[1, 3])
    traj3 = trajmol.copy()
    traj3.dropFrames(keep=[1, 3])
    assert mol_equal(
        traj2,
        trajmol,
        checkFields=Molecule._all_fields,
        exceptFields=["coords", "box", "boxangles", "fileloc", "step", "time"],
        dtypes=True,
    )
    assert mol_equal(traj2, traj3, dtypes=True)
    assert np.array_equal(traj2.coords, trajmol.coords[:, :, [1, 3]])
    assert not np.array_equal(traj2.coords, trajmol.coords[:, :, [2, 3]])

    traj2 = trajmol.copy(sel=trajmol.resid == 10, frames=[1, 3])
    traj3 = trajmol.copy()
    traj3.filter(traj3.resid == 10)
    traj3.dropFrames(keep=[1, 3])
    assert mol_equal(traj2, traj3, checkFields=Molecule._all_fields, dtypes=True)

    traj2 = trajmol.copy(sel=trajmol.resid == 10)
    traj3 = trajmol.copy()
    traj3.filter(traj3.resid == 10)
    assert mol_equal(traj2, traj3, checkFields=Molecule._all_fields, dtypes=True)

    traj2 = trajmol.copy(sel=np.where(trajmol.resid == 10)[0])
    traj3 = trajmol.copy()
    traj3.filter(traj3.resid == 10)
    assert mol_equal(traj2, traj3, checkFields=Molecule._all_fields, dtypes=True)


def _test_connected_components():
    import networkx as nx

    class _FakeMol:
        pass

    n_atoms = 1000
    n_bonds = 800

    for _ in range(10):
        bonds = []
        for _ in range(n_bonds):
            bonds.append(np.random.randint(0, n_atoms, 2))
        bonds = np.array(bonds, dtype=np.uint32)
        g = nx.Graph()
        g.add_nodes_from(range(n_atoms))
        g.add_edges_from(bonds)
        conn_comp = list(nx.connected_components(g))

        mol = _FakeMol()
        mol.bonds = bonds
        mol.numAtoms = n_atoms
        _, group_mask = getBondedGroups(mol)
        groups = [
            set(np.where(group_mask == x)[0]) for x in range(group_mask.max() + 1)
        ]
        assert len(groups) == len(conn_comp)

        for cc in conn_comp:
            assert cc in groups

    mol = Molecule(os.path.join(curr_dir, "test_wrapping", "6X18.psf"))

    _, group_mask = getBondedGroups(mol)
    expected = np.load(
        os.path.join(curr_dir, "test_wrapping", "6X18_expected_group_mask.npy")
    )
    assert np.array_equal(group_mask, expected)

    mol.read(os.path.join(curr_dir, "test_wrapping", "6X18.xtc"))
    mol.wrap("protein and segid P3")
    dims = mol.coords.max(axis=0) - mol.coords.min(axis=0)
    assert np.all(np.abs(dims - mol.box) < 17.5)  # 17.5 A because lipids stick out


def _test_atomselect():
    mol = Molecule("3ptb")
    sel = mol.atomselect("protein")
    assert sel.dtype == bool
    assert sel.sum() == 1629

    sel = mol.atomselect("protein", indexes=True)
    assert sel.dtype == np.uint32
    assert np.array_equal(sel, np.arange(1629))

    sel = mol.atomselect(np.arange(10, 20))
    assert sel.dtype == bool
    assert sel.sum() == 10

    sel = mol.atomselect(np.arange(10, 20), indexes=True)
    assert sel.dtype == np.uint32
    assert np.array_equal(sel, np.arange(10, 20))

    sel = mol.atomselect(mol.resid == 20)
    assert sel.dtype == bool
    assert sel.sum() == 12

    sel = mol.atomselect(mol.resid == 20, indexes=True)
    assert sel.dtype == np.uint32
    assert np.array_equal(sel, np.arange(23, 35))

    sel = mol.atomselect("all")
    assert sel.dtype == bool
    assert np.sum(sel) == mol.numAtoms

    sel = mol.atomselect("all", indexes=True)
    assert sel.dtype == np.uint32
    assert np.array_equal(sel, np.arange(mol.numAtoms))

    sel = mol.atomselect(None)
    assert sel.dtype == bool
    assert np.sum(sel) == mol.numAtoms

    sel = mol.atomselect(None, indexes=True)
    assert sel.dtype == np.uint32
    assert np.array_equal(sel, np.arange(mol.numAtoms))


def _test_large_time_fstep():
    mol = Molecule().empty(10)
    mol.time = np.arange(1e15, 1.000000001e15, 4, dtype=Molecule._dtypes["time"])
    mol.fileloc = ["x"] * mol.time.shape[0]
    assert mol.fstep == 4e-6
