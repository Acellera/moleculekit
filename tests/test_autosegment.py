import numpy as np
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_autoSegment():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment
    from os import path

    p = Molecule(path.join(curr_dir, "test_autosegment", "4dkl.pdb"))
    p.filter("(chain B and protein) or water")
    p = autoSegment(p, "protein", "P")
    m = Molecule(path.join(curr_dir, "test_autosegment", "membrane.pdb"))
    print(np.unique(m.get("segid")))

    mol = Molecule(path.join(curr_dir, "test_autosegment", "1ITG_clean.pdb"))
    ref = Molecule(path.join(curr_dir, "test_autosegment", "1ITG.pdb"))
    mol = autoSegment(mol, sel="protein")
    assert np.all(mol.segid == ref.segid)

    mol = Molecule(path.join(curr_dir, "test_autosegment", "3PTB_clean.pdb"))
    ref = Molecule(path.join(curr_dir, "test_autosegment", "3PTB.pdb"))
    mol = autoSegment(mol, sel="protein")
    assert np.all(mol.segid == ref.segid)


def _test_autoSegment2():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment2
    from os import path

    mol = Molecule(path.join(curr_dir, "test_autosegment", "3segments.pdb"))
    smol1 = autoSegment2(mol, sel="nucleic or protein or resname ACE NME")
    smol2 = autoSegment2(mol)
    assert np.array_equal(smol1.segid, smol2.segid)
    vals, counts = np.unique(smol1.segid, return_counts=True)
    assert len(vals) == 3
    assert np.array_equal(counts, np.array([331, 172, 1213]))
