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


def _test_autosegment_detailed():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule("3ptb")
    outmol = autoSegment(mol, fields=("chain", "segid"))

    prot_idx = np.arange(1629)
    ca_idx = np.array([1629])
    ben_idx = np.arange(1630, 1639)
    water_idx = np.arange(1639, 1701)

    chain_a = np.where(outmol.chain == "A")[0]
    assert np.array_equal(chain_a, prot_idx)
    chain_b = np.where(outmol.chain == "B")[0]
    assert np.array_equal(chain_b, ca_idx)
    chain_c = np.where(outmol.chain == "C")[0]
    assert np.array_equal(chain_c, ben_idx)
    chain_w = np.where(outmol.chain == "W")[0]
    assert np.array_equal(chain_w, water_idx)

    # A is kept by CA, BEN and waters. B is assigned to protein.
    outmol = autoSegment(mol, fields=("chain", "segid"), sel="protein")
    chain_a = np.where(outmol.chain == "A")[0]
    assert np.array_equal(chain_a, np.hstack([ca_idx, ben_idx, water_idx]))
    chain_b = np.where(outmol.chain == "B")[0]
    assert np.array_equal(chain_b, prot_idx)

    # A is kept by CA and waters. B is assigned to protein. C is assigned to BEN.
    outmol = autoSegment(mol, fields=("chain", "segid"), sel="protein or resname BEN")
    chain_a = np.where(outmol.chain == "A")[0]
    assert np.array_equal(chain_a, np.hstack([ca_idx, water_idx]))
    chain_b = np.where(outmol.chain == "B")[0]
    assert np.array_equal(chain_b, prot_idx)
    chain_c = np.where(outmol.chain == "C")[0]
    assert np.array_equal(chain_c, ben_idx)

    # There are no nucleic so it stays as original
    outmol = autoSegment(mol, fields=("chain", "segid"), sel="nucleic")
    chain_a = np.where(outmol.chain == "A")[0]
    assert np.array_equal(chain_a, np.arange(mol.numAtoms))
