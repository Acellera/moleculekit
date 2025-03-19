import numpy as np
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_maximalSubstructureAlignment():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.graphalignment import maximalSubstructureAlignment

    ref_lig = Molecule(os.path.join(curr_dir, "test_graphalignment", "ref_lig.pdb"))
    lig2align = Molecule(os.path.join(curr_dir, "test_graphalignment", "lig2align.pdb"))
    lig_aligned = maximalSubstructureAlignment(ref_lig, lig2align)
    lig_reference = Molecule(
        os.path.join(curr_dir, "test_graphalignment", "lig_aligned.pdb")
    )

    assert np.allclose(
        lig_aligned.coords, lig_reference.coords, rtol=1e-4
    ), "maximalSubstructureAlignment produced different coords"


def _test_mcs_atom_matching():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.graphalignment import mcsAtomMatching
    import numpy as np

    mol1 = Molecule(os.path.join(curr_dir, "test_graphalignment", "OIC.cif"))
    mol1.atomtype = mol1.element
    idx = np.random.permutation(np.arange(mol1.numAtoms))
    mol1.reorderAtoms(idx)

    mol2 = Molecule("5VBL")
    mol2.filter("resname OIC")
    atm1, atm2 = mcsAtomMatching(mol1, mol2, bondCompare="order")
    assert len(atm1) == 11
    assert np.array_equal(mol1.name[atm1], mol2.name[atm2])

    atm1, atm2 = mcsAtomMatching(mol1, mol2, bondCompare="any")
    assert len(atm1) == 11
    # Compare elements here since it can match O to OXT when ignoring bond type
    assert np.array_equal(mol1.element[atm1], mol2.element[atm2])
