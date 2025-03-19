from moleculekit.tools.sequencestructuralalignment import sequenceStructureAlignment
from moleculekit.molecule import Molecule
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_align_protein():
    import os

    mol1 = Molecule(
        os.path.join(curr_dir, "test_sequencestructuralalignment", "4OBE.pdb")
    )
    mol2 = Molecule(
        os.path.join(curr_dir, "test_sequencestructuralalignment", "6OB2.pdb")
    )
    refmol = Molecule(
        os.path.join(curr_dir, "test_sequencestructuralalignment", "4OBE_aligned.pdb")
    )
    mol3, _ = sequenceStructureAlignment(mol1, mol2, molsel="protein", refsel="protein")

    assert np.allclose(refmol.coords, mol3[0].coords, atol=1e-3)


def _test_align_rna():
    import os

    mol1 = Molecule(
        os.path.join(curr_dir, "test_sequencestructuralalignment", "5C45.pdb")
    )
    mol2 = Molecule(
        os.path.join(curr_dir, "test_sequencestructuralalignment", "5C45_sim.pdb")
    )
    refmol = Molecule(
        os.path.join(curr_dir, "test_sequencestructuralalignment", "5C45_aligned.pdb")
    )
    mol3, _ = sequenceStructureAlignment(mol1, mol2)

    assert np.allclose(refmol.coords, mol3[0].coords, atol=1e-3)
