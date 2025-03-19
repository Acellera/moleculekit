from moleculekit.tools.docking import dock, _VINA_EXISTS
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(not _VINA_EXISTS, reason="No vina executable found")
def _test_docking():
    from moleculekit.molecule import Molecule
    from os import path

    protein = Molecule(path.join(curr_dir, "test_docking", "protein.pdb"))
    ligand = Molecule(path.join(curr_dir, "test_docking", "ligand.pdb"))
    poses, scoring = dock(protein, ligand)

    assert scoring.shape[0] == len(poses)
    assert scoring.shape[0] <= 20 and scoring.shape[0] > 15
    assert scoring.shape[1] == 3
