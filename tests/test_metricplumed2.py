import pytest
import os
from moleculekit.projections.metricplumed2 import _plumedExists

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(not _plumedExists(), reason="Plumed not found")
def _test_metricplumed2():
    import os
    import numpy as np
    from moleculekit.molecule import Molecule
    from moleculekit.projections.metricplumed2 import MetricPlumed2

    mol = Molecule(
        os.path.join(curr_dir, "test_projections", "metricplumed2", "1kdx_0.pdb")
    )
    mol.read(os.path.join(curr_dir, "test_projections", "metricplumed2", "1kdx.dcd"))

    metric = MetricPlumed2(["d1: DISTANCE ATOMS=1,200", "d2: DISTANCE ATOMS=5,6"])
    data = metric.project(mol)
    ref = np.array(
        [
            0.536674,
            21.722393,
            22.689391,
            18.402114,
            23.431387,
            23.13392,
            19.16376,
            20.393544,
            23.665517,
            22.298349,
            22.659769,
            22.667669,
            22.484084,
            20.893447,
            18.791701,
            21.833056,
            19.901318,
        ]
    )
    assert np.all(np.abs(ref - data[:, 0]) < 0.01), "Plumed demo calculation is broken"
