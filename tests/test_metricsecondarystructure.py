from moleculekit.projections.metricsecondarystructure import MetricSecondaryStructure


def _test_2hbb():
    from moleculekit.molecule import Molecule
    import numpy as np

    mol = Molecule("2HBB")  # NTL9
    mol.filter("protein")
    metr = MetricSecondaryStructure()
    data = metr.project(mol)
    # fmt: off
    ref = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                    2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 0]], dtype=np.int32)
    assert np.array_equal(data, ref), 'MetricSecondaryStructure assertion failed'

    metr = MetricSecondaryStructure('resid 5 to 49')
    data = metr.project(mol)
    ref = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,
                    2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0]], dtype=np.int32)
    assert np.array_equal(data, ref), 'MetricSecondaryStructure assertion failed'
    # fmt: on


def _test_3ptb():
    from moleculekit.molecule import Molecule
    import numpy as np

    mol = Molecule("3PTB")  # Contains insertion which used to be a problem
    mol.filter("protein")
    metr = MetricSecondaryStructure()
    data = metr.project(mol)
    # fmt: off
    ref = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 0, 0]], dtype=np.int32)
    assert np.array_equal(data, ref), 'MetricSecondaryStructure assertion failed'
    assert MetricSecondaryStructure().getMapping(mol).shape == (223, 3)
    # fmt: on
