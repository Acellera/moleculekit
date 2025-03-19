import pytest
import os
import numpy as np
from moleculekit.smallmol.smallmolcdp import cdp_installed

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "smiles",
    ((0, "Oc1c(cccc3)c3nc2ccncc12", 3), (1, "CN=c1nc[nH]cc1", 3), (2, "CC=CO", 2)),
)
def _test_tautomers(smiles):
    from moleculekit.util import file_diff
    from moleculekit.smallmol.smallmol import SmallMol
    import tempfile

    i, smiles, n_tauts = smiles
    mol = SmallMol(smiles, fixHs=False)
    tauts, scores = mol.getTautomers(canonical=False, genConformers=False)
    assert len(tauts) == n_tauts
    with tempfile.TemporaryDirectory() as tmpdir:
        newfile = os.path.join(tmpdir, "tautomers.sdf")
        tauts.writeSdf(os.path.join(tmpdir, "tautomers.sdf"))
        reffile = os.path.join(curr_dir, "test_smallmol", f"tautomer_results_{i}.sdf")
        file_diff(newfile, reffile)


@pytest.mark.skipif(not cdp_installed, reason="CDPKit not installed. Skipping test.")
def _test_smallmolcdp():
    from moleculekit.home import home
    from moleculekit.smallmol.smallmolcdp import SmallMolCDP

    sm = SmallMolCDP(home("test-smallmol/Imatinib.sdf"))
    sm.generateConformers(10)
    assert sm.coords.shape == (68, 3, 10)

    mol = sm.toMolecule()
    assert mol.coords.shape == (68, 3, 10)
    assert mol.bonds.shape == (72, 2)
    assert mol.bondtype.shape == (72,)
    assert np.sum(mol.bondtype == "2") == 13
