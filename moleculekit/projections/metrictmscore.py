# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import unittest
import logging

logger = logging.getLogger(__name__)


class MetricTMscore(Projection):
    """Calculates the TMscore of a set of trajectories to a reference structure

    Parameters
    ----------
    refmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The reference Molecule to which we want to calculate the TMscore.
    trajtmstr : str
        Atom selection string for the trajectories from which to calculate the TMscore.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    reftmstr : str, optional
        Atom selection string for the reference structure from which to calculate the TMscore. If None, it defaults to
        `trajrmsdstr`. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    centerstr : str, optional
        Atom selection string around which to center the wrapping of the trajectories.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    """

    def __init__(self, refmol, trajtmstr, reftmstr=None):
        super().__init__()

        if reftmstr is None:
            reftmstr = trajtmstr
        self._refmol = refmol
        self._reftmsel = self._refmol.atomselect(reftmstr) & (self._refmol.name == "CA")
        self._trajtmsel = trajtmstr

    def _calculateMolProp(self, mol, props="all"):
        res = {}
        res["trajtmsel"] = mol.atomselect(self._trajtmsel) & (mol.name == "CA")
        if np.sum(res["trajtmsel"]) == 0:
            raise RuntimeError("RMSD atom selection resulted in 0 atoms.")
        return res

    def project(self, mol):
        """Project molecule.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object to project.

        Returns
        -------
        data : np.ndarray
            An array containing the projected data.
        """
        from moleculekit.align import molTMscore

        mol = mol.copy()
        trajtmsel = self._getMolProp(mol, "trajtmsel")

        tm, _, _ = molTMscore(mol, self._refmol, trajtmsel, self._reftmsel)
        return tm[:, np.newaxis]

    def getMapping(self, mol):
        """Returns the description of each projected dimension.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object which will be used to calculate the descriptions of the projected dimensions.

        Returns
        -------
        map : :class:`DataFrame <pandas.core.frame.DataFrame>` object
            A DataFrame containing the descriptions of each dimension
        """
        trajtmsel = self._getMolProp(mol, "trajtmsel")
        from pandas import DataFrame

        types = ["tmscore"]
        indexes = [np.where(trajtmsel)[0]]
        description = ["TMscore to reference structure."]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )


class _TestMetricTMscore(unittest.TestCase):
    def test_tmscore(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        import numpy as np
        from os import path

        mol = Molecule(
            path.join(home(dataDir="test-projections"), "trajectory", "filtered.pdb")
        )
        mol.read(path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc"))
        ref = mol.copy()

        ref.dropFrames(keep=0)
        mol.dropFrames(keep=np.arange(mol.numFrames - 20, mol.numFrames))

        metr = MetricTMscore(ref, "protein and name CA")
        data = metr.project(mol)

        lasttm = np.array(
            [
                0.9633381,
                0.96441294,
                0.96553609,
                0.96088852,
                0.96288511,
                0.95677591,
                0.96544727,
                0.96359811,
                0.95658912,
                0.96893117,
                0.96623924,
                0.96064913,
                0.96207041,
                0.95947848,
                0.96657048,
                0.95993426,
                0.96543296,
                0.96806875,
                0.96437248,
                0.96144066,
            ],
            dtype=np.float32,
        )
        assert np.all(
            np.abs(data.flatten() - lasttm) < 0.001
        ), "TMscore calculation is broken"


if __name__ == "__main__":
    unittest.main(verbosity=2)
