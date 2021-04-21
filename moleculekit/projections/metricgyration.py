# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
from moleculekit.periodictable import periodictable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricGyration(Projection):
    """Creates a MetricGyration object that calculates the radius of gyration (ROG) of a molecule.

    Parameters
    ----------
    atomsel : str
        Atom selection string for the atoms whose ROG we want to calculate.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__

    Returns
    -------
    metr : MetricGyration object
    """

    def __init__(self, atomsel):
        super().__init__()

        if atomsel is None:
            raise ValueError("Atom selection cannot be None")

        self._atomsel = atomsel

    def _calculateMolProp(self, mol, props="all"):
        props = ("atomsel", "masses") if props == "all" else props
        res = {}
        if "atomsel" in props:
            res["atomsel"] = mol.atomselect(self._atomsel)
            if np.sum(res["atomsel"]) == 0:
                raise RuntimeError("Atom selection resulted in 0 atoms.")

        if "masses" in props:
            sel = mol.atomselect(self._atomsel)
            res["masses"] = mol.masses[sel]
            if np.any(res["masses"] == 0):
                logger.warning(
                    "The Molecule has atoms with 0 mass. Guessing the masses from the elements."
                )
                res["masses"] = np.array(
                    [periodictable[el].mass for el in mol.element[sel]]
                )
                if np.sum(res["masses"]) == 0:
                    raise RuntimeError(
                        "The molecule selection has 0 total mass. Please read atom masses from a prmtop or psf file."
                    )

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
        getMolProp = lambda prop: self._getMolProp(mol, prop)
        atomsel = getMolProp("atomsel")
        masses = getMolProp("masses")
        total_mass = np.sum(masses)

        coords = mol.coords[atomsel].copy()

        # Calculate center of mass
        com = np.sum(coords * masses[:, None, None], axis=0) / total_mass

        # Calculate radius of gyration
        coords -= com  # Remove COM from the coordinates
        rog = np.sqrt(
            np.sum(masses[:, None] * np.sum(coords ** 2, axis=1), axis=0) / total_mass
        )

        return rog

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
        from pandas import DataFrame

        getMolProp = lambda prop: self._getMolProp(mol, prop)

        atomidx = np.where(getMolProp("atomsel"))[0]
        types = ["rog"]
        indexes = [atomidx]
        description = ["Radius of gyration"]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )


import unittest


class _TestMetricCoordinate(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(
            path.join(home(dataDir="test-projections"), "trajectory", "filtered.psf")
        )
        mol.read(path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc"))

        self.mol = mol

    def test_project(self):
        metr = MetricGyration("protein")
        data = metr.project(self.mol)

        lastrog = np.array(
            [
                18.002188,
                17.97491,
                17.973713,
                17.949549,
                17.950699,
                17.9308,
                17.928637,
                18.000408,
                18.02674,
                17.98852,
                18.015263,
                17.934515,
                17.94321,
                17.949211,
                17.93479,
                17.924484,
                17.920536,
                17.860697,
                17.849443,
                17.879776,
            ],
            dtype=np.float32,
        )
        assert np.all(
            np.abs(data[-20:] - lastrog) < 0.001
        ), "Radius of gyration calculation is broken"


if __name__ == "__main__":
    unittest.main(verbosity=2)