# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricShell(Projection):
    """Calculates the density of atoms around other atoms.

    The MetricShell class calculates the density of a set of
    interchangeable atoms in concentric spherical shells around some
    other atoms. Thus it can treat identical molecules (like water or
    ions) and calculate summary values like the changes in water density
    around atoms. It produces a n-by-s dimensional vector where n the
    number of atoms in the first selection and s the number of shells
    around each of the n atoms.

    Parameters
    ----------
    sel1 : str
        Atom selection string for the first set of atoms around which the shells will be calculated.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    sel2 : str
        Atom selection string for the second set of atoms whose density will be calculated in shells around `sel1`.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    periodic : str
        See the documentation of MetricDistance class for options.
    numshells : int, optional
        Number of shells to use around atoms of `sel1`
    shellwidth : int, optional
        The width of each concentric shell in Angstroms
    gap : int, optional
        Not functional yet
    truncate : float, optional
        Set all distances larger than `truncate` to `truncate`
    """

    def __init__(
        self,
        sel1,
        sel2,
        periodic,
        numshells=4,
        shellwidth=3,
        pbc=None,
        gap=None,
        truncate=None,
    ):
        super().__init__()

        if pbc is not None:
            raise DeprecationWarning(
                "The `pbc` option is deprecated please use the `periodic` option as described in MetricDistance."
            )

        from moleculekit.projections.metricdistance import MetricDistance

        self.symmetrical = sel1 == sel2
        self.metricdistance = MetricDistance(
            sel1=sel1,
            sel2=sel2,
            periodic=periodic,
            groupsel1=None,
            groupsel2=None,
            metric="distances",
            threshold=8,
            truncate=truncate,
        )

        self.numshells = numshells
        self.shellwidth = shellwidth
        self.description = None
        self.shellcenters = None

    def _calculateMolProp(self, mol, props="all"):
        props = (
            ("map", "shellcenters", "shelledges", "shellvol")
            if props == "all"
            else props
        )
        res = {}

        mapping = np.vstack(self.metricdistance.getMapping(mol).atomIndexes)
        if "map" in props:
            res["map"] = mapping
        if "shellcenters" in props:
            res["shellcenters"] = (
                np.unique(mapping[:, 0]) if not self.symmetrical else np.unique(mapping)
            )
        if "shelledges" in props:
            res["shelledges"] = np.arange(
                self.shellwidth * (self.numshells + 1), step=self.shellwidth
            )
        if "shellvol" in props:
            res["shellvol"] = (
                4
                / 3
                * np.pi
                * (res["shelledges"][1:] ** 3 - res["shelledges"][:-1] ** 3)
            )

        return res

    def project(self, mol):
        """Project molecule.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object to project.
        kwargs :
            Do not use this argument. Only used for backward compatibility. Will be removed in later versions.

        Returns
        -------
        data : np.ndarray
            An array containing the projected data.
        """
        molprops = self._getMolProp(mol, "all")

        distances = self.metricdistance.project(mol)
        if distances.ndim == 1:
            distances = distances[np.newaxis, :]

        return _shells(
            distances,
            molprops["map"],
            molprops["shellcenters"],
            self.numshells,
            molprops["shelledges"],
            molprops["shellvol"],
            self.symmetrical,
        )

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
        shellcenters = self._getMolProp(mol, "shellcenters")

        from pandas import DataFrame

        types = []
        indexes = []
        description = []
        for i in shellcenters:
            for n in range(self.numshells):
                types += ["shell"]
                indexes += [i]
                description += [
                    "Density of sel2 atoms in shell {}-{} A centered on atom {} {} {}".format(
                        n * self.shellwidth,
                        (n + 1) * self.shellwidth,
                        mol.resname[i],
                        mol.resid[i],
                        mol.name[i],
                    )
                ]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )


def _shells(
    distances, mapping, shellcenters, numshells, shelledges, shellvol, symmetrical
):
    shellcenters = np.unique(mapping[:, 0]) if not symmetrical else np.unique(mapping)

    shellmetric = np.ones((np.size(distances, 0), len(shellcenters) * numshells)) * -1

    for i in range(len(shellcenters)):
        if symmetrical:
            cols = np.any(mapping == shellcenters[i], axis=1)
        else:
            cols = mapping[:, 0] == shellcenters[i]

        for e in range(len(shelledges) - 1):
            inshell = (distances[:, cols] > shelledges[e]) & (
                distances[:, cols] <= shelledges[e + 1]
            )
            shellmetric[:, (i * numshells) + e] = np.sum(inshell, axis=1) / shellvol[e]

    return shellmetric


import unittest


class _TestMetricShell(unittest.TestCase):
    def test_metricshell(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(
            path.join(home(dataDir="test-projections"), "trajectory", "filtered.pdb")
        )
        mol.read(path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc"))

        metr = MetricShell(
            "protein and name CA", "resname MOL and noh", periodic="selections"
        )
        data = metr.project(mol)

        refdata = np.load(
            path.join(home(dataDir="test-projections"), "metricshell", "refdata.npy")
        )

        assert np.allclose(data, refdata), "Shell density calculation is broken"

    def test_metricshell_simple(self):
        from moleculekit.molecule import Molecule
        import numpy as np

        mol = Molecule().empty(3)
        mol.name[:] = "CL"
        mol.resname[:] = "CL"
        mol.element[:] = "Cl"
        mol.resid[:] = np.arange(3)
        mol.coords = np.zeros((3, 3, 1), dtype=np.float32)

        mol.coords[0, :, 0] = [0, 0, 0]
        mol.coords[1, :, 0] = [0.5, 0, 0]
        mol.coords[2, :, 0] = [0, 1.5, 0]

        metr = MetricShell("name CL", "name CL", periodic=None)
        data = metr.project(mol)
        # fmt: off
        refdata = np.array([[0.01768388256576615, 0.0, 0.0, 0.0, 0.01768388256576615, 0.0, 0.0, 0.0, 0.01768388256576615, 0.0, 0.0, 0.0]])
        # fmt: on
        assert np.allclose(data, refdata)

        metr = MetricShell(
            "name CL", "name CL", numshells=2, shellwidth=1, periodic=None
        )
        data = metr.project(mol)
        refdata = np.array(
            [[0.23873241, 0.03410463, 0.23873241, 0.03410463, 0.0, 0.06820926]]
        )
        assert np.allclose(data, refdata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
