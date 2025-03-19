# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricCoordinate(Projection):
    """Creates a MetricCoordinate object that calculates the atom coordinates from a set of trajectories.

    Parameters
    ----------
    atomsel : str
        Atom selection string for the atoms whose coordinates we want to calculate.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The reference Molecule to which we will align.
    trajalnsel : str, optional
        Atom selection string for the trajectories from which to align to the reference structure.
        If it's None and a `refmol` is passed it will default to 'protein and name CA'.
    refalnsel : str, optional
        Atom selection string for `refmol` from which to align to the reference structure. If None, it defaults to the
        same as `trajalnsel`.
    centersel : str, optional
        Atom selection string around which to wrap the simulation.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    metric : str
        Can be ["all", "com", "centroid"]
    groupsel : ['all', 'residue'], optional
        Group all atoms in `atomsel` to a single ('all' atoms in group) or multiple groups per residue ('residue').
    groupreduce : ['centroid', 'com'], optional
        The reduction to apply on `groupsel` if it is used. `centroid` will calculate the centroid coordinate of each group.
        `com` will calculate the coordinate of the center of mass of each group.
    pbc : bool
        Enable or disable coordinate wrapping based on periodic boundary conditions.

    Returns
    -------
    metr : MetricCoordinate object
    """

    def __init__(
        self,
        atomsel,
        refmol=None,
        trajalnsel=None,
        refalnsel=None,
        centersel="protein",
        groupsel=None,
        groupreduce="com",
        pbc=True,
    ):
        super().__init__()

        if atomsel is None:
            raise ValueError("Atom selection cannot be None")

        self._trajalnsel = trajalnsel
        self._centersel = centersel
        self._atomsel = atomsel
        self._pbc = pbc
        self._refmol = refmol
        self._groupsel = groupsel
        self._groupreduce = groupreduce

        if self._refmol is not None:
            if self._trajalnsel is None:
                self._trajalnsel = "protein and name CA"
            self._refalnsel = refalnsel if refalnsel is not None else self._trajalnsel
            self._refmol = refmol.copy()
            self._cache["refalnsel"] = self._refmol.atomselect(self._refalnsel)

    def _calculateMolProp(self, mol, props="all"):
        props = ("trajalnsel", "atomsel", "centersel") if props == "all" else props
        res = {}
        if "trajalnsel" in props:
            res["trajalnsel"] = None
            if self._trajalnsel is not None:
                res["trajalnsel"] = mol.atomselect(self._trajalnsel)
                if np.sum(res["trajalnsel"]) == 0:
                    raise RuntimeError("Alignment selection resulted in 0 atoms.")
        if "atomsel" in props:
            res["atomsel"] = mol.atomselect(self._atomsel)
            if np.sum(res["atomsel"]) == 0:
                raise RuntimeError("Atom selection resulted in 0 atoms.")
        if "centersel" in props and self._pbc:
            res["centersel"] = mol.atomselect(self._centersel)
            if np.sum(res["centersel"]) == 0:
                raise RuntimeError("Centering selection resulted in 0 atoms.")
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
        from moleculekit.periodictable import periodictable

        getMolProp = lambda prop: self._getMolProp(mol, prop)

        mol = mol.copy()
        if self._pbc:
            mol.wrap(getMolProp("centersel"))

        trajalnsel = getMolProp("trajalnsel")

        if trajalnsel is not None:
            if self._refmol is None:
                mol.align(trajalnsel)
            else:
                mol.align(
                    trajalnsel, refmol=self._refmol, refsel=getMolProp("refalnsel")
                )

        atomsel = getMolProp("atomsel")
        coords = mol.coords[atomsel]
        resids = mol.resid[atomsel]
        elem = mol.element[atomsel]

        if self._groupsel is not None:
            if self._groupsel == "all":
                groups = [range(coords.shape[0])]
            elif self._groupsel == "residue":
                groups = [np.where(resids == uq)[0] for uq in np.unique(resids)]
            else:
                raise RuntimeError(
                    "Invalid groupsel option. Can only be 'all' or 'residue'"
                )

            masses = np.array([periodictable[el].mass for el in elem], dtype=np.float32)
            masses = masses.reshape(-1, 1, 1)

            groupcoords = []
            for group in groups:
                if self._groupreduce == "centroid":
                    gcoords = coords[group].mean(axis=0)
                elif self._groupreduce == "com":
                    g_mass = masses[group]
                    total_mass = np.sum(g_mass[:, 0, 0])
                    gcoords = np.sum(coords[group] * g_mass, axis=0) / total_mass
                else:
                    raise RuntimeError(
                        "Invalid groupreduce option. Can onlye be 'centroid' or 'com'"
                    )
                groupcoords.append(gcoords[None, :, :])
            coords = np.vstack(groupcoords)

        coords = np.transpose(coords, axes=(2, 1, 0)).reshape(mol.numFrames, -1)

        return coords

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
        resids = mol.resid[atomidx]

        groups = []
        if self._groupsel == "all":
            groups = [atomidx]
        elif self._groupsel == "residue":
            groups = [atomidx[resids == uq] for uq in np.unique(resids)]

        types = []
        indexes = []
        description = []
        if self._groupsel is None:
            for xyz in ("X", "Y", "Z"):
                for i in atomidx:
                    types += ["coordinate"]
                    indexes += [i]
                    description += [
                        f"{xyz} coordinate of {mol.resname[i]} {mol.resid[i]} {mol.name[i]}"
                    ]
        else:
            for xyz in ("X", "Y", "Z"):
                for group in groups:
                    types += ["coordinate"]
                    indexes += [group]
                    description += [f"{xyz} {self._groupreduce} coordinate of group"]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )
