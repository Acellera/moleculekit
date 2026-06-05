# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


class Projection(abc.ABC):
    """Abstract base class for all trajectory projecting (Metric) classes.

    A projection maps the coordinates of a :class:`Molecule <moleculekit.molecule.Molecule>`
    onto a lower-dimensional representation (e.g. distances, dihedral angles, RMSD).
    Concrete subclasses (the ``Metric*`` classes) must implement :meth:`project` and
    :meth:`getMapping`. This class cannot be instantiated directly.
    """

    def __init__(self):
        self._cache = {}

    @abc.abstractmethod
    def project(self, mol: "Molecule"):
        """Projects a molecule onto the lower-dimensional representation.

        Subclasses must implement this method.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object to project.

        Returns
        -------
        data : np.ndarray
            An array containing the projected data, with one row per frame.
        """
        return

    @abc.abstractmethod
    def getMapping(self, mol: "Molecule"):
        """Returns the description of each projected dimension.

        Subclasses must implement this method.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object which will be used
            to calculate the descriptions of the projected dimensions.

        Returns
        -------
        map : :class:`DataFrame <pandas.core.frame.DataFrame>` object
            A DataFrame mapping each projected dimension (output column) to the atoms it
            describes, with columns describing the type, atom indexes and a description.
        """
        return

    @abc.abstractmethod
    def _calculateMolProp(self, mol, props="all"):
        return

    def _setCache(self, mol):
        resdict = self._calculateMolProp(mol)
        self._cache.update(resdict)

    def _getMolProp(self, mol, prop):
        if prop in self._cache:
            resdict = self._cache
        else:
            resdict = self._calculateMolProp(mol, [prop] if prop != "all" else "all")

        if prop == "all":
            return resdict
        else:
            return resdict[prop]

    def copy(self):
        """Produces a deep copy of the object

        Returns
        -------
        proj : :class:`Projection <moleculekit.projections.projection.Projection>` object
            A deep copy of this projection object.
        """
        from copy import deepcopy

        return deepcopy(self)
