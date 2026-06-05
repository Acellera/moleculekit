# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
from moleculekit.periodictable import periodictable
from typing import TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)


class MetricGyration(Projection):
    """Creates a MetricGyration object that calculates the radius of gyration (ROG) of a molecule.

    Parameters
    ----------
    atomsel : str or np.ndarray
        Atom selection for the atoms whose ROG we want to calculate (a selection string, boolean mask, or integer index array).
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refmol : Molecule
        The reference Molecule to which we will align.
    trajalnsel : str or np.ndarray, optional
        Atom selection for the trajectories from which to align to the reference structure (a selection string, boolean mask, or integer index array).
        If it's None and a `refmol` is passed it will default to 'protein and name CA'.
    refalnsel : str or np.ndarray, optional
        Atom selection for `refmol` from which to align to the reference structure (a selection string, boolean mask, or integer index array). If None, it defaults to the
        same as `trajalnsel`.
    centersel : str or np.ndarray, optional
        Atom selection around which to wrap the simulation (a selection string, boolean mask, or integer index array).
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    pbc : bool
        Enable or disable coordinate wrapping based on periodic boundary conditions.

    Returns
    -------
    metr : MetricGyration object
    """

    def __init__(
        self,
        atomsel: str | np.ndarray | None,
        refmol: "Molecule | None" = None,
        trajalnsel: str | np.ndarray | None = None,
        refalnsel: str | np.ndarray | None = None,
        centersel: str | np.ndarray = "protein",
        pbc: bool = True,
    ):
        super().__init__()

        if atomsel is None:
            raise ValueError("Atom selection cannot be None")

        self._atomsel = atomsel
        self._trajalnsel = trajalnsel
        self._centersel = centersel
        self._pbc = pbc
        self._refmol = refmol

        if self._refmol is not None:
            if self._trajalnsel is None:
                self._trajalnsel = "protein and name CA"
            self._refalnsel = refalnsel if refalnsel is not None else self._trajalnsel
            self._refmol = refmol.copy()
            self._cache["refalnsel"] = self._refmol.atomselect(self._refalnsel)

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

        if "trajalnsel" in props:
            res["trajalnsel"] = None
            if self._trajalnsel is not None:
                res["trajalnsel"] = mol.atomselect(self._trajalnsel)
                if np.sum(res["trajalnsel"]) == 0:
                    raise RuntimeError("Alignment selection resulted in 0 atoms.")

        if "centersel" in props and self._pbc:
            res["centersel"] = mol.atomselect(self._centersel)
            if np.sum(res["centersel"]) == 0:
                raise RuntimeError("Centering selection resulted in 0 atoms.")
        return res

    def project(self, mol: "Molecule") -> np.ndarray:
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
        masses = getMolProp("masses")
        total_mass = np.sum(masses)

        coords = mol.coords[atomsel].copy()

        # Calculate center of mass
        com = np.sum(coords * masses[:, None, None], axis=0) / total_mass

        # Calculate radius of gyration
        coords -= com  # Remove COM from the coordinates
        sq_coords = coords**2
        sq = np.sum(sq_coords, axis=1)
        sq_x = np.sum(sq_coords[:, [1, 2]], axis=1)
        sq_y = np.sum(sq_coords[:, [0, 2]], axis=1)
        sq_z = np.sum(sq_coords[:, [0, 1]], axis=1)
        sq_radius = np.array([sq, sq_x, sq_y, sq_z])

        rog = np.sqrt(np.sum(masses[None, :, None] * sq_radius, axis=1) / total_mass).T

        return rog

    def getMapping(self, mol: "Molecule"):
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
        types = ["rog", "rog", "rog", "rog"]
        indexes = [atomidx, atomidx, atomidx, atomidx]
        description = [
            "Radius of gyration",
            "x component",
            "y component",
            "z component",
        ]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )
