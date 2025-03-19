# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricSasa(Projection):
    """Calculate solvent accessible surface area of a molecule.

    Implementation and documentation taken from MDtraj shrake_rupley code.
    It returns the SASA in units of Angstrom squared.

    Parameters
    ----------
    sel : str
        Atom selection string for atoms or residues for which to calculate the SASA.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    filtersel : str
        Keep only the selected atoms in the system. All other atoms will be removed and will
        not contribute to the SASA calculation. Keep in mind that the SASA of an atom or residue
        is affected by the presence of other atoms around it so this will change the SASA of the remaining atoms.
    probeRadius : float
        The radius of the probe, in Angstrom.
    numSpherePoints : int
        The number of points representing the surface of each atom, higher values lead to more accuracy.
    mode : str
        In mode == 'atom', the extracted areas are resolved per-atom. In mode == 'residue', this is consolidated down
        to the per-residue SASA by summing over the atoms in each residue.

    Returns
    -------
    metr : MetricSasa object
    """

    def __init__(
        self,
        sel="protein",
        filtersel="all",
        probeRadius=1.4,
        numSpherePoints=960,
        mode="atom",
    ):
        super().__init__()

        self._probeRadius = probeRadius / 10  # Convert to nanometers
        self._numSpherePoints = numSpherePoints
        self._mode = mode
        self._sel = sel
        self._filtersel = filtersel

    def _calculateMolProp(self, mol, props="all"):
        props = (
            ("radii", "atom_mapping", "sel", "filtersel", "tokeep")
            if props == "all"
            else props
        )
        res = {}

        sel = mol.atomselect(self._sel)
        selidx = np.where(sel)[0]
        if "sel" in props:
            res["sel"] = sel

        filtersel = mol.atomselect(self._filtersel)
        filterselidx = np.where(filtersel)[0]
        if "filtersel" in props:
            res["filtersel"] = filtersel

        if len(np.setdiff1d(selidx, filterselidx)) != 0:
            raise RuntimeError(
                "Some atoms selected by `sel` are not selected by `filtersel` and thus would not be calculated. Make sure `sel` is a subset of `filtersel`."
            )

        if "tokeep" in props:
            filterselmod = filtersel.copy().astype(int)
            filterselmod[filterselmod == 0] = -1
            filterselmod[filtersel] = np.arange(np.count_nonzero(filtersel))
            res["tokeep"] = filterselmod[sel]

        if "radii" in props:
            from mdtraj.geometry.sasa import _ATOMIC_RADII

            atom_radii = np.vectorize(_ATOMIC_RADII.__getitem__)(mol.element[filtersel])
            res["radii"] = np.array(atom_radii, np.float32) + self._probeRadius

        if "atom_mapping" in props:
            if self._mode == "atom":
                res["atom_mapping"] = np.arange(np.sum(filtersel), dtype=np.int32)
            elif self._mode == "residue":
                from moleculekit.util import sequenceID

                res["atom_mapping"] = sequenceID(
                    (mol.resid[filtersel], mol.chain[filtersel], mol.segid[filtersel])
                ).astype(np.int32)
            else:
                raise ValueError(
                    f'mode must be one of "residue", "atom". "{self._mode}" supplied'
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
        radii = getMolProp("radii")
        atom_mapping = getMolProp("atom_mapping")
        filtersel = getMolProp("filtersel")
        sel = getMolProp("sel")
        tokeep = getMolProp("tokeep")
        tokeep = np.unique(atom_mapping[tokeep])

        xyz = np.swapaxes(
            np.swapaxes(np.atleast_3d(mol.coords[filtersel, :, :]), 1, 2), 0, 1
        )
        xyz = np.array(xyz.copy(), dtype=np.float32) / 10  # converting to nm

        try:
            from mdtraj.geometry._geometry import _sasa as sasa
        except ImportError:
            raise ImportError(
                "To calculate SASA you need to install mdtraj with `conda install mdtraj -c conda-forge`"
            )

        out = np.full((mol.numFrames, atom_mapping.max() + 1), -1, dtype=np.float32)
        out[:, tokeep] = 0

        sasa(
            xyz,
            radii,
            int(self._numSpherePoints),
            atom_mapping,
            sel[filtersel].astype(np.int32).copy(),
            out,
        )
        out = out[:, tokeep] * 100  # Convert from square nm to square A
        assert not np.any(out == -1), "Some atoms are not excluded"
        return out

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
        getMolProp = lambda prop: self._getMolProp(mol, prop)
        atom_mapping = getMolProp("atom_mapping")
        tokeep = getMolProp("tokeep")
        atomsel = getMolProp("sel")

        if self._mode == "atom":
            atomidx = np.where(atomsel)[0]
        elif self._mode == "residue":
            atom_mapping = atom_mapping[tokeep]
            _, firstidx = np.unique(atom_mapping, return_index=True)
            atomidx = np.where(atomsel)[0][firstidx]
        else:
            raise ValueError(
                f'mode must be one of "residue", "atom". "{self._mode}" supplied'
            )

        from pandas import DataFrame

        types = []
        indexes = []
        description = []
        for i in atomidx:
            types += ["SASA"]
            indexes += [i]
            description += [f"SASA of {mol.resname[i]} {mol.resid[i]} {mol.name[i]}"]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )
