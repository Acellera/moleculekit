# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#

from moleculekit.projections.projection import Projection
from moleculekit.util import ensurelist
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricDistance(Projection):
    """Creates a MetricDistance object

    Parameters
    ----------
    sel1 : str
        Atom selection string for the first set of atoms.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    sel2 : str
        Atom selection string for the second set of atoms. If sel1 != sel2, it will calculate inter-set distances.
        If sel1 == sel2, it will calculate intra-set distances.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    periodic : str
        If periodic distances should be calculated and between which elements. If set to "chains" it will only calculate
        periodic distances between different chains. If set to "selections" it will calculate periodic distances between
        the two selections. If set to None it will not calculate any periodic distances.
    groupsel1 : ['all','residue'], optional
        Group all atoms in `sel1` to the single closest/COM distance. Alternatively can calculate the closest/COM distance of a
        residue containing the atoms in sel1.
    groupsel2 : ['all','residue'], optional
        Same as groupsel1 but for `sel2`
    metric : ['distances','contacts'], optional
        Set to 'contacts' to calculate contacts instead of distances
    threshold : float, optional
        The threshold under which a distance is considered in contact. Units in Angstrom.
    truncate : float, optional
        Set all distances larger than `truncate` to `truncate`. Units in Angstrom.
    groupreduce1 : ['closest', 'com'], optional
        The reduction to apply on group 1 if `groupsel1` is used. `closest` will calculate the closest distance of
        group 1 to selection 2. `com` will calculate the distance of the center of mass of group 1 to selection 2.
    groupreduce2 : ['closest', 'com'], optional
        Same as `groupreduce1` but for group 2 if `groupsel2` is used.
    pairs : bool
        If set to True it will match atoms in sel1 to atoms in sel2 in their given order and only calculate
        distances of those pairs of atoms instead of all-against-all distances

    Returns
    -------
    proj : MetricDistance object

    Examples
    --------
    Calculate periodic distances between all protein CA atoms and all atoms of a ligand called MOL
    >>> metr = MetricDistance("protein and name CA", "resname MOL", periodic="selections")
    >>> data = metr.project(mol)

    Calculate the single periodic distance between the closest atom of the protein to the closest atom of the ligand
    >>> MetricDistance("protein", "resname MOL", "selections", groupsel1="all", groupsel2="all")

    Calculate the periodic distances between the closest atom of each protein residue to the single closest ligand atom
    >>> MetricDistance("protein", "resname MOL", "selections", groupsel1="residue", groupsel2="all")

    Calculate the periodic distance between the COM of the protein to the COM of the ligand
    >>> MetricDistance("protein", "resname MOL", "selections", groupsel1="all", groupsel2="all", groupreduce1="com", groupreduce2="com")

    Calculate the non-periodic distance between a ligand atom and a protein atom
    >>> MetricDistance("protein and name CA and resid 10", "resname MOL and name C7", periodic=None)

    Calculate the distance of two nucleic chains
    >>> MetricDistance("nucleic and chain A", "nucleic and chain B", periodic="chains")
    """

    def __init__(
        self,
        sel1,
        sel2,
        periodic,
        groupsel1=None,
        groupsel2=None,
        metric="distances",
        threshold=8,
        truncate=None,
        groupreduce1="closest",
        groupreduce2="closest",
        pairs=False,
    ):
        super().__init__()

        self.sel1 = sel1
        self.sel2 = sel2
        self.periodic = periodic
        if periodic is not None and periodic not in ["chains", "selections"]:
            raise RuntimeError(
                "Option `periodic` can only be None, 'chains' or 'selections'."
            )

        self.groupsel1 = groupsel1
        self.groupsel2 = groupsel2
        self.metric = metric
        self.threshold = threshold
        self.truncate = truncate
        self.groupreduce1 = groupreduce1
        self.groupreduce2 = groupreduce2
        self.pairs = pairs

    def _checkChains(self, mol, sel1, sel2):
        if np.array_equal(sel1, sel2):
            return
        if np.ndim(sel1) == 1:
            sel1 = sel1[None, :]
        if np.ndim(sel2) == 1:
            sel2 = sel2[None, :]
        sel1 = np.any(sel1, axis=0)
        sel2 = np.any(sel2, axis=0)
        chains_sel1 = mol.chain[sel1]
        chains_sel2 = mol.chain[sel2]
        if len(np.intersect1d(chains_sel1, chains_sel2)):
            logger.warning(
                "Atomselections sel1 and sel2 of MetricDistance contain atoms belonging to a common chain. "
                "Atoms within the same chain will not have periodic distances computed. "
                "Ensure that chains are properly defined in your topology file."
            )

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
        from moleculekit.projections.util import pp_calcDistances, get_reduced_distances

        getMolProp = lambda prop: self._getMolProp(mol, prop)
        sel1 = getMolProp("sel1")
        sel2 = getMolProp("sel2")
        if self.periodic == "chains":
            self._checkChains(mol, sel1, sel2)

        if np.ndim(sel1) == 1 and np.ndim(sel2) == 1:  # normal distances
            if self.pairs:
                raise RuntimeError("Pairs calculation not implemented without groups")
            metric = pp_calcDistances(
                mol,
                sel1,
                sel2,
                self.periodic,
                self.metric,
                self.threshold,
                truncate=self.truncate,
            )
        else:  # minimum distances by groups
            metric = get_reduced_distances(
                mol,
                sel1,
                sel2,
                self.periodic,
                self.metric,
                self.threshold,
                truncate=self.truncate,
                reduction1=self.groupreduce1,
                reduction2=self.groupreduce2,
                pairs=self.pairs,
            )

        return metric

    def _calculateMolProp(self, mol, props="all"):
        props = ("sel1", "sel2") if props == "all" else props
        res = {}
        if "sel1" in props:
            res["sel1"] = self._processSelection(mol, self.sel1, self.groupsel1)
        if "sel2" in props:
            res["sel2"] = self._processSelection(mol, self.sel2, self.groupsel2)
        return res

    def _processSelection(self, mol, sel, groupsel):
        # If user passes simple string selections or 1D array of ints or bools
        if isinstance(sel, str) or (
            isinstance(sel, np.ndarray)
            and sel.ndim == 1
            and (np.issubdtype(sel.dtype, np.integer) or sel.dtype == bool)
        ):
            if groupsel is None:
                sel = mol.atomselect(sel)
            elif groupsel == "all":
                sel = self._processMultiSelections(mol, [sel])
            elif groupsel == "residue":
                sel = self._groupByResidue(mol, sel)
            else:
                raise RuntimeError("Invalid groupsel argument")
        elif isinstance(sel, np.ndarray) or isinstance(
            sel, list
        ):  # If user passes his own sets of groups
            sel = self._processMultiSelections(mol, sel)
        else:
            raise RuntimeError(
                "Invalid atom selection. Either provide a string, a list of string, a 1D numpy array (int/bool) or a 2D numpy array for groups."
            )

        if np.sum(sel) == 0:
            raise RuntimeError("Selection returned 0 atoms")
        return sel

    def _processMultiSelections(self, mol, sel):
        newsel = np.zeros((len(sel), mol.numAtoms), dtype=bool)
        for s in range(len(sel)):
            if isinstance(sel[s], str):
                newsel[s, :] = mol.atomselect(sel[s])
            elif isinstance(sel[s], np.ndarray) and (
                np.issubdtype(sel[s].dtype, np.integer) or sel[s].dtype == bool
            ):
                newsel[s, sel[s]] = True
            else:
                raise RuntimeError("Invalid selection provided for groups")
        return newsel

    def _groupByResidue(self, mol, sel):
        import pandas as pd

        idx = mol.atomselect(sel, indexes=True)
        df = pd.DataFrame({"a": mol.resid[idx]})
        gg = df.groupby(by=df.a).groups  # Grouping by same resids

        newsel = np.zeros((len(gg), mol.numAtoms), dtype=bool)
        for i, res in enumerate(sorted(gg)):
            # Setting the selected indexes to True which correspond to the same residue
            newsel[i, idx[gg[res]]] = True
        return newsel

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
        sel1 = getMolProp("sel1")
        sel2 = getMolProp("sel2")

        if np.ndim(sel1) == 2:
            protatoms = []
            for i in range(sel1.shape[0]):
                protatoms.append(np.where(sel1[i, :] == True)[0])
        else:
            protatoms = np.where(sel1)[0]
        if np.ndim(sel2) == 2:
            ligatoms = []
            for i in range(sel2.shape[0]):
                ligatoms.append(np.where(sel2[i, :] == True)[0])
        else:
            ligatoms = np.where(sel2)[0]

        numatoms1 = len(protatoms)
        numatoms2 = len(ligatoms)

        prot_labels = [
            f"{mol.resname[i]} {mol.resid[i]} {mol.name[i]}" for i in protatoms
        ]

        types = []
        indexes = []
        description = []
        if np.array_equal(sel1, sel2):
            for i in range(numatoms1):
                atm1 = protatoms[i]
                for j in range(i + 1, numatoms1):
                    atm2 = protatoms[j]
                    desc = f"{self.metric[:-1]} between {prot_labels[i]} and {prot_labels[j]}"
                    types += [self.metric[:-1]]
                    indexes += [[atm1, atm2]]
                    description += [desc]
        else:
            lig_labels = [
                f"{mol.resname[i]} {mol.resid[i]} {mol.name[i]}" for i in ligatoms
            ]
            if not self.pairs:
                for i in range(numatoms1):
                    atm1 = protatoms[i]
                    for j in range(numatoms2):
                        atm2 = ligatoms[j]
                        desc = f"{self.metric[:-1]} between {prot_labels[i]} and {lig_labels[j]}"
                        types += [self.metric[:-1]]
                        indexes += [[atm1, atm2]]
                        description += [desc]
            else:
                for i in range(numatoms1):
                    atm1 = protatoms[i]
                    atm2 = ligatoms[i]
                    desc = f"{self.metric[:-1]} between {prot_labels[i]} and {lig_labels[i]}"
                    types += [self.metric[:-1]]
                    indexes += [[atm1, atm2]]
                    description += [desc]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )


class MetricSelfDistance(MetricDistance):
    def __init__(
        self,
        sel,
        groupsel=None,
        metric="distances",
        threshold=8,
        periodic=None,
        truncate=None,
    ):
        """Creates a MetricSelfDistance object

        Parameters
        ----------
        sel : str
            Atom selection string for which to calculate the self distance.
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
        groupsel : ['all','residue'], optional
            Group all atoms in `sel` to the single minimum distance. Alternatively can calculate the minimum distance
            of a residue containing the atoms in `sel`.
        metric : ['distances','contacts'], optional
            Set to 'contacts' to calculate contacts instead of distances
        threshold : float, optional
            The threshold under which a distance is considered in contact
        periodic : str
            See documentation of MetricDistance class
        truncate : float, optional
            Set all distances larger than `truncate` to `truncate`
        update :
            Not functional yet

        Returns
        -------
        proj : MetricDistance object
        """

        super().__init__(
            sel1=sel,
            sel2=sel,
            periodic=periodic,
            groupsel1=groupsel,
            groupsel2=groupsel,
            metric=metric,
            threshold=threshold,
            truncate=truncate,
        )


def contactVecToMatrix(vector, atomIndexes):
    from copy import deepcopy

    if np.ndim(vector) != 1:
        raise RuntimeError(
            "Please pass a 1D vector to the contactVecToMatrix function."
        )
    # Calculating the unique atom groups in the mapping
    uqAtomGroups = []
    atomIndexes = deepcopy(list(atomIndexes))
    for ax in atomIndexes:
        ax[0] = ensurelist(ax[0])
        ax[1] = ensurelist(ax[1])
        if ax[0] not in uqAtomGroups:
            uqAtomGroups.append(ax[0])
        if ax[1] not in uqAtomGroups:
            uqAtomGroups.append(ax[1])
    uqAtomGroups.sort(key=lambda x: x[0])  # Sort by first atom in each atom list
    num = len(uqAtomGroups)

    matrix = np.zeros((num, num), dtype=vector.dtype)
    mapping = np.ones((num, num), dtype=int) * -1
    for i in range(len(vector)):
        row = uqAtomGroups.index(atomIndexes[i][0])
        col = uqAtomGroups.index(atomIndexes[i][1])
        matrix[row, col] = vector[i]
        matrix[col, row] = vector[i]
        mapping[row, col] = i
        mapping[col, row] = i
    return matrix, mapping, uqAtomGroups


def reconstructContactMap(
    vector,
    mapping,
    truecontacts=None,
    plot=True,
    figsize=(7, 7),
    dpi=80,
    title=None,
    outfile=None,
    colors=None,
):
    """Plots a given vector as a contact map

    Parameters
    ----------
    vector : np.ndarray or list
        A 1D vector of contacts
    mapping : pd.DataFrame
        A pandas DataFrame which describes the dimensions of the projection
    truecontacts : np.ndarray or list
        A 1D vector of true contacts
    plot : bool
        To plot or not to plot
    figsize : tuple
        The size of the final plot in inches
    dpi : int
        Dots per inch
    outfile : str
        Path of file in which to save the plot

    Returns
    -------
    cm : np.ndarray
        The input vector converted into a 2D numpy array

    Examples
    --------
    >>> reconstructContactMap(contacts, mapping)
    To use it with distances instead of contacts pass ones as the concat vector
    >>> reconstructContactMap(np.ones(dists.shape, dtype=bool), mapping, colors=dists)
    """
    from copy import deepcopy
    from matplotlib import cm as colormaps

    if np.ndim(vector) != 1:
        raise RuntimeError(
            "Please pass a 1D vector to the reconstructContactMap function."
        )

    if truecontacts is None:
        truecontacts = np.zeros(len(vector), dtype=bool)
    if len(vector) != len(mapping):
        raise RuntimeError("Vector and map length must match.")

    # Checking if contacts or distances exist in the data
    contactidx = mapping.type == "contact"
    if not np.any(contactidx):
        contactidx = mapping.type == "distance"
        if not np.any(contactidx):
            raise RuntimeError(
                "No contacts or distances found in the MetricData object. Check the `.map` property of the object for a description of your projection."
            )

    # Creating the 2D contact maps
    cm, newmapping, uqAtomGroups = contactVecToMatrix(vector, mapping.atomIndexes)
    cmtrue, _, _ = contactVecToMatrix(truecontacts, mapping.atomIndexes)
    num = len(uqAtomGroups)

    if plot:
        from matplotlib import pylab as plt

        f = plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(
            cmtrue / 2,
            interpolation="none",
            vmin=0,
            vmax=1,
            aspect="equal",
            cmap="Greys",
        )  # /2 to convert to gray from black

        rows, cols = np.where(cm)
        colorbar = False
        if colors is None:
            truecms = vector & truecontacts
            colors = np.array(["r"] * len(vector), dtype=object)
            colors[truecms] = "#ffff00"
        elif isinstance(colors, np.ndarray) and isinstance(colors[0], float):
            mpbl = colormaps.ScalarMappable(cmap=colormaps.jet)
            mpbl.set_array(colors)
            colors = mpbl.to_rgba(colors)
            colorbar = True
        if len(colors) == len(vector):
            colors = colors[newmapping[rows, cols]]

        plt.scatter(rows, cols, s=figsize[0] * 5, marker="o", c=colors, lw=0)
        if colorbar:
            plt.colorbar(mpbl)

        ax = f.axes[0]
        # Major ticks
        ax.set_xticks(np.arange(0, num, 1))
        ax.set_yticks(np.arange(0, num, 1))

        # Labels for major ticks
        ax.set_xticklabels([x[0] for x in uqAtomGroups])
        ax.set_yticklabels(
            [x[0] for x in uqAtomGroups],
        )

        # Minor ticks
        ax.set_xticks(np.arange(-0.5, num, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num, 1), minor=True)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

        # Gridlines based on minor ticks
        ax.grid(which="minor", color="#969696", linestyle="-", linewidth=1)
        ax.tick_params(axis="both", which="both", length=0)
        plt.xlim([-0.5, num - 0.5])
        plt.ylim([-0.5, num - 0.5])
        plt.xlabel("Atom index")
        plt.ylabel("Atom index")
        if title:
            plt.title(title)
        if outfile is not None:
            plt.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
            plt.close()
        else:
            plt.show()
    return cm


import unittest
from moleculekit.home import home
import os


class _TestMetricDistance(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from os import path

        self.mol = Molecule(
            path.join(home(dataDir="test-projections"), "trajectory", "filtered.pdb")
        )
        self.mol_skipped = self.mol.copy()

        self.mol.read(
            path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc")
        )
        self.mol_skipped.read(
            path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc"),
            skip=10,
        )

    def test_contacts(self):
        metr = MetricDistance(
            "protein and name CA",
            "resname MOL and noh",
            periodic="selections",
            metric="contacts",
            threshold=8,
        )
        data = metr.project(self.mol)
        reffile = os.path.join(
            home(dataDir="test-projections"), "metricdistance", "distances.npy"
        )
        refdata = np.load(reffile) < 8

        assert np.allclose(data, refdata, atol=1e-3), "Contact calculation is broken"

    def test_atomselects(self):
        # String atomselection
        metr = MetricSelfDistance("protein and name CA", metric="contacts", threshold=8)
        data = metr.project(self.mol_skipped)
        mapping = metr.getMapping(self.mol_skipped)
        assert data.shape == (20, 38226) and mapping.shape == (38226, 3)

        # Integer index atomselection
        ca_atoms = self.mol_skipped.atomselect("protein and name CA", indexes=True)
        metr = MetricSelfDistance(ca_atoms, metric="contacts", threshold=8)
        data = metr.project(self.mol_skipped)
        mapping = metr.getMapping(self.mol_skipped)
        assert data.shape == (20, 38226) and mapping.shape == (38226, 3)

        # Boolean atomselection
        ca_atoms = self.mol_skipped.atomselect("protein and name CA")
        metr = MetricSelfDistance(ca_atoms, metric="contacts", threshold=8)
        data = metr.project(self.mol_skipped)
        mapping = metr.getMapping(self.mol_skipped)
        assert data.shape == (20, 38226) and mapping.shape == (38226, 3)

        # Manual group atomselection with integers
        ca_atoms = self.mol_skipped.atomselect("protein and name CA", indexes=True)
        metr = MetricSelfDistance(
            [ca_atoms[0::3], ca_atoms[1::3], ca_atoms[2::3]],
        )
        data = metr.project(self.mol_skipped)
        mapping = metr.getMapping(self.mol_skipped)
        assert data.shape == (20, 3) and mapping.shape == (3, 3)

        # Manual group atomselection with booleans
        ca_atoms = self.mol_skipped.atomselect("protein and name CA", indexes=True)
        ca_bools1 = self.mol_skipped.atomselect(
            f"index {' '.join(map(str, ca_atoms[0::3]))}"
        )
        ca_bools2 = self.mol_skipped.atomselect(
            f"index {' '.join(map(str, ca_atoms[1::3]))}"
        )
        ca_bools3 = self.mol_skipped.atomselect(
            f"index {' '.join(map(str, ca_atoms[2::3]))}"
        )
        metr = MetricSelfDistance([ca_bools1, ca_bools2, ca_bools3])
        newdata = metr.project(self.mol_skipped)
        mapping = metr.getMapping(self.mol_skipped)
        assert (
            newdata.shape == (20, 3)
            and mapping.shape == (3, 3)
            and np.array_equal(data, newdata)
        )

    def test_distances_trivial(self):
        from moleculekit.molecule import Molecule
        import numpy as np

        mol = Molecule().empty(3)
        mol.name[:] = "C"
        mol.element[:] = "C"
        mol.chain[:] = list(
            map(str, range(3))
        )  # If they are in the same chain, no wrapping is done for distances
        mol.coords = np.zeros(
            (3, 3, 2), dtype=np.float32
        )  # Make two frames so we check if the code works for nframes
        mol.coords[1, :, 0] = [3, 3, 3]
        mol.coords[2, :, 0] = [5, 5, 5]
        mol.coords[1, :, 1] = [7, 7, 7]
        mol.coords[2, :, 1] = [6, 6, 6]

        realdistances = np.linalg.norm(mol.coords[[1, 2], :, :], axis=1).T

        metr = MetricDistance("index 0", "index 1 2", metric="distances", periodic=None)
        data = metr.project(mol)
        assert np.allclose(
            data, realdistances
        ), "Trivial distance calculation is broken"

        # Test wrapped distances
        wrappedcoords = np.mod(mol.coords, 2)
        wrappedrealdistances = np.linalg.norm(wrappedcoords[[1, 2], :, :], axis=1).T

        mol.box = np.full((3, 2), 2, dtype=np.float32)  # Make box 2x2x2A large
        metr = MetricDistance(
            "index 0", "index 1 2", metric="distances", periodic="selections"
        )
        data = metr.project(mol)
        assert np.allclose(
            data, wrappedrealdistances
        ), "Trivial wrapped distance calculation is broken"

        # Test min distances
        metr = MetricDistance(
            "index 0",
            "index 1 2",
            metric="distances",
            periodic=None,
            groupsel1="all",
            groupsel2="all",
        )
        data = metr.project(mol)
        assert np.allclose(
            data.flatten(), np.min(realdistances, axis=1)
        ), "Trivial distance calculation is broken"

        # Test ordering
        mol = Molecule().empty(4)
        mol.name[:] = "C"
        mol.element[:] = "C"
        mol.chain[:] = list(
            map(str, range(4))
        )  # If they are in the same chain, no wrapping is done for distances
        mol.coords = np.zeros(
            (4, 3, 2), dtype=np.float32
        )  # Make two frames so we check if the code works for nframes
        mol.coords[1, :, 0] = [1, 1, 1]
        mol.coords[2, :, 0] = [3, 3, 3]
        mol.coords[3, :, 0] = [5, 5, 5]
        mol.coords[1, :, 1] = [1, 1, 1]
        mol.coords[2, :, 1] = [7, 7, 7]
        mol.coords[3, :, 1] = [6, 6, 6]

        realdistances = np.linalg.norm(
            mol.coords[[2, 3], :, :] - mol.coords[0], axis=1
        ).T
        realdistances = np.hstack(
            (
                realdistances,
                np.linalg.norm(mol.coords[[2, 3], :, :] - mol.coords[1], axis=1).T,
            )
        )

        metr = MetricDistance(
            "index 0 1", "index 2 3", metric="distances", periodic=None
        )
        data = metr.project(mol)
        assert np.allclose(
            data, realdistances
        ), "Trivial distance calculation has broken ordering"

    def test_distances(self):
        metr = MetricDistance(
            "protein and name CA",
            "resname MOL and noh",
            metric="distances",
            periodic="selections",
        )
        data = metr.project(self.mol)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"), "metricdistance", "distances.npy"
            )
        )
        assert np.allclose(data, refdata, atol=1e-3), "Distance calculation is broken"

    def test_mindistances(self):
        metr = MetricDistance(
            "protein and noh",
            "resname MOL and noh",
            periodic="selections",
            groupsel1="residue",
            groupsel2="all",
        )
        data = metr.project(self.mol)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"), "metricdistance", "mindistances.npy"
            )
        )
        assert np.allclose(
            data, refdata, atol=1e-3
        ), "Minimum distance calculation is broken"

    def test_mindistances_truncate(self):
        metr = MetricDistance(
            "protein and noh",
            "resname MOL and noh",
            periodic="selections",
            groupsel1="residue",
            groupsel2="all",
            truncate=3,
        )
        data = metr.project(self.mol)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"), "metricdistance", "mindistances.npy"
            )
        )
        assert np.allclose(
            data, np.clip(refdata, 0, 3), atol=1e-3
        ), "Minimum distance calculation is broken"

    def test_selfmindistance_manual(self):
        metr = MetricDistance(
            "protein and resid 1 to 50 and noh",
            "protein and resid 1 to 50 and noh",
            periodic=None,
            groupsel1="residue",
            groupsel2="residue",
        )
        data = metr.project(self.mol)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"),
                "metricdistance",
                "selfmindistance.npy",
            )
        )
        assert np.allclose(data, refdata, atol=1e-3), "Manual self-distance is broken"

    def test_selfmindistance_auto(self):
        metr = MetricSelfDistance(
            "protein and resid 1 to 50 and noh", groupsel="residue"
        )
        data = metr.project(self.mol)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"),
                "metricdistance",
                "selfmindistance.npy",
            )
        )
        assert np.allclose(
            data, refdata, atol=1e-3
        ), "Automatic self-distance is broken"

    def test_mindistances_skip(self):
        metr = MetricSelfDistance(
            "protein and resid 1 to 50 and noh", groupsel="residue"
        )
        data = metr.project(self.mol_skipped)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"),
                "metricdistance",
                "selfmindistance.npy",
            )
        )
        assert np.allclose(
            data, refdata[::10, :], atol=1e-3
        ), "Minimum distance calculation with skipping is broken"

    def test_reconstruct_contact_map(self):
        from moleculekit.util import tempname
        from moleculekit.molecule import Molecule
        import matplotlib

        matplotlib.use("Agg")

        mol = Molecule("1yu8")

        metr = MetricSelfDistance("protein and name CA", metric="contacts")
        data = metr.project(mol)[0]
        mapping = metr.getMapping(mol)

        tmpfile = tempname(suffix=".svg")
        cm, newmapping, uqAtomGroups = contactVecToMatrix(data, mapping.atomIndexes)
        reconstructContactMap(data, mapping, outfile=tmpfile)
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"),
                "metricdistance",
                "contactvectomatrix.npz",
            )
        )

        assert np.array_equal(refdata["cm"], cm)
        assert np.array_equal(refdata["newmapping"], newmapping)
        assert np.array_equal(refdata["uqAtomGroups"], uqAtomGroups)

    def test_description(self):
        metr = MetricDistance(
            "protein and noh", "resname MOL and noh", truncate=3, periodic="selections"
        )
        data = metr.project(self.mol)
        atomIndexes = metr.getMapping(self.mol).atomIndexes.values
        refdata = np.load(
            os.path.join(
                home(dataDir="test-projections"), "metricdistance", "description.npy"
            ),
            allow_pickle=True,
        )
        assert np.array_equal(refdata, atomIndexes)

    def test_periodicity(self):
        metr = MetricDistance(
            "protein and resid 1 to 20 and noh",
            "resname MOL and noh",
            periodic="selections",
        )
        data1 = metr.project(self.mol)
        metr = MetricDistance(
            "protein and resid 1 to 20 and noh",
            "resname MOL and noh",
            periodic="chains",
        )
        data2 = metr.project(self.mol)
        assert np.allclose(data1, data2)

        mol_tmp = self.mol.copy()
        mol_tmp.chain[:] = ""
        metr = MetricDistance(
            "protein and resid 1 to 20 and noh",
            "resname MOL and noh",
            periodic="chains",
        )
        data3 = metr.project(mol_tmp)
        assert not np.allclose(data1, data3)

    def test_com_distances(self):
        from moleculekit.molecule import Molecule
        from moleculekit.periodictable import periodictable

        mol = Molecule().empty(4)
        mol.coords = np.array(
            [
                [0, 0, 0],
                [-1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )[:, :, None]
        mol.element[:] = "H"
        mol.resid[:] = [0, 1, 1, 0]

        fixargs = {
            "periodic": None,
            "groupsel1": "all",
            "groupsel2": "all",
            "groupreduce1": "com",
            "groupreduce2": "com",
        }

        dist = MetricDistance("index 0", "index 1", **fixargs).project(mol)
        assert np.abs(dist[0][0] - 1) < 1e-5

        dist = MetricDistance("index 0 1", "index 2", **fixargs).project(mol)
        assert np.abs(dist[0][0] - 1.5) < 1e-5

        dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
        assert np.abs(dist[0][0] - 1) < 1e-5

        fixargs["groupsel1"] = "residue"
        dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
        assert np.allclose(dist, [[1, 1]])

        fixargs["groupsel1"] = "all"
        fixargs["groupreduce1"] = "closest"
        dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
        assert np.allclose(dist, [[1]])

        fixargs["groupsel1"] = "residue"
        fixargs["groupreduce1"] = "closest"
        dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
        assert np.allclose(dist, [[1, 1.4142135]])

        # Real test
        mol = Molecule("3ptb")

        sel1 = "protein"
        sel2 = "resname BEN"

        mol1 = mol.copy()
        mol1.filter(sel1)
        masses = np.array(
            [periodictable[el].mass for el in mol1.element], dtype=np.float32
        )
        com1 = np.sum(mol1.coords[:, :, 0] * masses[:, None], axis=0) / masses.sum()

        mol2 = mol.copy()
        mol2.filter(sel2)
        masses = np.array(
            [periodictable[el].mass for el in mol2.element], dtype=np.float32
        )
        com2 = np.sum(mol2.coords[:, :, 0] * masses[:, None], axis=0) / masses.sum()

        dist = MetricDistance(
            sel1,
            sel2,
            None,
            groupsel1="all",
            groupsel2="all",
            groupreduce1="com",
            groupreduce2="com",
        ).project(mol)

        assert (
            np.abs(np.linalg.norm(com1 - com2) - dist) < 1e-2
        ), f"{np.linalg.norm(com1 - com2)}, {dist}"

        dist = MetricDistance(
            sel1,
            sel2,
            None,
            groupsel1="all",
            groupsel2="all",
            groupreduce1="com",
            groupreduce2="closest",
        ).project(mol)
        assert np.abs(dist[0][0] - 8.978174) < 1e-5

        dist = MetricDistance(
            sel1,
            sel2,
            None,
            groupsel1="all",
            groupsel2="all",
            groupreduce1="closest",
            groupreduce2="com",
        ).project(mol)
        assert np.abs(dist[0][0] - 3.8286476) < 1e-5

        dist = MetricDistance(
            sel1,
            sel2,
            None,
            groupsel1="all",
            groupsel2="all",
            groupreduce1="closest",
            groupreduce2="closest",
        ).project(mol)
        assert np.abs(dist[0][0] - 2.8153415) < 1e-5

    def test_pair_distances(self):
        from moleculekit.molecule import Molecule

        sel1 = np.array([0, 1, 2]).reshape(-1, 1)
        sel2 = np.array([1, 2, 3]).reshape(-1, 1)
        mol = Molecule("3ptb")
        res = MetricDistance(sel1, sel2, None, pairs=True).project(mol)
        ref = np.linalg.norm(
            mol.coords[sel1.flatten(), :, 0] - mol.coords[sel2.flatten(), :, 0], axis=1
        )
        assert np.allclose(res, ref)
        mmp = MetricDistance(sel1, sel2, None, pairs=True).getMapping(mol)
        assert mmp.shape == (3, 3)

        res = MetricDistance(
            "residue 1 2",
            "residue 3 4",
            None,
            pairs=True,
            groupsel1="residue",
            groupsel2="residue",
        ).project(mol)
        res1 = MetricDistance("residue 1", "residue 3", None).project(mol)
        res2 = MetricDistance("residue 2", "residue 4", None).project(mol)

        assert np.allclose(res, [[res1.min(), res2.min()]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
