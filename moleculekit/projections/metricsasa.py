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

        out = np.zeros((mol.numFrames, atom_mapping.max() + 1), dtype=np.float32)
        sasa(xyz, radii, int(self._numSpherePoints), atom_mapping, out)
        return out[:, tokeep] * 100  # Convert from square nm to square A

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


import unittest


class _TestMetricSasa(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(
            path.join(home(dataDir="test-projections"), "trajectory", "filtered.pdb")
        )
        mol.read(path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc"))
        mol.dropFrames(keep=[0, 1])  # Keep only two frames because it's super slow
        self.mol = mol

    def test_sasa_atom(self):
        from os import path
        from moleculekit.home import home

        metr = MetricSasa(mode="atom")
        sasaA = metr.project(self.mol.copy())
        sasaA_ref = np.load(
            path.join(home(dataDir="test-projections"), "metricsasa", "sasa_atom.npy")
        )
        assert np.allclose(sasaA, sasaA_ref, atol=0.1), f"Failed with max diff {np.abs(sasaA-sasaA_ref).max()}"

    def test_sasa_residue(self):
        from os import path
        from moleculekit.home import home

        metr = MetricSasa(mode="residue")
        sasaR = metr.project(self.mol.copy())
        sasaR_ref = np.load(
            path.join(
                home(dataDir="test-projections"), "metricsasa", "sasa_residue.npy"
            )
        )
        assert np.allclose(sasaR, sasaR_ref, atol=0.3), f"Failed with max diff {np.abs(sasaR-sasaR_ref).max()}"

    def test_set_diff_error(self):
        try:
            metr = MetricSasa(mode="atom", sel="index 3000", filtersel="not index 3000")
            metr._calculateMolProp(self.mol.copy())
        except RuntimeError:
            print("Correctly threw a runtime error for bad selections")
        else:
            raise AssertionError(
                "This should throw an error as the selected atom does not exist in the filtered system"
            )

    def test_selection_and_filtering(self):
        metr = MetricSasa(mode="atom")
        sasaA_ref = metr.project(self.mol.copy())

        metr = MetricSasa(
            mode="atom", sel="index 20"
        )  # Get just the SASA of the 20th atom
        sasaA = metr.project(self.mol.copy())
        assert np.allclose(
            sasaA, sasaA_ref[:, [20]], atol=3e-3
        ), "SASA atom selection failed to give same results as without selection"

        metr = MetricSasa(
            mode="atom", sel="index 20", filtersel="index 20"
        )  # Get just the SASA of the 20th atom, remove all else
        sasaA = metr.project(self.mol.copy())
        assert not np.allclose(
            sasaA, sasaA_ref[:, [20]], atol=3e-3
        ), "SASA filtering gave same results as without filtering. Bad."

    def test_mappings(self):
        metr = MetricSasa(mode="atom")
        mapping = metr.getMapping(self.mol)
        assert np.array_equal(mapping.atomIndexes, np.arange(4480))

        # fmt: off
        metr = MetricSasa(mode='residue')
        mapping = metr.getMapping(self.mol)
        ref = np.array([0,   17,   27,   39,   49,   56,   75,   99,  113,  132,  152, 167,  189,  211,  222,  241,  256,  268,  290,  304,  319,  343,
                        358,  377,  396,  411,  422,  443,  463,  484,  500,  515,  522, 533,  545,  555,  570,  589,  596,  613,  624,  638,  662,  679,
                        695,  712,  731,  751,  775,  797,  808,  822,  839,  854,  873, 892,  902,  909,  919,  930,  949,  968,  979,  991, 1015, 1039,
                        1055, 1074, 1088, 1098, 1108, 1125, 1135, 1154, 1173, 1194, 1208, 1222, 1246, 1258, 1280, 1294, 1314, 1328, 1343, 1357, 1369, 1388,
                        1407, 1423, 1447, 1466, 1473, 1495, 1512, 1523, 1547, 1561, 1585, 1606, 1621, 1645, 1659, 1678, 1693, 1715, 1734, 1745, 1762, 1781,
                        1796, 1818, 1837, 1858, 1877, 1894, 1908, 1932, 1953, 1967, 1991, 2015, 2030, 2044, 2063, 2075, 2099, 2111, 2130, 2140, 2159, 2176,
                        2198, 2217, 2239, 2261, 2275, 2291, 2301, 2321, 2332, 2344, 2365, 2384, 2401, 2415, 2431, 2441, 2460, 2474, 2486, 2510, 2525, 2539,
                        2549, 2559, 2570, 2589, 2608, 2625, 2635, 2642, 2663, 2685, 2692, 2716, 2732, 2746, 2753, 2777, 2784, 2798, 2817, 2839, 2854, 2861,
                        2878, 2892, 2903, 2919, 2938, 2955, 2971, 2987, 3001, 3020, 3034, 3053, 3069, 3084, 3108, 3122, 3138, 3148, 3170, 3182, 3193, 3207,
                        3231, 3250, 3274, 3293, 3307, 3319, 3333, 3350, 3370, 3380, 3390, 3397, 3418, 3440, 3454, 3466, 3481, 3488, 3510, 3534, 3541, 3553,
                        3563, 3573, 3588, 3595, 3607, 3618, 3625, 3632, 3646, 3666, 3682, 3699, 3721, 3732, 3746, 3766, 3780, 3794, 3818, 3842, 3863, 3880,
                        3897, 3904, 3923, 3939, 3950, 3974, 3981, 3996, 4003, 4013, 4025, 4049, 4061, 4068, 4090, 4111, 4118, 4138, 4159, 4173, 4190, 4206,
                        4226, 4250, 4269, 4291, 4313, 4337, 4356, 4373, 4395, 4411, 4430, 4442, 4459])
        # fmt: on
        assert np.array_equal(mapping.atomIndexes, ref)


if __name__ == "__main__":
    unittest.main(verbosity=2)
