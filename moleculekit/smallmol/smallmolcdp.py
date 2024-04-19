import numpy as np
import logging
import unittest
import platform

logger = logging.getLogger(__name__)


class SmallMolCDP:
    def __init__(self, filename):
        import CDPL.Chem as Chem

        self._mol = Chem.BasicMolecule()
        reader = Chem.MoleculeReader(filename)
        reader.read(self._mol)

    def generateConformers(
        self, num_confs=1, timeout=3600, min_rmsd=0.5, e_window=20.0
    ):
        """Generate conformers for the molecule.

        Parameters
        ----------
        num_confs : int
            Number of conformers to generate. If set to 1 it will generate the lowest energy conformer.
        timeout : int
            Maximum allowed molecule processing time in seconds.
        min_rmsd : float
            Output conformer RMSD threshold.
        e_window : float
            Output conformer energy window.
        """
        import CDPL.Chem as Chem
        import CDPL.ConfGen as ConfGen

        status_to_str = {
            ConfGen.ReturnCode.SUCCESS: "success",
            ConfGen.ReturnCode.UNINITIALIZED: "uninitialized",
            ConfGen.ReturnCode.TIMEOUT: "max. processing time exceeded",
            ConfGen.ReturnCode.ABORTED: "aborted",
            ConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED: "force field setup failed",
            ConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED: "force field structure refinement failed",
            ConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET: "fragment library not available",
            ConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED: "fragment conformer generation failed",
            ConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT: "fragment conformer generation timeout",
            ConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED: "fragment already processed",
            ConfGen.ReturnCode.TORSION_DRIVING_FAILED: "torsion driving failed",
            ConfGen.ReturnCode.CONF_GEN_FAILED: "conformer generation failed",
        }
        # Clear out any existing conformers. TODO: There must be a nicer way to do this
        conf_gen = ConfGen.ConformerGenerator()
        conf_gen.setConformers(self._mol)

        if num_confs == 1:
            struct_gen = ConfGen.StructureGenerator()
            struct_gen.settings.timeout = timeout * 1000

            ConfGen.prepareForConformerGeneration(self._mol)
            status = struct_gen.generate(self._mol)
            # struct_gen.coordinates.toArray(False)
            if status != ConfGen.ReturnCode.SUCCESS:
                logger.warning(
                    f"Conformer generation failed with status: {status_to_str[status]}"
                )
                return

            struct_gen.setCoordinates(self._mol)

            Chem.setMDLDimensionality(self._mol, 3)
        else:
            conf_gen = ConfGen.ConformerGenerator()
            conf_gen.settings.timeout = timeout * 1000
            conf_gen.settings.minRMSD = min_rmsd
            conf_gen.settings.energyWindow = e_window
            conf_gen.settings.maxNumOutputConformers = num_confs

            ConfGen.prepareForConformerGeneration(self._mol)
            status = conf_gen.generate(self._mol)
            num_confs = conf_gen.getNumConformers()

            logger.info(
                f"Generated {num_confs} conformers with status '{status_to_str[status]}'"
            )

            if status not in (
                ConfGen.ReturnCode.SUCCESS,
                ConfGen.ReturnCode.TOO_MUCH_SYMMETRY,
            ):
                logger.warning(
                    f"Conformer generation failed with status: {status_to_str[status]}"
                )
                return
            conf_gen.setConformers(self._mol)

    @property
    def element(self):
        import CDPL.Chem as Chem

        return np.array(
            [a.getProperty(Chem.AtomProperty.SYMBOL) for a in self._mol.atoms],
            dtype=object,
        )

    @property
    def formalcharge(self):
        import CDPL.Chem as Chem

        return np.array(
            [a.getProperty(Chem.AtomProperty.FORMAL_CHARGE) for a in self._mol.atoms],
            dtype=object,
        )

    @property
    def charge(self):
        import CDPL.Chem as Chem

        return np.array(
            [
                a.getPropertyOrDefault(Chem.AtomProperty.MOL2_CHARGE, 0)
                for a in self._mol.atoms
            ],
            dtype=object,
        )

    @property
    def name(self):
        import CDPL.Chem as Chem

        return np.array(
            [
                a.getPropertyOrDefault(Chem.AtomProperty.MOL2_NAME, "")
                for a in self._mol.atoms
            ],
            dtype=object,
        )

    @property
    def bonds(self):
        return np.array(
            [[b.getBegin().index, b.getEnd().index] for b in self._mol.bonds]
        )

    @property
    def bondtype(self):
        import CDPL.Chem as Chem

        return np.array(
            [str(b.getProperty(Chem.BondProperty.ORDER)) for b in self._mol.bonds],
            dtype=object,
        )

    @property
    def atomtype(self):
        import CDPL.Chem as Chem

        return np.array(
            [
                a.getPropertyOrDefault(Chem.AtomProperty.SYBYL_TYPE, "")
                for a in self._mol.atoms
            ],
            dtype=object,
        )

    @property
    def coords(self):
        import CDPL.Chem as Chem

        if (
            Chem.AtomProperty.COORDINATES_3D_ARRAY
            in self._mol.atoms[0].getPropertyKeys()
        ):
            cc = [
                a.getProperty(Chem.AtomProperty.COORDINATES_3D_ARRAY).toArray(False)
                for a in self._mol.atoms
            ]
            return np.swapaxes(np.stack(cc).astype(np.float32), 1, 2).copy()
        else:
            cc = [
                a.getProperty(Chem.Entity3DProperty.COORDINATES_3D).toArray()
                for a in self._mol.atoms
            ]
            return np.vstack(cc).astype(np.float32)[:, :, None].copy()

    @property
    def numAtoms(self):
        return len(self._mol.atoms)

    @property
    def numFrames(self):
        return self.coords.shape[2]

    @property
    def ligname(self):
        import CDPL.Chem as Chem

        return self._mol.getPropertyOrDefault(Chem.MolecularGraphProperty.NAME, "LIG")

    def toMolecule(self):
        """
        Return a moleculekit.molecule.Molecule

        Returns
        -------
        mol: moleculekit.molecule.Molecule
            The moleculekit Molecule object

        """
        from moleculekit.molecule import Molecule

        class NoConformerError(Exception):
            pass

        # if self.numFrames == 0:
        #     raise NoConformerError(
        #         "No Conformers are found in the molecule. Generate at least one conformer."
        #     )

        mol = Molecule()
        mol.empty(self.numAtoms)
        mol.record[:] = "HETATM"
        mol.resname[:] = "MOL"
        mol.resid[:] = 1
        mol.coords = self.coords
        mol.name[:] = self.element
        mol.element[:] = self.element
        mol.formalcharge[:] = self.formalcharge
        mol.charge[:] = self.charge
        mol.box = np.zeros((3, self.numFrames), dtype=np.float32)
        mol.boxangles = np.zeros((3, self.numFrames), dtype=np.float32)
        mol.viewname = self.ligname
        mol.bonds = self.bonds
        mol.bondtype = self.bondtype
        mol.atomtype = self.atomtype
        return mol

    def view(self, *args, **kwargs):
        self.toMolecule().view(*args, **kwargs)


class _Test_SmallMolCDP(unittest.TestCase):
    @unittest.skipUnless(
        platform.system() == "Linux", "CDPKit only works on Linux. Skipping test."
    )
    def test_smallmolcdp(self):
        from moleculekit.home import home

        sm = SmallMolCDP(home("test-smallmol/Imatinib.sdf"))
        sm.generateConformers(10)
        assert sm.coords.shape == (68, 3, 10)

        mol = sm.toMolecule()
        assert mol.coords.shape == (68, 3, 10)
        assert mol.bonds.shape == (72, 2)
        assert mol.bondtype.shape == (72,)
        assert np.sum(mol.bondtype == "2") == 13


if __name__ == "__main__":
    unittest.main(verbosity=2)
