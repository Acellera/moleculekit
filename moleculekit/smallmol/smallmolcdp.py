import numpy as np
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)


class SmallMolCDP:
    """
    Class to manipulate small molecule structures backed by the Chemical Data Processing Library (CDPKit).

    The molecule is stored internally as a ``CDPL.Chem.BasicMolecule`` and atom/bond properties are
    exposed as numpy arrays. This class is primarily useful for CDPKit-based conformer generation.

    Parameters
    ----------
    filename : str or moleculekit.molecule.Molecule
        Either the path to a molecule file readable by CDPKit, or a moleculekit
        moleculekit.molecule.Molecule object, which is written to a temporary SDF file and read back.
    """

    def __init__(self, filename: 'str | "Molecule"'):
        from moleculekit.molecule import Molecule
        import CDPL.Chem as Chem
        import tempfile
        import os

        self._mol = Chem.BasicMolecule()

        with tempfile.TemporaryDirectory() as tmpdir:
            if isinstance(filename, Molecule):
                filename.write(os.path.join(tmpdir, "tmp.sdf"))
                filename = os.path.join(tmpdir, "tmp.sdf")

            reader = Chem.MoleculeReader(filename)
            reader.read(self._mol)

    def generateConformers(
        self,
        num_confs: int = 1,
        timeout: int = 3600,
        min_rmsd: float = 0.5,
        e_window: float = 20.0,
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
    def element(self) -> np.ndarray:
        """The element symbol of each atom.

        Returns
        -------
        element : np.ndarray
            An object array with the element symbol of each atom
        """
        import CDPL.Chem as Chem

        return np.array(
            [a.getProperty(Chem.AtomProperty.SYMBOL) for a in self._mol.atoms],
            dtype=object,
        )

    @property
    def formalcharge(self) -> np.ndarray:
        """The formal charge of each atom.

        Returns
        -------
        formalcharge : np.ndarray
            An object array with the formal charge of each atom
        """
        import CDPL.Chem as Chem

        return np.array(
            [a.getProperty(Chem.AtomProperty.FORMAL_CHARGE) for a in self._mol.atoms],
            dtype=object,
        )

    @property
    def charge(self) -> np.ndarray:
        """The partial charge of each atom.

        Returns the MOL2 partial charge of each atom, defaulting to 0 when not set.

        Returns
        -------
        charge : np.ndarray
            An object array with the partial charge of each atom
        """
        import CDPL.Chem as Chem

        return np.array(
            [
                a.getPropertyOrDefault(Chem.AtomProperty.MOL2_CHARGE, 0)
                for a in self._mol.atoms
            ],
            dtype=object,
        )

    @property
    def name(self) -> np.ndarray:
        """The name of each atom.

        Returns the MOL2 atom name of each atom, defaulting to an empty string when not set.

        Returns
        -------
        name : np.ndarray
            An object array with the name of each atom
        """
        import CDPL.Chem as Chem

        return np.array(
            [
                a.getPropertyOrDefault(Chem.AtomProperty.MOL2_NAME, "")
                for a in self._mol.atoms
            ],
            dtype=object,
        )

    @property
    def bonds(self) -> np.ndarray:
        """The bonds of the molecule as pairs of atom indices.

        Returns
        -------
        bonds : np.ndarray
            An array of shape (nbonds, 2) with the begin and end atom indices of each bond
        """
        return np.array(
            [[b.getBegin().index, b.getEnd().index] for b in self._mol.bonds]
        )

    @property
    def bondtype(self) -> np.ndarray:
        """The bond order of each bond as a string.

        Returns
        -------
        bondtype : np.ndarray
            An object array with the bond order of each bond
        """
        import CDPL.Chem as Chem

        return np.array(
            [str(b.getProperty(Chem.BondProperty.ORDER)) for b in self._mol.bonds],
            dtype=object,
        )

    @property
    def atomtype(self) -> np.ndarray:
        """The Sybyl atom type of each atom.

        Returns the Sybyl atom type of each atom, defaulting to an empty string when not set.

        Returns
        -------
        atomtype : np.ndarray
            An object array with the Sybyl atom type of each atom
        """
        import CDPL.Chem as Chem

        return np.array(
            [
                a.getPropertyOrDefault(Chem.AtomProperty.SYBYL_TYPE, "")
                for a in self._mol.atoms
            ],
            dtype=object,
        )

    @property
    def coords(self) -> np.ndarray:
        """The atom coordinates of the molecule.

        Returns
        -------
        coords : np.ndarray
            A float32 array of shape (natoms, 3, nframes) with the coordinates of each atom for each
            conformer
        """
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
    def numAtoms(self) -> int:
        """The number of atoms in the molecule.

        Returns
        -------
        numatoms : int
            The number of atoms
        """
        return len(self._mol.atoms)

    @property
    def numFrames(self) -> int:
        """The number of conformers (frames) of the molecule.

        Returns
        -------
        numframes : int
            The number of conformers
        """
        return self.coords.shape[2]

    @property
    def ligname(self) -> str:
        """The ligand name of the molecule.

        Returns the molecule's name property, defaulting to ``"LIG"`` when not set.

        Returns
        -------
        ligname : str
            The ligand name
        """
        import CDPL.Chem as Chem

        return self._mol.getPropertyOrDefault(Chem.MolecularGraphProperty.NAME, "LIG")

    def toMolecule(self) -> "Molecule":
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
        """
        Visualizes the molecule.

        The molecule is converted to a moleculekit.molecule.Molecule and all arguments are forwarded
        to its ``view`` method.
        """
        self.toMolecule().view(*args, **kwargs)
