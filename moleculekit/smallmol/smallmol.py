# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType
from collections import defaultdict
from moleculekit.smallmol.util import _depictMol, convertToString
import logging

logger = logging.getLogger(__name__)


_bondtypes_IdxToType = BondType.values
_bondtypes_TypeToString = defaultdict(lambda: "un")
_bondtypes_TypeToString.update(
    {
        BondType.SINGLE: "1",
        BondType.DOUBLE: "2",
        BondType.TRIPLE: "3",
        BondType.QUADRUPLE: "4",
        BondType.QUINTUPLE: "5",
        BondType.HEXTUPLE: "6",
        BondType.AROMATIC: "ar",
    }
)
_bondtypes_FullStringToType = {
    "AROMATIC": BondType.AROMATIC,
    "SINGLE": BondType.SINGLE,
    "DOUBLE": BondType.DOUBLE,
    "TRIPLE": BondType.TRIPLE,
    "QUADRUPLE": BondType.QUADRUPLE,
    "QUINTUPLE": BondType.QUINTUPLE,
    "HEXTUPLE": BondType.HEXTUPLE,
}
_bondtypes_StringToType = {val: key for key, val in _bondtypes_TypeToString.items()}

_hybridizations_IdxToType = HybridizationType.values
_hybridizations_StringToType = {
    "S": HybridizationType.S,
    "SP": HybridizationType.SP,
    "SP2": HybridizationType.SP2,
    "SP3": HybridizationType.SP3,
}


class SmallMol(object):
    """
    Class to manipulate small molecule structures

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol or filename or smile or moleculekit.smallmol.smallmol.SmallMol
        (i) Rdkit molecule or (ii) Location of molecule file (".pdb"/".mol2") or (iii) a smile string or iv) another
        SmallMol object or v) moleculekit.molecule.Molecule object
    ignore_errors: bool
        If True, errors will not be raised.
    force_reading: bool
        If True, and the mol provided is not accepted, the molecule will be initially converted into sdf
    fixHs: bool
        If True, the missing hydrogens are assigned, the others are correctly assinged into the graph of the molecule
    removeHs: bool
        If True, remove the hydrogens

    Examples
    --------
    >>> import os
    >>> from moleculekit.smallmol.smallmol import SmallMol
    >>> SmallMol('CCO')  # doctest: +SKIP
    >>> SmallMol('ligand.pdb', fixHs=False, removeHs=True )  # doctest: +SKIP
    >>> sm = SmallMol('benzamidine.mol2')
    >>> print(sm)                                     # doctest: +ELLIPSIS
    SmallMol with 18 atoms and 1 conformers
    Atom field - bondtype
    Atom field - charge
    ...

    .. rubric:: Methods
    .. autoautosummary:: moleculekit.smallmol.smallmol.SmallMol
       :methods:
    .. rubric:: Attributes
    .. autoautosummary:: moleculekit.smallmol.smallmol.SmallMol
       :attributes:
    """

    _atom_fields = [
        "idx",
        "name",
        "charge",
        "formalcharge",
        "element",
        "chiral",
        "hybridization",
        "neighbors",
        "bondtype",
        "coords",
        "chiraltags",
        "virtualsite",
    ]

    _mol_fields = ["ligname", "_mol"]

    _dtypes = {
        "idx": int,
        "name": object,
        "charge": np.float32,
        "formalcharge": int,
        "element": object,
        "chiral": object,
        "hybridization": int,
        "neighbors": object,
        "bondtype": object,
        "coords": np.float32,
        "chiraltags": object,
        "bonds": np.uint32,
        "virtualsite": bool,
    }

    def __init__(
        self,
        mol,
        ignore_errors=False,
        force_reading=False,
        fixHs=True,
        removeHs=False,
        verbose=True,
        sanitize=True,
        _logger=True,
        **kwargs,
    ):
        self._frame = 0
        _logger = _logger & verbose

        self._mol = self._initializeMolObj(
            mol, force_reading, ignore_errors, sanitize, _logger, **kwargs
        )
        if sanitize:
            if _logger:
                logger.info("Sanitizing molecule")
            self.sanitize()
        if removeHs:
            if _logger:
                logger.info("Removing hydrogens (removeHs=True)")
            self.removeHs()
        if fixHs:
            if _logger:
                logger.info("Adding any missing hydrogens (fixHs=True)")
            self.addHs(addCoords=True)

    def _initializeMolObj(
        self, mol, force_reading, ignore_errors, sanitize, _logger, **kwargs
    ):
        """
        Read the input and tries to convert it into a rdkit.Chem.rdchem.Mol obj

        Parameters
        ----------
        mol: str or rdkit.Chem.rdchem.Mol or moleculekit.smallmol.smallmol.SmallMol
            i) rdkit.Chem.rdchem.Mol ii) The path to the pdb/mol2 to load iii) The smile string iv) SmallMol object
            v) moleculekit.molecule.nolecule.Molecule
        force_reading: bool
        If the mol provided is not accepted, the molecule will be initially converted into sdf

        Returns
        -------
        _mol: rdkit.Chem.Molecule object
            The rdkit molecule
        smallMol: moleculekit.smallmol.smallmol.SmallMol
            The smallMol object if SmallMol was passed
        """
        from moleculekit.molecule import Molecule
        from moleculekit.tools.obabel_tools import openbabelConvert

        _mol = None
        if isinstance(mol, Chem.Mol):
            _mol = mol
        elif isinstance(mol, Molecule):
            _mol = mol.toRDKitMol(
                sanitize=kwargs.get("sanitize", False),
                kekulize=kwargs.get("kekulize", False),
                assignStereo=kwargs.get("assignStereo", True),
                _logger=_logger,
            )
        elif isinstance(mol, str):
            if os.path.isfile(mol):
                name_suffix = os.path.splitext(mol)[-1]
                # load mol2 file
                if name_suffix == ".mol2":
                    _mol = Chem.MolFromMol2File(mol, removeHs=False, sanitize=False)
                # load pdb file
                elif name_suffix == ".pdb":
                    _mol = Chem.MolFromPDBFile(mol, removeHs=False, sanitize=False)
                # load single-molecule sdf file
                elif name_suffix == ".sdf":
                    sms = Chem.SDMolSupplier(mol, removeHs=False, sanitize=False)
                    if len(sms) != 1:
                        logger.warning(
                            f"More than one molecules found in {mol}. SmallMol will only read the first. If you want to read them all use the SmallMolLib class."
                        )
                    _mol = sms[0]
                # if the file failed to be loaded and 'force_reading' = True, file convert to sdf and than loaded
                if _mol is None and force_reading:
                    logger.warning(f"Reading {mol} with force_reading procedure")
                    sdf = openbabelConvert(mol, name_suffix, "sdf")
                    _mol = Chem.SDMolSupplier(sdf, removeHs=False)[0]
                    os.remove(sdf)
            else:
                # assuming it is a smile
                psmile = Chem.SmilesParserParams()
                psmile.removeHs = False
                psmile.sanitize = sanitize
                _mol = Chem.MolFromSmiles(mol, psmile)

        if _mol is None and not ignore_errors:
            if isinstance(mol, str):
                frerr = (
                    " Try by setting the force_reading option as True."
                    if not force_reading
                    else ""
                )
                raise ValueError(f"Failed to read file {mol}.{frerr}")
            elif isinstance(mol, Molecule):
                raise ValueError("Failed converting Molecule to SmallMol")
            else:
                raise RuntimeError(f"Failed reading molecule {mol}.")

        return _mol

    @property
    def _idx(self):
        return np.array(
            [a.GetIdx() for a in self._mol.GetAtoms()], dtype=SmallMol._dtypes["idx"]
        )

    @property
    def _element(self):
        return np.array(
            [a.GetSymbol() for a in self._mol.GetAtoms()],
            dtype=SmallMol._dtypes["element"],
        )

    @property
    def _name(self):
        return np.array(
            [
                (
                    a.GetProp("_Name")
                    if (a.HasProp("_Name") and a.GetProp("_Name") != "")
                    else f"{a.GetSymbol()}{a.GetIdx()}"
                )
                for a in self._mol.GetAtoms()
            ],
            dtype=SmallMol._dtypes["name"],
        )

    @property
    def _charge(self):
        return np.array(
            [
                (
                    a.GetPropsAsDict()["_TriposPartialCharge"]
                    if a.HasProp("_TriposPartialCharge")
                    else 0.0
                )
                for a in self._mol.GetAtoms()
            ],
            dtype=SmallMol._dtypes["charge"],
        )

    @property
    def _formalcharge(self):
        return np.array(
            [a.GetFormalCharge() for a in self._mol.GetAtoms()],
            dtype=SmallMol._dtypes["formalcharge"],
        )

    @property
    def _chiral(self):
        return np.array(
            [
                a.GetPropsAsDict()["_CIPCode"] if a.HasProp("_CIPCode") else ""
                for a in self._mol.GetAtoms()
            ],
            dtype=SmallMol._dtypes["chiral"],
        )

    @property
    def _hybridization(self):
        return np.array(
            [int(a.GetHybridization()) for a in self._mol.GetAtoms()],
            dtype=SmallMol._dtypes["hybridization"],
        )

    @property
    def _neighbors(self):
        return np.array(
            [[na.GetIdx() for na in a.GetNeighbors()] for a in self._mol.GetAtoms()],
            dtype=SmallMol._dtypes["neighbors"],
        )

    @property
    def _neighborsbondtype(self):
        return np.array(
            [
                [
                    self._mol.GetBondBetweenAtoms(int(a), int(n)).GetBondType()
                    for n in neighs
                ]
                for a, neighs in enumerate(self._neighbors)
            ],
            dtype=SmallMol._dtypes["bondtype"],
        )

    @property
    def _bonds(self):
        bonds = [
            [bo.GetBeginAtomIdx(), bo.GetEndAtomIdx()] for bo in self._mol.GetBonds()
        ]
        if len(bonds):
            return np.vstack(bonds).astype(SmallMol._dtypes["bonds"])
        else:
            return np.zeros((0, 2), dtype=SmallMol._dtypes["bonds"])

    @property
    def _bondtype(self):
        return np.array(
            [
                _bondtypes_TypeToString[
                    self._mol.GetBondBetweenAtoms(int(b1), int(b2)).GetBondType()
                ]
                for b1, b2 in self._bonds
            ],
            dtype=SmallMol._dtypes["bondtype"],
        )

    @property
    def _chiraltags(self):
        return np.array(
            [a.GetChiralTag() for a in self._mol.GetAtoms()],
            dtype=SmallMol._dtypes["chiraltags"],
        )

    @property
    def ligname(self):
        return self._mol.GetProp("_Name") if self._mol.HasProp("_Name") else "UNK"

    @property
    def _resname(self):
        return np.array(
            [self.ligname.strip() for _ in range(self.numAtoms)], dtype=object
        )

    @property
    def _coords(self):
        if self._mol.GetNumConformers() != 0:
            return np.array(
                np.stack(
                    [cc.GetPositions() for cc in self._mol.GetConformers()], axis=2
                ),
                dtype=SmallMol._dtypes["coords"],
            )
        else:
            return np.zeros((self.numAtoms, 3, 0), dtype=SmallMol._dtypes["coords"])

    @property
    def _atomtype(self):
        return np.array(
            [
                (
                    a.GetPropsAsDict()["_TriposAtomType"]
                    if a.HasProp("_TriposAtomType")
                    else ""
                )
                for a in self._mol.GetAtoms()
            ],
            dtype=object,
        )

    @property
    def _virtualsite(self):
        # TODO: Not implemented yet. Not sure if rdkit supports virtual sites
        return np.zeros(self.numAtoms, dtype=SmallMol._dtypes["virtualsite"])

    @property
    def frame(self):
        if self._frame < 0 or self._frame >= self.numFrames:
            raise RuntimeError("frame out of range")
        return self._frame

    @frame.setter
    def frame(self, value):
        if value < 0 or ((self.numFrames != 0) and (value >= self.numFrames)):
            raise RuntimeError(
                f"Frame index out of range. Molecule contains {self.numFrames} frame(s). Frames are 0-indexed."
            )
        self._frame = value

    # Dummy properties for compatibility with Molecule
    @property
    def _serial(self):
        return np.arange(self.numAtoms, dtype=np.int32)

    @property
    def _resid(self):
        return np.ones(self.numAtoms, dtype=np.int32)

    @property
    def _occupancy(self):
        return np.zeros(self.numAtoms, dtype=np.float32)

    @property
    def _beta(self):
        return np.zeros(self.numAtoms, dtype=np.float32)

    @property
    def _masses(self):
        return np.zeros(self.numAtoms, dtype=np.float32)

    @property
    def _insertion(self):
        return np.array(["" for _ in range(self.numAtoms)], dtype=object)

    @property
    def _chain(self):
        return np.array(["" for _ in range(self.numAtoms)], dtype=object)

    @property
    def _segid(self):
        return np.array(["" for _ in range(self.numAtoms)], dtype=object)

    @property
    def _altloc(self):
        return np.array(["" for _ in range(self.numAtoms)], dtype=object)

    @property
    def _record(self):
        return np.array(["HETATM" for _ in range(self.numAtoms)], dtype=object)

    @property
    def numAtoms(self):
        return self._mol.GetNumAtoms()

    @property
    def numFrames(self):
        return self._mol.GetNumConformers()

    @property
    def _totalcharge(self):
        return sum(self._formalcharge)

    def _getBonds(self, fileBonds=True, guessBonds=True):
        return self._bonds

    def getProp(self, prop_name):
        """Returns a given property of the molecule"""
        return self._mol.GetProp(prop_name)

    def filter(self, sel):
        # Not implemented
        raise NotImplementedError("Filtering atoms not supported yet.")

    def copy(self):
        """
        Create a copy of the molecule object

        Returns
        -------
        newsmallmol : :class:`SmallMol`
            A copy of the object
        """
        return SmallMol(
            Chem.Mol(self._mol), fixHs=False, removeHs=False
        )  # Chem.Mol creates a deep copy of the C++ object

    def get(self, returnField, sel="all", convertType=True, invert=False):
        """
        Returns the property for the atom specified with the selection. The selection is another atom property

        Parameters
        ----------
        returnField: str
            The field of the atom to return
        sel: str
            The selection string. atom field name followed by spaced values for that field
        convertType: bool
            If True, and where possible the returnField is converted in rdkit object
            Default: True
        invert: bool
            If True, the selection is inverted
            Default: False

        Returns
        -------
        values: np.array
            The array of values for the property

        Example
        -------
        >>> sm.get('element', 'idx 0 1 7')  # doctest: +SKIP
        array(['C', 'C', 'H'],
              dtype='<U1')
        >>> sm.get('hybridization', 'element N')  # doctest: +SKIP
        array([rdkit.Chem.rdchem.HybridizationType.SP2,
               rdkit.Chem.rdchem.HybridizationType.SP2], dtype=object)
        >>> sm.get('hybridization', 'element N', convertType=False)
        array([3, 3])
        >>> sm.get('element', 'hybridization sp2')  # doctest: +SKIP
        array(['C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N'],
              dtype='<U1')
        >>> sm.get('element', 'hybridization S')  # doctest: +SKIP
        array(['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
              dtype='<U1')
        >>> sm.get('element', 'hybridization 1')  # doctest: +SKIP
        array(['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
              dtype='<U1')
        >>> sm.get('atomobject', 'element N')  # doctest: +SKIP
        array([<rdkit.Chem.rdchem.Atom object at 0x7faf616dd120>,
               <rdkit.Chem.rdchem.Atom object at 0x7faf616dd170>], dtype=object)
        """
        if sel == "all":
            sel = f"idx {convertToString(self._idx.tolist())}"
        # get the field key and the value to grep
        key = sel.split()[0]
        selector = sel.split()[1:]

        if key not in self._atom_fields:
            raise KeyError(f"The property passed {key} does not exist")
        if len(selector) == 0:
            raise ValueError("No selection was provided")

        # get the returnField and process exceptional field
        if hasattr(self, "_" + key):
            _arrayFrom = self.__getattribute__("_" + key)
        else:
            _arrayFrom = self.__getattribute__(key)

        # special selector for hybridization: can be idx, or rdkit.Chem.rdchem.HybridizationType
        if key == "hybridization":
            try:
                selector = [_hybridizations_StringToType[s.upper()] for s in selector]
            except Exception:
                pass

        _dtype = self._dtypes[key]
        if _dtype is not object:
            selector = [_dtype(s) for s in selector]
        idxs = np.concatenate([np.where(_arrayFrom == s)[0] for s in selector])
        if invert:
            idxs = np.array([i for i in self._idx if i not in idxs])
        idxs = np.sort(idxs)

        if returnField == "atomobject":
            return self.getAtoms()[idxs]
        elif returnField == "bondtype":
            if convertType:
                return np.array(
                    [
                        [_bondtypes_IdxToType[bt] for bt in neighbt]
                        for neighbt in self._neighborsbondtype[idxs]
                    ],
                    dtype=object,
                )
            else:
                return self._neighborsbondtype[idxs]
        elif returnField == "hybridization" and convertType:
            _arrayTo = np.array(
                [_hybridizations_IdxToType[v] for v in self._hybridization],
                dtype=object,
            )
        else:
            _arrayTo = self.__getattribute__("_" + returnField)

        return _arrayTo[idxs]

    def foundBondBetween(self, sel1, sel2, bondtype=None):
        """
        Returns True if at least a bond is found between the two selections. It is possible to check for specific bond
        type. A tuple is returned in the form (bool, [ [(idx1,idx2), rdkit.Chem.rdchem.BondType]] ])

        Parameters
        ----------
        sel1: str
            The selection for the first set of atoms
        sel2: str
            The selection for the second set of atoms
        bondtype: str or int
            The bondtype as index or string
            Default: None

        Returns
        -------
        isbond: bool
            True if a bond was found
        details: list
            A list of lists with the index of atoms in the bond and its type
        """

        if isinstance(bondtype, str):
            _btype = _bondtypes_FullStringToType[bondtype]
        else:
            _btype = bondtype

        atomIdx_sel1 = self.get("idx", sel1)
        neighbors_sel1 = self.get("neighbors", sel1)
        bondtypes_sel1 = self.get("bondtype", sel1)
        atomIdx_sel2 = self.get("idx", sel2)

        founds = []
        for aIdx, neighbors_set, btype_set in zip(
            atomIdx_sel1, neighbors_sel1, bondtypes_sel1
        ):
            for neighbor, btype in zip(neighbors_set, btype_set):
                if neighbor in atomIdx_sel2:
                    btype_str = _bondtypes_TypeToString[btype]
                    if _btype is not None and _btype == btype:
                        founds.append([(aIdx, neighbor), btype_str])
                    elif bondtype is None:
                        founds.append([(aIdx, neighbor), btype_str])
        if len(founds) != 0:
            return True, founds

        return False

    def isChiral(self, returnDetails=False):
        """
        Returns True if the molecule has at least one chiral atom. If returnDetails is set as True,
        a list of tuples with the atom idx and chiral type is returned.

        Parameters
        ----------
        returnDetails: bool
            If True, returns the chiral atoms and their chiral types
            Default: False

        Returns
        -------
        ischiral: bool
            True if the atom has at least a chiral atom
        details: list
            A list of tuple with the chiral atoms and their types

        Example
        -------
        >>> chiralmol.isChiral()  # doctest: +SKIP
        True
        >>> chiralmol.isChiral(returnDetails=True)  # doctest: +SKIP
        (True, [('C2', 'R')])
        """

        _chirals = self._chiral

        idxs = np.where(_chirals != "")[0]
        if len(idxs) == 0:
            return False, None
        if returnDetails:
            idxs = idxs.astype(str)
            idxs_str = " ".join(idxs)
            names = self.get("name", f"idx {idxs_str}")
            chirals = self.get("chiral", f"idx {idxs_str}")

            return True, [(a, c) for a, c in zip(names, chirals)]

        return True, None

    def getAtoms(self):
        """
        Retuns an array with the rdkit.Chem.rdchem.Atom present in the molecule
        """
        return np.array([a for a in self._mol.GetAtoms()])

    def getCenter(self):
        """
        Returns geometrical center of molecule conformation
        """
        coords = self._coords[:, :, self.frame]
        return coords.mean(axis=0).astype(np.float32)

    def generateConformers(
        self,
        num_confs=400,
        optimizemode="mmff",
        align=True,
        append=True,
        pruneRmsThresh=0.5,
        maxAttempts=10000,
        seed=None,
        numThreads=1,
        useRandomCoords=True,
    ):
        """
        Generates ligand conformers

        Parameters
        ----------
        num_confs: int
           Number of conformers to generate.
        optimizemode: str
            The optimizemode to use. Can be  'uff', 'mmff'
        align: bool
            If True, the conformer are aligned to the first one
        append: bool
            If False, the current conformers are deleted
        pruneRmsThresh: float
            The RMSD threshold for pruning conformers
        maxAttempts: int
            The maximum number of attempts to generate conformers
        seed: int
            The seed for the random number generator
        numThreads: int
            The number of threads to use when embedding multiple conformations
        useRandomCoords: bool
            Start the embedding from random coordinates instead of using eigenvalues of the distance matrix
        """
        from rdkit.Chem.AllChem import (
            UFFOptimizeMolecule,
            MMFFOptimizeMolecule,
            EmbedMultipleConfs,
            ETKDGv3,
        )
        from rdkit.Chem.rdMolAlign import AlignMolConformers

        ps = ETKDGv3()
        if seed is not None:
            ps.randomSeed = seed
        ps.numThreads = numThreads
        ps.useRandomCoords = useRandomCoords
        ps.pruneRmsThresh = pruneRmsThresh
        ps.maxIterations = maxAttempts
        ps.clearConfs = False

        if not append:
            self.dropFrames(np.arange(self.numFrames))

        # get the rdkit mol and copy it.
        mol = deepcopy(self._mol)
        # hydrogens are added for safety
        mol = Chem.AddHs(mol)

        # generating conformations
        ids = EmbedMultipleConfs(mol, num_confs, ps)
        if optimizemode not in ["uff", "mmff"]:
            raise ValueError('Unknown optimizemode. Should be  "uff", "mmff"')
        # optimizing conformations depends on the optimizemode passed
        for id in ids:
            if optimizemode == "mmff":
                MMFFOptimizeMolecule(mol, confId=id)
            elif optimizemode == "uff":
                UFFOptimizeMolecule(mol, confId=id)

        if align:
            AlignMolConformers(mol)

        self._mol = mol

    def align(self, refmol):
        from rdkit.Chem.rdMolAlign import GetO3A
        from moleculekit.molecule import Molecule

        if isinstance(refmol, SmallMol):
            refmol = refmol._mol
        if isinstance(refmol, Molecule):
            refmol = SmallMol(refmol)._mol

        pyO3A = GetO3A(self._mol, refmol)
        rmsd = pyO3A.Align()
        logger.info(f"Alignment with a RMSD of {rmsd}")

    def dropFrames(self, frames="all"):
        if isinstance(frames, int):
            frames = np.array([frames])
        if isinstance(frames, list):
            frames = np.array(frames)
        if isinstance(frames, str) and frames == "all":
            ids = [cc.GetId() for cc in self._mol.GetConformers()]
            for f in ids:
                self._mol.RemoveConformer(int(f))
        else:
            if np.any(frames >= self.numFrames):
                raise RuntimeError(
                    f"Cannot remove more frames than existing which is {self.numFrames}"
                )
            ids = [cc.GetId() for cc in self._mol.GetConformers()]
            for f in frames:
                self._mol.RemoveConformer(int(ids[f]))

    def write(self, fname, frames=None, merge=True):
        ext = os.path.splitext(fname)[-1]
        if ext == ".sdf":
            chemwrite = Chem.SDWriter
            # If merge is set as True a unique file is generated
            if merge:
                writer = chemwrite(fname)
                writer.write(self._mol)
            else:  # If merge is set as False a file is created for each conformer
                if frames is None:
                    frames = list(range(self.numFrames))
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                for i in frames:
                    fname_first = os.path.splitext(fname)[0]
                    currfname = os.path.join(f"{fname_first}_{i}{ext}")
                    writer = chemwrite(currfname)
                    writer.write(self._mol, confId=i)
        else:
            mol = self.toMolecule()
            mol.write(fname, frames)

    def view(self, *args, **kwargs):
        self.toMolecule().view(*args, **kwargs)

    def getDescriptors(self, prefix="", ignore=("Ipc",)):
        """Calculate descriptors for the molecule

        Returns rdkit descriptors for the molecule, like DESC_NumRotatableBonds or DESC_MolLogP.
        See rdkit.Chem.Descriptors for more.

        Parameters
        ----------
        prefix : str
            A string prefix to add to all dictionary keys
        ignore : list
            A list of descriptors which to not calculate

        Returns
        -------
        descriptors : dict
            A dictionary containing all descriptors of the molecule
        """
        from rdkit.Chem import Descriptors

        desc_funcs = dict(Descriptors.descList)
        for name in ignore:
            del desc_funcs[name]

        descriptors = {}
        for desc_name, desc_f in desc_funcs.items():
            try:
                value = desc_f(self._mol)
            except Exception as e:
                logger.warning(
                    f"Descriptor {desc_name} was not calculated for this small molecule due to {e}"
                )
                value = None

            descriptors[f"{prefix}{desc_name}"] = value

        return descriptors

    def getFingerprint(self, mode, radius=2, num_bits=1024):
        """
        Returns Morgan, MACCS and AvalonCount fingerprints at specified radius and num_bits

        Parameters
        ----------
        mode : str
            One of 'Morgan', 'MACCS', 'AvalonCount'
        radius: int
           Radius to define a local environment for the relevant fingeprints
        num_bits: int
            The number of bits to use in the fingerprint. Larger avoids collisions.

        Returns
        -------
        A list with the three fingerprints
        """
        if mode not in ("Morgan", "MACCS", "AvalonCount"):
            raise RuntimeError(
                "mode can be only one of 'Morgan', 'MACCS', 'AvalonCount'"
            )

        if mode == "Morgan":
            from rdkit.Chem import AllChem

            return AllChem.GetHashedMorganFingerprint(
                self._mol, radius, num_bits, useChirality=True
            )
        if mode == "MACCS":
            from rdkit.Chem import MACCSkeys

            return MACCSkeys.GenMACCSKeys(self._mol)
        if mode == "AvalonCount":
            from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP

            return GetAvalonCountFP(self._mol, num_bits)

    def stripSalts(self):
        """Removes any salts from the molecule"""
        from rdkit.Chem.SaltRemover import SaltRemover

        remover = SaltRemover()
        self._mol = remover.StripMol(self._mol)

    def containsMetals(
        self, metalSMARTS="[Mg,Ca,Zn,As,Mn,Al,Pd,Pt,Co,Ba,Cr,Cu,Ni,Ag,Fe,Hg,Cd,Gd,Na]"
    ):
        """Returns True if the molecule contains metals

        Parameters
        ----------
        metalSMARTS : str
            SMARTS for detecting metals

        Returns
        -------
        contains : bool
            True if the molecule contains metals, else False
        """
        query = Chem.MolFromSmarts(metalSMARTS)
        return self._mol.HasSubstructMatch(query)

    def assignStereoChemistry(self, from3D=True):
        if from3D:
            Chem.AssignStereochemistryFrom3D(self._mol)
        else:
            Chem.AssignStereochemistry(self._mol, force=True, cleanIt=True)

    def toSMARTS(self, explicitHs=False):
        """
        Returns the smarts string of the molecule

        Parameters
        ----------
        explicitHs: bool
            Set as True for keep the hydrogens

        Returns
        -------
        smart: str
            The smarts string
        """
        from copy import deepcopy

        rmol = deepcopy(self._mol)
        if not explicitHs and len(np.where(self._element == "H")[0]) != 0:
            rmol = Chem.RemoveHs(rmol)

        return Chem.MolToSmarts(rmol, isomericSmiles=True)

    def toSMILES(self, explicitHs=False, kekulizeSmile=True):
        """
        Returns the smiles string of the molecule

        Parameters
        ----------
        explicitHs: bool
            Set as True to keep the hydrogens
        kekulizeSmile: bool
            Set as True to return the kekule smile format

        Returns
        -------
        smi: str
            The smiles string
        """
        from copy import deepcopy

        rmol = deepcopy(self._mol)

        if not explicitHs and len(np.where(self._element == "H")[0]) != 0:
            rmol = Chem.RemoveHs(rmol)

        if kekulizeSmile:
            Chem.Kekulize(rmol)
            smi = Chem.MolToSmiles(rmol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = Chem.MolToSmiles(rmol, isomericSmiles=True)

        return smi

    def toMolecule(self, ids=None):
        """
        Return the moleculekit.molecule.Molecule

        Parameters
        ----------
        ids: list
            The list of conformer ids to store in the moleculekit Molecule object- If None, all are returned
            Default: None

        Returns
        -------
        mol: moleculekit.molecule.Molecule
            The moleculekit Molecule object

        """
        from moleculekit.molecule import Molecule

        class NoConformerError(Exception):
            pass

        if ids is None:
            ids = np.arange(self.numFrames)

        if self.numFrames == 0:
            raise NoConformerError(
                "No Conformers are found in the molecule. Generate at least one conformer."
            )
        elif not isinstance(ids, list) and not isinstance(ids, np.ndarray):
            raise ValueError("The argument ids should be a list of confomer ids")

        mol = Molecule()
        mol.empty(self.numAtoms, numFrames=len(ids))
        mol.record[:] = "HETATM"
        mol.resname[:] = self.ligname[:3]
        mol.resid[:] = self._resid
        mol.coords = self._coords[:, :, ids]
        mol.name[:] = self._name
        mol.element[:] = self._element
        mol.formalcharge[:] = self._formalcharge
        mol.charge[:] = self._charge
        mol.viewname = self.ligname
        mol.bonds = self._bonds
        mol.bondtype = self._bondtype
        mol.atomtype = self._atomtype
        return mol

    def depict(
        self,
        sketch=True,
        filename=None,
        ipython=False,
        optimize=False,
        optimizemode="std",
        removeHs=True,
        atomlabels=None,
        highlightAtoms=None,
        resolution=(400, 200),
    ):
        """
        Depicts the molecules. It is possible to save it into an svg file and also generates a jupiter-notebook rendering

        Parameters
        ----------
        sketch: bool
            Set to True for 2D depiction
        filename: str
            Set the filename for the svg file
        ipython: bool
            Set to True to return the jupiter-notebook rendering
        optimize: bool
            Set to True to optimize the conformation. Works only with 3D.
        optimizemode: ['std', 'mmff']
            Set the optimization mode for 3D conformation
        removeHs: bool
            Set to True to hide hydrogens in the depiction
        atomlabels: str
            Accept any combinations of the following pararemters as unique string '%a%i%c%*' a:atom name, i:atom index,
            c:atom formal charge (+/-), *:chiral (* if atom is chiral)
        highlightAtoms: list
            List of atom to highlight. It can be also a list of atom list, in this case different colors will be used
        resolution: tuple of integers
            Resolution in pixels: (X, Y)

        Returns
        -------
            ipython_svg: SVG object if ipython is set to True

        Example
        -------
        >>> sm.depict(ipython=True, optimize=True, optimizemode='std')  # doctest: +SKIP
        >>> sm.depict(ipython=True, sketch=True)  # doctest: +SKIP
        >>> sm.depict(ipython=True, sketch=True)  # doctest: +SKIP
        >>> sm.depict(ipython=True, sketch=True, atomlabels="%a%i%c")  # doctest: +SKIP
        >>> ids = np.intersect1d(sm.get('idx', 'hybridization SP2'), sm.get('idx', 'element C'))  # doctest: +SKIP
        >>> sm.depict(ipython=True, sketch=True,highlightAtoms=ids.tolist(), removeHs=False)  # doctest: +SKIP
        """
        from rdkit import Chem
        from rdkit.Chem.AllChem import (
            Compute2DCoords,
            EmbedMolecule,
            MMFFOptimizeMolecule,
            ETKDG,
        )
        from copy import deepcopy

        if sketch and optimize:
            raise ValueError(
                "Impossible to use optimization in  2D sketch representation"
            )

        if optimizemode not in ["std", "mmff"]:
            raise ValueError(
                f'Optimization mode {optimizemode} not understood. Can be "std" or "ff"'
            )

        elements = self._element
        indexes = self._idx
        formalcharges = self._formalcharge
        chirals = self._chiral

        _mol = deepcopy(self._mol)

        if sketch:
            Compute2DCoords(_mol)

        if removeHs:
            _mol = Chem.RemoveHs(_mol)
            elements = self.get("element", "element H", invert=True)
            indexes = self.get("idx", "element H", invert=True)
            formalcharges = self.get("formalcharge", "element H", invert=True)
            chirals = self.get("chiral", "element H", invert=True)

        _labelsFunc = ["a", "i", "c", "*"]

        if atomlabels is not None:
            labels = atomlabels.split("%")[1:]
            formalcharges = [
                "" if c == 0 else "+" if c == 1 else "-" for c in formalcharges
            ]
            chirals = ["" if c == "" else "*" for c in chirals]
            values = [elements, indexes, formalcharges, chirals]

            idxs = [_labelsFunc.index(lab) for lab in labels]
            labels_required = [values[i] for i in idxs]
            atomlabels = [
                "".join([str(i) for i in a]) for a in list(zip(*labels_required))
            ]

        if optimize:
            if optimizemode == "std":
                EmbedMolecule(_mol, ETKDG())
            elif optimizemode == "mmff":
                MMFFOptimizeMolecule(_mol)

        return _depictMol(
            _mol,
            filename=filename,
            ipython=ipython,
            atomlabels=atomlabels,
            highlightAtoms=highlightAtoms,
            resolution=resolution,
        )

    def addHs(self, addCoords=True):
        self._mol = Chem.AddHs(self._mol, addCoords=addCoords)

    def removeHs(self):
        self._mol = Chem.RemoveHs(self._mol)

    def sanitize(self):
        Chem.SanitizeMol(self._mol)

    def getTautomers(
        self,
        canonical=True,
        genConformers=False,
        returnScores=True,
        maxTautomers=200,
        filterTauts=None,
    ):
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from moleculekit.smallmol.smallmollib import SmallMolLib

        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(maxTautomers)
        if canonical:
            tauts = [SmallMol(enumerator.Canonicalize(self._mol))]
        else:
            tauts = [SmallMol(x) for x in enumerator.Enumerate(self._mol)]

        if returnScores or filterTauts is not None:
            scores = []
            for tt in tauts:
                scores.append(enumerator.ScoreTautomer(tt._mol))

            if filterTauts is not None:
                max_score = max(scores)
                new_tauts = []
                new_scores = []
                for i in range(len(tauts)):
                    if scores[i] >= max_score - filterTauts:
                        new_tauts.append(tauts[i])
                        new_scores.append(scores[i])
                tauts = new_tauts
                scores = new_scores

        if genConformers:
            for tt in tauts:
                tt.generateConformers(num_confs=1)

        sms = SmallMolLib()
        sms._mols = tauts
        if returnScores:
            return sms, scores

        return sms

    def setProp(self, key, value):
        self._mol.SetProp(key, str(value))

    def __repr__(self):
        return (
            "<{}.{} object at {}>\n".format(
                self.__class__.__module__, self.__class__.__name__, hex(id(self))
            )
            + self.__str__()
        )

    def __str__(self):
        def formatstr(name, field):
            if isinstance(field, np.ndarray) or isinstance(field, list):
                rep = f"{name} shape: {np.shape(field)}"
            elif field == "reps":
                rep = f"{name}: {len(self.reps.replist)}"
            else:
                rep = f"{name}: {field}"
            return rep

        rep = f"SmallMol with {self.numAtoms} atoms and {self.numFrames} conformers"
        for p in sorted(self._atom_fields):
            if p.startswith("_"):
                continue
            rep += "\n"
            rep += f"Atom field - {p}"
        for j in sorted(self.__dict__.keys() - list(SmallMol._atom_fields)):
            if j[0] == "_":
                continue
            rep += "\n"
            rep += formatstr(j, self.__dict__[j])

        return rep
