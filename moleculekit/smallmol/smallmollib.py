# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from moleculekit.smallmol.smallmol import SmallMol
import gzip
import logging


logger = logging.getLogger(__name__)


def csvReader(file, removeHs: bool, fixHs: bool, isgzip: bool = False, _logger=True):
    from tqdm import tqdm
    import pandas as pd

    if isgzip:
        with gzip.open(file, "rb") as f:
            return csvReader(
                f, removeHs=removeHs, fixHs=fixHs, isgzip=False, _logger=_logger
            )

    if isinstance(file, str):
        with open(file, "r") as f:
            return csvReader(f, removeHs, fixHs, _logger=_logger)

    df = pd.read_csv(file)
    smiles_col = [x for x in df.columns if x.upper() == "SMILES"]
    if len(smiles_col) == 0:
        raise RuntimeError(
            f"Could not find a column named SMILES in the CSV file. Found columns {df.columns}"
        )
    if len(smiles_col) > 1:
        raise RuntimeError(
            f"Found multiple columns named SMILES in the CSV file. Found columns {df.columns}"
        )
    smiles_col = smiles_col[0]

    mols = []
    for i, row in tqdm(df.iterrows(), disable=not _logger):
        smi = row[smiles_col]
        try:
            mols.append(SmallMol(smi, removeHs=removeHs, fixHs=fixHs, _logger=False))
        except Exception as e:
            logger.warning(
                f"Failed to load molecule index {i} with error {e}. Skipping to next molecule."
            )
        for col in df.columns:
            mols[-1].setProp(col, row[col])
        if "_Name" not in df.columns:
            mols[-1].setProp("_Name", f"mol_{i}")
    return mols


def smiReader(file, removeHs: bool, fixHs: bool, isgzip: bool = False, _logger=True):
    from tqdm import tqdm

    if isgzip:
        with gzip.open(file, "rb") as f:
            return smiReader(f, removeHs=removeHs, fixHs=fixHs, isgzip=False)

    if isinstance(file, str):
        with open(file, "r") as f:
            return smiReader(f, removeHs, fixHs)

    decode = False
    if isinstance(file, gzip.GzipFile):
        decode = True

    lines = file.readlines()[1:]
    mols = []
    for i, line in enumerate(tqdm(lines, disable=not _logger)):
        if decode:
            line = line.decode("utf-8")
        smi, name = line.strip().split()
        try:
            mols.append(SmallMol(smi, removeHs=removeHs, fixHs=fixHs))
        except Exception as e:
            logger.warning(
                "Failed to load molecule with name {} with error {}. Skipping to next molecule.".format(
                    name, e
                )
            )
    return mols


def sdfReader(file, removeHs: bool, fixHs: bool, sanitize: bool, isgzip: bool = False, _logger=True):
    from tqdm import tqdm
    from moleculekit.util import tempname

    if isgzip:
        with gzip.open(file, "rb") as f:
            # SDMolSupplier does not support file handles, need to write temp file
            file = tempname(suffix=".sdf")
            with open(file, "wb") as fout:
                fout.write(f.read())

    supplier = Chem.SDMolSupplier(file, removeHs=removeHs, sanitize=sanitize)
    mols = []
    countfailed = 0
    for mol in tqdm(supplier, disable=not _logger):
        if mol is None:
            countfailed += 1
            continue
        try:
            mols.append(SmallMol(mol, removeHs=removeHs, fixHs=fixHs, _logger=False))
        except Exception:
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
            countfailed += 1
            logger.warning(
                f"Failed to load molecule with name {name}. Skipping to next molecule."
            )
    if countfailed:
        logger.info(f"Failed to load {countfailed}/{len(supplier)} molecules")
    return mols


class SmallMolLib(object):
    """
    Class to manage ligands databases (sdf). Ligands are stored as moleculekit.smallmol.smallmol.SmallMol objects and
    fields type in the sdf are stored in a list

    Parameters
    ----------
    libfile: str
        The sdf or smi file path
    removeHs: bool
        If True, the hydrogens of the molecules will be removed
    fixHs: bool
        If True, the hydrogens are added and optimized
    sanitize: bool
        If True, the molecules are sanitized after reading.

    Example
    -------
    >>> import os
    >>> lib = SmallMolLib('fda_drugs_light.sdf')
    >>> lib.numMols
    100

    .. rubric:: Methods
    .. autoautosummary:: moleculekit.smallmol.smallmol.SmallMolLib
       :methods:
    .. rubric:: Attributes
    .. autoautosummary:: moleculekit.smallmol.smallmol.SmallMolLib
       :attributes:
    """

    def __init__(
        self,
        libfile: str | None = None,
        removeHs: bool = False,
        fixHs: bool = True,
        sanitize: bool = True,
        _logger=True,
    ):  # , n_jobs=1
        if removeHs and _logger:
            logger.info("Removing hydrogens from library molecules (removeHs=True)")
        if fixHs and _logger:
            logger.info(
                "Adding any missing hydrogens to library molecules (fixHs=True)"
            )
        if libfile is not None:
            self._mols = self._loadLibrary(
                libfile,
                removeHs=removeHs,
                fixHs=fixHs,
                sanitize=sanitize,
                ext=None,
                _logger=_logger,
            )

    def _loadLibrary(
        self, libfile, removeHs=False, fixHs=True, sanitize=True, ext=None, _logger=True
    ):
        isgzip = False
        if ext is None:
            ext = os.path.splitext(libfile)[-1]
        if ext == ".gz":
            isgzip = True
            ext = os.path.splitext(os.path.splitext(libfile)[-2])[-1]

        if ext == ".sdf":
            return sdfReader(libfile, removeHs, fixHs, sanitize, isgzip, _logger)
        elif ext == ".smi":
            return smiReader(libfile, removeHs, fixHs, isgzip, _logger)
        elif ext == ".csv":
            return csvReader(libfile, removeHs, fixHs, isgzip, _logger)
        else:
            raise RuntimeError(f"Invalid file extension {ext}. Could not read it.")

    @property
    def numMols(self) -> int:
        """
        Returns the number of molecules in the library.

        Returns
        -------
        nummols : int
            The number of molecules
        """
        return len(self._mols)

    def getMols(self, ids: list | None = None):
        """
        Returns the SmallMol objects that corresponds ot the indexes of the list passed

        Parameters
        ----------
        ids: list
            The index list of the molecules to return

        Returns
        -------
        smallmollist: list
            The list of SmallMol objects

        Example
        -------
        >>> lib2 = lib.getMols([1,2,3])
        >>> len(lib2)
        3
        """

        if ids is None:
            return self._mols
        if not isinstance(ids, list):
            raise TypeError(f"The argument ids {type(ids)} should be list")

        return np.array(self._mols)[ids]

    def writeSdf(self, sdf_name: str, fields: list | None = None):
        """
        Writes an sdf file with molecules stored. Is it possible also to manage which field will be written

        Parameters
        ----------
        sdf_name: str
            The ouput sdf filename
        fields: list
            A list of the fields to write. If None all are saved
        """

        from rdkit.Chem import SDWriter

        writer = SDWriter(sdf_name)
        if fields is not None:
            if not isinstance(fields, list):
                raise TypeError(f"The fields argument {type(fields)} should be a list")
            writer.SetProps(fields)

        for m in self._mols:
            writer.write(m._mol)

    def writeSmiles(
        self,
        smi_name: str,
        explicitHs: bool = True,
        names: bool = False,
        header: str | None = None,
    ):
        """
        Writes a smi file with molecules stored. Is it possible to specify the header of the smi file. The name of the
        ligands can be their ligand name or a sequential ID.

        Parameters
        ----------
        smi_name: str
            The ouput smi filename
        explicitHs: bool
            Set as True to write explicit hydrogens in the SMILES strings.
        names: bool
            Set as True to use the own ligand name for each ligand. Otherwise a sequential ID will be used
        header: str
            The header of the smi file. If is None the smi filename will be used.
        """

        smi_name = os.path.splitext(smi_name)[0] + ".smi"

        f = open(smi_name, "w")

        if header is None:
            header = os.path.splitext(smi_name)[0]

        f.write(header + "\n")

        for n, sm in enumerate(self.getMols()):
            smi = sm.toSMILES(explicitHs=explicitHs)
            name = n if not names else sm.ligname
            f.write(smi + f" {name} \n")

        f.close()

    def appendSmallLib(
        self, smallLib: "SmallMolLib", strictField: bool = False, strictDirection: int = 1
    ):
        """
        Merge two moleculekit.smallmol.smallmol.SmallMolLib objects

        Parameters
        ----------
        smallLib: moleculekit.smallmol.smallmol.SmallMolLib
            The new SmallMolLib to merge
        strictField: bool
            Currently unused.
            Default: False
        strictDirection: int
            Currently unused.
            Default: 1
        """
        self._mols += smallLib._mols

    def appendSmallMol(
        self, smallmolecule: "SmallMol", strictField: bool = False, strictDirection: int = 0
    ):
        """
        Adds a new moleculekit.smallmol.smallmol.SmallMol object in the current SmallMolLib object

        Parameters
        ---------
        smallmolecule: moleculekit.smallmol.smallmol.SmallMol
            The SmallMol object to add
        strictField: bool
            Currently unused.
            Default: False
        strictDirection: int
            Currently unused.
            Default: 0
        """
        self._mols = np.append(self._mols, smallmolecule)

    def removeMols(self, ids: list):
        """
        Removes the moleculekit.smallmol.smallmol.SmallMol object based on the indexes in the list

        Parameters
        ----------
        ids: list
            The list of molecules index to remove from the SmallMolLib
        """

        if not isinstance(ids, list):
            raise TypeError(
                f"The argument ids {type(ids)} is not valid. Should be list"
            )
        _oldNumMols = self.numMols
        self._mols = np.delete(self._mols, ids)

        logger.warning(
            "[num mols before deleting: {}]. The molecules {} were removed, now the number of "
            "molecules are {} ".format(_oldNumMols, ids, self.numMols)
        )

    def toDataFrame(self, fields: list | None = None, molAsImage: bool = True, sketch: bool = True):
        """
        Returns a pandas.DataFrame of the SmallMolLib object.

        Parameters
        ----------
        fields: list
            The list of fields to convert into a pandas DataFrame column
        molAsImage: bool
            If True, the rdkit.Chem.rdchem.Mol is converted into an image
        sketch: bool
            If True, the molecule are rendered to be 2D

        Returns
        -------
        dataframe: pandas.DataFrame
            The pandas DataFrame
        """
        from rdkit.Chem.AllChem import Compute2DCoords
        import pandas as pd
        from copy import deepcopy

        if fields is not None:
            if not isinstance(fields, list):
                raise TypeError(
                    f"The argument fields passed {type(fields)} should be a list "
                )
        else:
            fields = ["ligname", "_mol"] if molAsImage else ["ligname"]

        records = []
        indexes = []
        for i, m in enumerate(self._mols):
            row = dict((f, m.__getattribute__(f)) for f in fields)
            if sketch:
                mm = deepcopy(m._mol)
                Compute2DCoords(mm)
                row["_mol"] = mm
            records.append(row)
            indexes.append(i)

        df = pd.DataFrame(records, columns=fields, index=indexes)
        if molAsImage:
            PandasTools.ChangeMoleculeRendering(df)
        return df

    def copy(self) -> "SmallMolLib":
        """
        Returns a deep copy of the SmallMolLib object.

        Returns
        -------
        newlib : :class:`SmallMolLib`
            A copy of the object
        """
        from copy import deepcopy

        return deepcopy(self)

    def __len__(self):
        return len(self._mols)

    def __getitem__(self, item):
        return self._mols[item]

    def __iter__(self):
        _mols = self._mols
        for smallmol in _mols:
            yield smallmol

    def __str__(self):
        _mols = self._mols

        return (
            "Stack of Small molecules."
            "\n\tContains {} Molecules."
            '\n\tSource file: "{}".'
        ).format(len(_mols), self._sdffile)

    def depict(
        self,
        ids: list | None = None,
        sketch: bool = True,
        filename: str | None = None,
        ipython: bool = False,
        optimize: bool = False,
        optimizemode: str = "std",
        removeHs: bool = True,
        legends: str | None = None,
        highlightAtoms: list | None = None,
        mols_perrow: int = 3,
    ):
        """
        Depicts the molecules into a grid. It is possible to save it into an svg file and also generates a
        jupiter-notebook rendering

        Parameters
        ----------
        ids: list
            The index of the molecules to depict
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
        legends: str
            A legend text to add under each molecule. Can be 'names':the name of the molecule; 'items': a incremental id, or any other SDF property name.
        highlightAtoms: list
            A List of atom to highligh for each molecule. It can be also a list of atom list, in this case different
            colors will be used
        mols_perrow: int
            The number of molecules to depict per row of the grid

        Returns
        -------
        ipython_svg : IPython.display.SVG or None
            An SVG rendering object if ``ipython`` is True, otherwise None
        """
        from rdkit.Chem.AllChem import (
            Compute2DCoords,
            EmbedMolecule,
            MMFFOptimizeMolecule,
            ETKDG,
        )
        from rdkit.Chem import RemoveHs
        from moleculekit.smallmol.util import depictMultipleMols

        if sketch and optimize:
            raise ValueError(
                "Impossible to use optmization in  2D sketch representation"
            )

        _smallmols = self.getMols(ids)

        if ids is None:
            _mols = [m._mol for m in self._mols]
        else:
            _mols = [m._mol for m in self.getMols(ids)]

        if highlightAtoms is not None:
            if len(highlightAtoms) != len(_mols):
                raise ValueError(
                    "The highlightAtoms {} should have the same length of the "
                    "mols {}".format(len(highlightAtoms), len(_mols))
                )

        if sketch:
            for _m in _mols:
                Compute2DCoords(_m)

        if removeHs:
            _mols = [RemoveHs(_m) for _m in _mols]

        # activate 3D coords optimization
        if optimize:
            if optimizemode == "std":
                for _m in _mols:
                    EmbedMolecule(_m)
            elif optimizemode == "mmff":
                for _m in _mols:
                    MMFFOptimizeMolecule(_m, ETKDG())

        legends_list = []
        if legends == "names":
            legends_list = [_m.ligname for _m in _smallmols]
        elif legends == "items":
            legends_list = [str(n + 1) for n in range(len(_smallmols))]
        else:
            try:
                legends_list = [
                    f"{legends}={_m.GetDoubleProp(legends)}" for _m in _mols
                ]
            except Exception as e:
                raise RuntimeError(
                    f"Failed at getting molecule property '{legends}' passed in `legends` argument with error {e}"
                )

        return depictMultipleMols(
            _mols,
            ipython=ipython,
            legends=legends_list,
            highlightAtoms=highlightAtoms,
            filename=filename,
            mols_perrow=mols_perrow,
        )


SmallMolStack = SmallMolLib
