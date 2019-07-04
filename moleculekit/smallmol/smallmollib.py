# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
import math
import numpy as np
from rdkit import Chem
from moleculekit.smallmol.smallmol import SmallMol
import gzip
import logging


logger = logging.getLogger(__name__)


def smiReader(file, removeHs, fixHs, isgzip=False):
    from tqdm import tqdm
    if isgzip:
        with gzip.open(file, 'rb') as f:
            return smiReader(f, removeHs=removeHs, fixHs=fixHs, isgzip=False)

    if isinstance(file, str):
        with open(file, 'r') as f:
            return smiReader(f, removeHs, fixHs)

    decode = False
    if isinstance(file, gzip.GzipFile):
        decode = True
            
    lines = file.readlines()[1:]
    mols = []
    for i, line in enumerate(tqdm(lines)):
        if decode:
            line = line.decode('utf-8')
        smi, name = line.strip().split()
        try:
            mols.append(SmallMol(smi, removeHs=removeHs, fixHs=fixHs))
        except Exception as e:
            logger.warning('Failed to load molecule with name {} with error {}. Skipping to next molecule.'.format(name, e))
    return mols

def sdfReader(file, removeHs, fixHs, sanitize, isgzip=False):
    from tqdm import tqdm
    from moleculekit.util import tempname

    if isgzip:
        with gzip.open(file, 'rb') as f:
            # SDMolSupplier does not support file handles, need to write temp file
            file = tempname(suffix='.sdf')
            with open(file, 'wb') as fout:
                fout.write(f.read())

    supplier = Chem.SDMolSupplier(file, removeHs=removeHs, sanitize=sanitize)
    mols = []
    countfailed = 0
    for mol in tqdm(supplier):
        if mol is None:
            countfailed += 1
            continue
        try:
            mols.append(SmallMol(mol, removeHs=removeHs, fixHs=fixHs))
        except:
            if mol.HasProp('_Name'):
                name = mol.GetProp('_Name')
            countfailed += 1
            logger.warning('Failed to load molecule{}. Skipping to next molecule.'.format(' with name {}'.format(name)))
    if countfailed:
        logger.info('Failed to load {}/{} molecules'.format(countfailed, len(supplier)))
    return mols

class SmallMolLib(object):
    """
    Class to manage ligands databases (sdf). Ligands are stored as moleculekit.smallmol.smallmol.SmallMol objects and
    fields type in the sdf are stored in a list

    Parameters
    ----------
    lib_file: str
        The sdf or smi file path
    removeHs: bool
        If True, the hydrogens of the molecules will be removed
    fixHs: bool
        If True, the hydrogens are added and optimized

    Example
    -------
    >>> import os
    >>> from moleculekit.home import home
    >>> lib = SmallMolLib(os.path.join(home(dataDir='test-smallmol'), 'fda_drugs_light.sdf'))
    >>> lib.numMols
    100

    .. rubric:: Methods
    .. autoautosummary:: moleculekit.smallmol.smallmol.SmallMolLib
       :methods:
    .. rubric:: Attributes
    .. autoautosummary:: moleculekit.smallmol.smallmol.SmallMolLib
       :attributes:

    Attributes
    ----------
    numMols: int
        Number of SmallMol molecules

    """

    def __init__(self, libfile=None, removeHs=False, fixHs=True, sanitize=True):  # , n_jobs=1
        if libfile is not None:
            self._mols = self._loadLibrary(libfile, removeHs=removeHs, fixHs=fixHs, sanitize=sanitize, ext=None)


    def _loadLibrary(self, libfile, removeHs=False, fixHs=True, sanitize=True, ext=None):
        isgzip = False
        if ext == None:
            ext = os.path.splitext(libfile)[-1] 
        if ext == '.gz':
            isgzip = True
            ext = os.path.splitext(os.path.splitext(libfile)[-2])[-1]

        if ext == '.sdf':
            return sdfReader(libfile, removeHs, fixHs, sanitize, isgzip)
        elif ext == '.smi':
            return smiReader(libfile, removeHs, fixHs, isgzip)
        else:
            raise RuntimeError('Invalid file extension {}. Could not read it.'.format(ext))

    @property
    def numMols(self):
        """
        Returns the number of molecules
        """
        return len(self._mols)

    def getMols(self, ids=None):
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
            raise TypeError("The argument ids {} should be list".format(type(ids)))

        return np.array(self._mols)[ids]

    def writeSdf(self, sdf_name, fields=None):
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
                raise TypeError("The fields argument {} should be a list".format(type(fields)))
            writer.SetProps(fields)

        for m in self._mols:
            writer.write(m._mol)

    def writeSmiles(self, smi_name, explicitHs=True, names=False, header=None):
        """
        Writes a smi file with molecules stored. Is it possible to specify the header of the smi file. The name of the
        ligands can be their ligand name or a sequential ID.

        Parameters
        ----------
        smi_name: str
            The ouput smi filename
        names: bool
            Set as True to use the own ligand name for each ligand. Otherwise a sequential ID will be used
        header: str
            The header of the smi file. If is None the smi filename will be used.
        """

        smi_name = os.path.splitext(smi_name)[0] + '.smi'

        f = open(smi_name, 'w')

        if header is None:
            header = os.path.splitext(smi_name)[0]

        f.write(header + '\n')

        for n, sm in enumerate(self.getMols()):
            smi = sm.toSMILES(explicitHs=explicitHs)
            name = n if not names else sm.ligname
            f.write(smi + ' {} \n'.format(name))

        f.close()

    def appendSmallLib(self, smallLib, strictField=False, strictDirection=1):
        """
        Merge two moleculekit.smallmol.smallmol.SmallMolLib objects

        Parameters
        ----------
        smallLib: moleculekit.smallmol.smallmol.SmallMolLib
            The new SmallMolLib to merge
        """
        self._mols += smallLib._mols


    def appendSmallMol(self, smallmolecule, strictField=False, strictDirection=0):
        """
        Adds a new moleculekit.smallmol.smallmol.SmallMol object in the current SmallMolLib object

        Parameters
        ---------
        smallmol: moleculekit.smallmol.smallmol.SmallMol
            The SmallMol object to add
        """
        self._mols = np.append(self._mols, smallmolecule)

    def removeMols(self, ids):
        """
        Removes the moleculekit.smallmol.smallmol.SmallMol object based on the indexes in the list

        Parameters
        ----------
        ids: list
            The list of molecules index to remove from the SmallMolLib
        """

        if not isinstance(ids, list):
            raise TypeError('The argument ids {} is not valid. Should be list'.format(type(ids)))
        _oldNumMols = self.numMols
        self._mols = np.delete(self._mols, ids)

        logger.warning("[num mols before deleting: {}]. The molecules {} were removed, now the number of "
                       "molecules are {} ".format(_oldNumMols, ids, self.numMols))

    def toDataFrame(self, fields=None, molAsImage=True, sketch=True):
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
        from rdkit.Chem import PandasTools
        from rdkit.Chem.AllChem import Compute2DCoords
        import pandas as pd
        from copy import deepcopy

        if fields is not None:
            if not isinstance(fields, list):
                raise TypeError('The argument fields passed {} should be a list '.format(type(fields)))
        else:
            fields = ['ligname', '_mol'] if molAsImage else ['ligname']

        records = []
        indexes = []
        for i, m in enumerate(self._mols):
            row = dict((f, m.__getattribute__(f)) for f in fields)
            if sketch:
                mm = deepcopy(m._mol)
                Compute2DCoords(mm)
                row['_mol'] = mm
            records.append(row)
            indexes.append(i)

        df = pd.DataFrame(records, columns=fields, index=indexes)
        if molAsImage:
            Chem.PandasTools.ChangeMoleculeRendering(df)
        return df

    def copy(self):
        """
        Returns a copy of the SmallMolLib object
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

        return ('Stack of Small molecules.'
                '\n\tContains {} Molecules.'
                '\n\tSource file: "{}".').format(len(_mols), self._sdffile)

    def depict(self, ids=None, sketch=True, filename=None, ipython=False, optimize=False, optimizemode='std',
               removeHs=True,  legends=None, highlightAtoms=None, mols_perrow=3):

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
            The name to used for each molecule. Can be 'names':the name of themselves; or 'items': a incremental id
        highlightAtoms: list
            A List of atom to highligh for each molecule. It can be also a list of atom list, in this case different
            colors will be used
        mols_perrow: int
            The number of molecules to depict per row of the grid

        Returns
        -------
            ipython_svg: SVG object if ipython is set to True

        """
        from rdkit.Chem.AllChem import Compute2DCoords, EmbedMolecule, MMFFOptimizeMolecule, ETKDG
        from rdkit.Chem import RemoveHs
        from moleculekit.smallmol.util import depictMultipleMols

        if sketch and optimize:
            raise ValueError('Impossible to use optmization in  2D sketch representation')

        if legends is not None and legends not in ['names', 'items']:
            raise ValueError('The "legends" should be "names" or "items"')

        _smallmols = self.getMols(ids)

        if ids is None:
            _mols = [m._mol for m in self._mols]
        else:
            _mols = [m._mol for m in self.getMols(ids)]

        if highlightAtoms is not None:
            if len(highlightAtoms) != len(_mols):
                raise ValueError('The highlightAtoms {} should have the same length of the '
                                 'mols {}'.format(len(highlightAtoms), len(_mols)))

        if sketch:
            for _m in _mols:
                Compute2DCoords(_m)

        if removeHs:
            _mols = [RemoveHs(_m) for _m in _mols]

        # activate 3D coords optimization
        if optimize:
            if optimizemode == 'std':
                for _m in _mols:
                    EmbedMolecule(_m)
            elif optimizemode == 'mmff':
                for _m in _mols:
                    MMFFOptimizeMolecule(_m, ETKDG())

        legends_list = []
        if legends == 'names':
            legends_list = [_m.getProp('ligname') for _m in _smallmols]
        elif legends == 'items':
            legends_list = [str(n+1) for n in range(len(_smallmols))]

        return depictMultipleMols(_mols, ipython=ipython, legends=legends_list, highlightAtoms=highlightAtoms,
                                  filename=filename, mols_perrow=mols_perrow)

SmallMolStack = SmallMolLib

if __name__ == '__main__':
    import doctest
    import os
    from moleculekit.home import home

    lib = SmallMolLib(os.path.join(home(dataDir='test-smallmol'), 'fda_drugs_light.sdf'))
    doctest.testmod(extraglobs={'lib': lib})