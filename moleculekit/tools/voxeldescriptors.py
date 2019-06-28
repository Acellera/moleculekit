# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.util import boundingBox
import numpy as np
import ctypes
import moleculekit.home
import os
from numba import cuda, jit
import numba
from math import sqrt, exp
from functools import lru_cache
import logging


logger = logging.getLogger(__name__)

_order = ('hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable',
          'negative_ionizable', 'metal', 'occupancies')
libdir = moleculekit.home.home(libDir=True)
occupancylib = ctypes.cdll.LoadLibrary(os.path.join(libdir, "occupancy_ext.so"))


def viewVoxelFeatures(features, centers, nvoxels, voxelsize=None, draw='wireframe'):
    """ Visualize in VMD the voxel features produced by getVoxelDescriptors.

    Parameters
    ----------
    features : np.ndarray
        An array of (n_centers, n_features) shape containing the voxel features of each center
    centers : np.ndarray
        An array of (n_centers, 3) shape containing the coordinates of each voxel center
    nvoxels : np.ndarray
        An array of (3,) shape containing the number of voxels in each X, Y, Z dimension

    Example
    -------
    >>> feats, centers, N = getVoxelDescriptors(mol)
    >>> viewVoxelFeatures(feats, centers, N)
    """
    from moleculekit.vmdgraphics import VMDIsosurface
    from moleculekit.vmdviewer import getCurrentViewer

    if voxelsize is None:
        voxelsize = abs(centers[0, 2] - centers[1, 2])
        voxelsize = np.repeat(voxelsize, 3)

    voxelsize = np.array(voxelsize)

    features = features.reshape(list(nvoxels) + [len(_order),])
    centers = centers.reshape(list(nvoxels) + [3,])
    loweredge = np.min(centers, axis=(0, 1, 2)) - (voxelsize / 2)

    for i, name in enumerate(_order):
        VMDIsosurface(features[..., i], loweredge, voxelsize, color=i, name=name, draw=draw)

def rotateCoordinates(coords, rotations, center):
    from moleculekit.util import rotationMatrix

    def rotate(coords, rotMat, center=(0, 0, 0)):
        newcoords = coords - center
        return np.dot(newcoords, np.transpose(rotMat)) + center

    rotation = list(rotations)
    matx = rotationMatrix([1, 0, 0], rotation[0])
    maty = rotationMatrix([0, 1, 0], rotation[1])
    matz = rotationMatrix([0, 0, 1], rotation[2])

    coords = coords.copy()
    coords = rotate(coords, matx, center=center)
    coords = rotate(coords, maty, center=center)
    coords = rotate(coords, matz, center=center)
    return coords

def _getChannelRadii(molelements):
    from moleculekit.periodictable import periodictable
    radii = np.array([periodictable[e].vdw_radius for e in molelements])
    return radii


@lru_cache(maxsize=10)  # Caches results of the calculation to not have to recalculate it for each molecule
def _getGridCenters(x, y, z, resolution):
    firstdim = np.repeat(np.arange(x) * resolution, y*z)
    seconddim = np.tile(np.repeat(np.arange(y) * resolution, z), x)
    thirddim = np.tile(np.arange(z) * resolution, x*y)
    combined = np.vstack((firstdim.T, seconddim.T, thirddim.T)).T.astype(np.float64)
    combined = combined.reshape([x, y, z, 3])
    return combined


def getChannels(mol, aromaticNitrogen=False, version=2, validitychecks=True):
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.molecule import Molecule
    if isinstance(mol, SmallMol):
            channels = _getPropertiesRDkit(mol)
    elif isinstance(mol, Molecule):
        if version == 1:
            channels = _getAtomtypePropertiesPDBQT(mol)
        elif version == 2:
            from moleculekit.tools.atomtyper import getFeatures, getPDBQTAtomTypesAndCharges
            mol.atomtype, mol.charge = getPDBQTAtomTypesAndCharges(mol, aromaticNitrogen=aromaticNitrogen, validitychecks=validitychecks)
            channels = getFeatures(mol)

    if channels.dtype == bool:
        # Calculate for each channel the atom sigmas
        sigmas = _getChannelRadii(mol.get('element'))
        channels = sigmas[:, np.newaxis] * channels.astype(float)
    return channels, mol


def getCenters(mol=None, buffer=0, boxsize=None, center=None, voxelsize=1):
    """ Get a set of centers for voxelization.

    Parameters
    ----------
    mol : Molecule or SmallMol object
        Use this together with the buffer option.
    buffer : float
        The buffer space to add to the bounding box. This adds zeros to the grid around the protein so that properties
        which are at the edge of the box can be found in the center of one. Should be usually set to localimagesize/2.
    boxsize : list
        If this argument is None, the function will calculate the bounding box of the molecule, add to that the `buffer` argument
        to expand the bounding box in all dimensions and discretize it into voxels of `voxelsize` xyz length. In this case, 
        the `center` argument is ignored.
        If a list of [x, y, z] lengths is given in Angstrom, it will create a box of those dimensions centered around the `center` 
        argument.
    center : list
        The [x, y, z] coordinates of the center. Use this only in combination with the `boxsize` argument.
    voxelsize : float
        The voxel size in A
    """
    if boxsize is None:
        # Calculate the bbox and the number of voxels
        [bb_min, bb_max] = boundingBox(mol)
        bb_min -= buffer
        bb_max += buffer
        nvoxels = np.ceil((bb_max - bb_min) / voxelsize).astype(int) + 1  # TODO: Why the +1?
    else:
        boxsize = np.array(boxsize)
        center = np.array(center)
        nvoxels = np.ceil(boxsize / voxelsize).astype(int)
        bb_min = center - (boxsize / 2)  
    # Calculate grid centers
    centers = _getGridCenters(*list(nvoxels), voxelsize) + bb_min
    centers = centers.reshape(np.prod(nvoxels), 3).copy()  # Copy otherwise C code reads it wrong
    return centers, nvoxels


def getVoxelDescriptors(mol, boxsize=None, voxelsize=1, buffer=0, center=None, usercenters=None, userchannels=None, 
                        usercoords=None, aromaticNitrogen=False, method='C', version=2, validitychecks=True):
    """ Calculate descriptors of atom properties for voxels in a grid bounding the Molecule object.

    Parameters
    ----------
    mol : Molecule or SmallMol object
        A molecule
    boxsize : list
        If this argument is None, the function will calculate the bounding box of the molecule, add to that the `buffer` argument
        to expand the bounding box in all dimensions and discretize it into voxels of `voxelsize` xyz length. In this case, 
        the `center` argument is ignored.
        If a list of [x, y, z] lengths is given in Angstrom, it will create a box of those dimensions centered around the `center` 
        argument.
    voxelsize : float
        The voxel size in A
    buffer : float
        The buffer space to add to the bounding box. This adds zeros to the grid around the protein so that properties
        which are at the edge of the box can be found in the center of one. Should be usually set to localimagesize/2.
    center : list
        The [x, y, z] coordinates of the center. Use this only in combination with the `boxsize` argument.
    usercenters : np.ndarray
        A 2D array specifying the centers of the voxels. If None is given, it will calculate the centers from the above options
    userchannels : np.ndarray
        A 2D array of size (mol.numAtoms, nchannels) where nchannels is the number of channels we want to have.
        Each column i then has True (or a float) in the rows of the atoms which belong to channel i and False (or 0) 
        otherwise. Such boolean arrays can be obtained for example by using mol.atomselect.
        If the array is boolean, each atom will get assigned its VdW radius. If the array is float, these floats will 
        be used as the corresponding atom radii. Make sure the numpy array is of dtype=bool if passing boolean values.
        If no channels are given, the default ('hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 
        'positive_ionizable', 'negative_ionizable', 'metal', 'occupancies') channels will be used.
    usercoords : np.ndarray
        A numpy array containing the molecule coordinates. You can use this with userchannels and usercenters instead 
        of passing a `mol` object (set it to None if that's the case). 
    validitychecks : bool
        Set to False to disable validity checks for atomtyping. This improves performance but only use it if you
        are sure you have properly prepared your molecules.

    Returns
    -------
    features : np.ndarray
        A 2D array of size (centers, channels) containing the effect of each channel in the voxel with that center. 
    centers : np.ndarray
        A list of the centers of all boxes
    N : np.ndarray
        Is returned only when no user centers are passed. It corresponds to the number of centers in each of the x,y,z
        dimensions
    method : str
        Voxel descriptors can be calculated either with our C implementation or CUDA or NUMBA implementations.

    Examples
    --------
    >>> mol = Molecule('3PTB')
    >>> mol.filter('protein')
    >>> features, centers, N = getVoxelDescriptors(mol, buffer=8)
    >>> # Features can be reshaped to a 4D array (3D for each grid center in xyz, 1D for the properties) like this:
    >>> features = features.reshape(N[0], N[1], N[2], features.shape[1])
    >>> # The user can provide his own centers
    >>> features, centers = getVoxelDescriptors(mol, usercenters=[[0, 0, 0], [16, 24, -5]])
    """
    channels = userchannels
    if channels is None:
        channels, mol = getChannels(mol, aromaticNitrogen, version, validitychecks)

    if channels.dtype == bool:
        # Calculate for each channel the atom sigmas
        sigmas = _getChannelRadii(mol.element)
        channels = sigmas[:, np.newaxis] * channels.astype(float)

    nvoxels = None
    centers = usercenters
    if centers is None:
        centers, nvoxels = getCenters(mol, buffer, boxsize, center, voxelsize)

    coords = usercoords
    if coords is None:
        coords = mol.get('coords')
    if coords.ndim == 3:
        if coords.shape[2] != 1:
            raise RuntimeError('Only a single set of coordinates should be passed for voxelixation. ' \
                               'Make sure your coordinates are either 3D with a last dim of 1 or 2D.')
        else:
            coords = coords[:, :, 0]

    # Calculate features
    if method.upper() == 'C':
        features = _getOccupancyC(coords, centers.copy(), channels)
    elif method.upper() == 'CUDA':
        features = _getOccupancyCUDA(coords, centers, channels)
    elif method.upper() == 'NUMBA':
        features = _getOccupancyNUMBA(coords, centers, channels, 5)

    if nvoxels is None:
        return features, centers
    else:
        return features, centers, nvoxels


def _getPropertiesRDkit(smallmol):
    """
    Returns ndarray of shape (n_atoms x n_properties) molecule atom types,
    according to the following definitions and order:
        0. Hydrophibic
        1. Aromatic
        2. Acceptor
        3. Donor
        4. - Ionizable
        5. + Ionizable
        6. Metal (empty)
        7. Occupancy (No hydrogens)
    """
    from moleculekit.smallmol.util import factory
    n_atoms = smallmol.numAtoms

    atom_mapping = {"Hydrophobe": 0,
                    "LumpedHydrophobe": 0,
                    "Aromatic": 1,
                    "Acceptor": 2,
                    "Donor": 3,
                    "PosIonizable": 4,
                    "NegIonizable": 5}

    feats = factory.GetFeaturesForMol(smallmol._mol)
    properties = np.zeros((n_atoms, 8), dtype=bool)

    for feat in feats:
        fam = feat.GetFamily()
        if fam not in atom_mapping:  # Non relevant property
            continue
        properties[feat.GetAtomIds(), atom_mapping[fam]] = 1

    # Occupancy, ignoring hydrogens.
    properties[:, 7] = smallmol.get('element') != 'H'
    return properties


def _getAtomtypePropertiesPDBQT(mol):
    """ Matches PDBQT atom types to specific properties
    ('hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable', 'negative_ionizable', 'metal')

    Parameters
    ----------
    mol :
        A Molecule object

    Returns
    -------
    properties : dict
        A dictionary of atom masks for the Molecule showing for each property (key) which atoms (value) belong to it.
    """
    """ AutoDock 4 atom types http://autodock.scripps.edu/faqs-help/faq/faqsection_view?section=Scientific%20Questions
    #        --     ----  -----  -------  --------  ---  ---  -  --  -- --
    atom_par H      2.00  0.020   0.0000   0.00051  0.0  0.0  0  -1  -1  3    # Non H-bonding Hydrogen
    atom_par HD     2.00  0.020   0.0000   0.00051  0.0  0.0  2  -1  -1  3    # Donor 1 H-bond Hydrogen
    atom_par HS     2.00  0.020   0.0000   0.00051  0.0  0.0  1  -1  -1  3    # Donor S Spherical Hydrogen
    atom_par C      4.00  0.150  33.5103  -0.00143  0.0  0.0  0  -1  -1  0    # Non H-bonding Aliphatic Carbon
    atom_par A      4.00  0.150  33.5103  -0.00052  0.0  0.0  0  -1  -1  0    # Non H-bonding Aromatic Carbon
    atom_par N      3.50  0.160  22.4493  -0.00162  0.0  0.0  0  -1  -1  1    # Non H-bonding Nitrogen
    atom_par NA     3.50  0.160  22.4493  -0.00162  1.9  5.0  4  -1  -1  1    # Acceptor 1 H-bond Nitrogen
    atom_par NS     3.50  0.160  22.4493  -0.00162  1.9  5.0  3  -1  -1  1    # Acceptor S Spherical Nitrogen
    atom_par OA     3.20  0.200  17.1573  -0.00251  1.9  5.0  5  -1  -1  2    # Acceptor 2 H-bonds Oxygen
    atom_par OS     3.20  0.200  17.1573  -0.00251  1.9  5.0  3  -1  -1  2    # Acceptor S Spherical Oxygen
    atom_par F      3.09  0.080  15.4480  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Fluorine
    atom_par Mg     1.30  0.875   1.5600  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Magnesium
    atom_par MG     1.30  0.875   1.5600  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Magnesium
    atom_par P      4.20  0.200  38.7924  -0.00110  0.0  0.0  0  -1  -1  5    # Non H-bonding Phosphorus
    atom_par SA     4.00  0.200  33.5103  -0.00214  2.5  1.0  5  -1  -1  6    # Acceptor 2 H-bonds Sulphur
    atom_par S      4.00  0.200  33.5103  -0.00214  0.0  0.0  0  -1  -1  6    # Non H-bonding Sulphur
    atom_par Cl     4.09  0.276  35.8235  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Chlorine
    atom_par CL     4.09  0.276  35.8235  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Chlorine
    atom_par Ca     1.98  0.550   2.7700  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Calcium
    atom_par CA     1.98  0.550   2.7700  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Calcium
    atom_par Mn     1.30  0.875   2.1400  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Manganese
    atom_par MN     1.30  0.875   2.1400  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Manganese
    atom_par Fe     1.30  0.010   1.8400  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Iron
    atom_par FE     1.30  0.010   1.8400  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Iron
    atom_par Zn     1.48  0.550   1.7000  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Zinc
    atom_par ZN     1.48  0.550   1.7000  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Zinc
    atom_par Br     4.33  0.389  42.5661  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Bromine
    atom_par BR     4.33  0.389  42.5661  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Bromine
    atom_par I      4.72  0.550  55.0585  -0.00110  0.0  0.0  0  -1  -1  4    # Non H-bonding Iodine
    """
    from collections import OrderedDict
    props = OrderedDict()
    atomtypes = np.array([el.upper() for el in mol.atomtype])

    props['hydrophobic'] = (atomtypes == 'C') | (atomtypes == 'A')
    props['aromatic'] = atomtypes == 'A'
    props['hbond_acceptor'] = (atomtypes == 'NA') | (atomtypes == 'NS') | (atomtypes == 'OA') | (atomtypes == 'OS') | (
    atomtypes == 'SA')
    props['hbond_donor'] = _findDonors(mol, mol._getBonds())
    props['positive_ionizable'] = mol.charge > 0
    props['negative_ionizable'] = mol.charge < 0
    props['metal'] = (atomtypes == 'MG') | (atomtypes == 'ZN') | (atomtypes == 'MN') | \
                     (atomtypes == 'CA') | (atomtypes == 'FE')
    props['occupancies'] = (atomtypes != 'H') & (atomtypes != 'HS') & (atomtypes != 'HD')

    channels = np.zeros((len(atomtypes), len(props)), dtype=bool)
    for i, p in enumerate(_order):
        channels[:, i] = props[p]
    return channels


def _findDonors(mol, bonds):
    """ Finds atoms connected to HD and HS atoms to find the heavy atom donor (O or N)

    Parameters
    ----------
    mol :
        A Molecule object
    bonds : np.ndarray
        An array of atom bonds

    Returns
    -------
    donors : np.ndarray
        Boolean array indicating the donor heavy atoms in Mol
    """
    donors = np.zeros(mol.numAtoms, dtype=bool)
    hydrogens = np.where((mol.element == 'HD') | (mol.element == 'HS'))[0]
    for h in hydrogens:
        partners = bonds[bonds[:, 0] == h, 1]
        partners = np.hstack((partners, bonds[bonds[:, 1] == h, 0]))
        for p in partners:
            if mol.name[p][0] == 'N' or mol.name[p][0] == 'O':
                donors[p] = True
    return donors


def _getOccupancyC(coords, centers, channelsigmas):
    """ Calls the C code to calculate the voxels values for each property."""
    centers = centers.astype(np.float64)
    channelsigmas = channelsigmas.astype(np.float64)

    nchannels = channelsigmas.shape[1]
    occus = np.zeros((centers.shape[0], nchannels))

    occupancylib.descriptor_ext(centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       channelsigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       occus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       ctypes.c_int(occus.shape[0]),  # n of centers
                       ctypes.c_int(coords.shape[0]),  # n of atoms
                       ctypes.c_int(nchannels))  # n of channels
    return occus


@jit('float64[:,:](float32[:,:], float64[:,:], float64[:,:], float64)', nopython=True)
def _getOccupancyNUMBA(coords, centers, channelsigmas, trunc):
    ncenters = centers.shape[0]
    natoms = coords.shape[0]
    nchannels = channelsigmas.shape[1]
    trunc = trunc * trunc  # Since we calculate the d**2 we need to get trunc**2
    occus = np.zeros((ncenters, nchannels))
    for a in range(natoms):
        coo = coords[a, :]
        atomsigmas = channelsigmas[a, :]

        for c in range(ncenters):
            cent = centers[c, :]
            d = np.zeros(3)
            d[0] = coo[0] - cent[0]
            d[1] = coo[1] - cent[1]
            d[2] = coo[2] - cent[2]
            d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if d2 < trunc:
                d1 = 1 / sqrt(d2)
                for h in range(nchannels):
                    if atomsigmas[h] == 0:
                        continue
                    x = atomsigmas[h] * d1
                    x3 = x * x * x
                    x12 = x3 * x3 * x3 * x3
                    value = 1 - exp(-x12)
                    occus[c, h] = max(occus[c, h], value)
    return occus


def _memsetArray(array, val=0, threadsperblock=256):
    from math import ceil
    totalelem = np.prod(array.shape)
    nblocks = ceil(totalelem / threadsperblock)
    _memsetArrayCUDAkernel[nblocks, threadsperblock](array.reshape(totalelem), val)

@cuda.jit
def _memsetArrayCUDAkernel(array, val):
    threadidx = (cuda.threadIdx.x + (cuda.blockDim.x * cuda.blockIdx.x))
    if threadidx >= array.shape[0]:
        return
    array[threadidx] = val

def _getOccupancyCUDA(coords, centers, channelsigmas, trunc=5, device=0, resD=None, asnumpy=True, threadsperblock=256):
    #cuda.select_device(device)
    if resD is None:
        resD = cuda.device_array((centers.shape[0], channelsigmas.shape[1]), dtype=np.float32)
    _memsetArray(resD, val=0)

    natomblocks = int(np.ceil(coords.shape[0] / threadsperblock))
    blockspergrid = (centers.shape[0], natomblocks)

    centers = cuda.to_device(centers)
    coords = cuda.to_device(coords)
    channelsigmas = cuda.to_device(channelsigmas)
    _getOccupancyCUDAkernel[blockspergrid, threadsperblock](resD, coords, centers, channelsigmas, trunc * trunc)

    if asnumpy:
        return resD.copy_to_host()

@cuda.jit
def _getOccupancyCUDAkernel(occus, coords, centers, channelsigmas, trunc):
    centeridx = cuda.blockIdx.x
    blockidx = cuda.blockIdx.y
    atomidx = (cuda.threadIdx.x + (cuda.blockDim.x * blockidx))

    if atomidx >= coords.shape[0] or centeridx >= centers.shape[0]:
        return

    # TODO: Can remove this. Barely any speedup
    centcoor = cuda.shared.array(shape=(3), dtype=numba.float32)
    centcoor[0] = centers[centeridx, 0]
    centcoor[1] = centers[centeridx, 1]
    centcoor[2] = centers[centeridx, 2]
    cuda.syncthreads()

    dx = coords[atomidx, 0] - centcoor[0]
    dy = coords[atomidx, 1] - centcoor[1]
    dz = coords[atomidx, 2] - centcoor[2]
    d2 = dx * dx + dy * dy + dz * dz
    if d2 >= trunc:
        return

    d1 = 1 / sqrt(d2)
    for h in range(channelsigmas.shape[1]):
        if channelsigmas[atomidx, h] == 0:
            continue
        x = channelsigmas[atomidx, h] * d1
        value = 1 - exp(-(x ** 12))
        cuda.atomic.max(occus, (centeridx, h), value)


import unittest
class _TestVoxel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.tools.preparation import proteinPrepare
        from moleculekit.tools.autosegment import autoSegment

        self.testf = os.path.join(home(), 'test-data', 'test-voxeldescriptors')
        mol = Molecule(os.path.join(self.testf, '3PTB.pdb'))
        mol.filter('protein')
        mol = autoSegment(mol, field='both')
        mol = proteinPrepare(mol)
        mol.bonds = mol._guessBonds()
        self.mol = mol

    def test_radii(self):
        sigmas = _getChannelRadii(self.mol.element)
        refsigmas = np.load(os.path.join(self.testf, '3PTB_sigmas.npy'), allow_pickle=True)
        assert np.allclose(sigmas, refsigmas)

    def test_celecoxib(self):
        from moleculekit.smallmol.smallmol import SmallMol
        sm = SmallMol(os.path.join(self.testf, 'celecoxib.mol2'))
        features, centers, nvoxels = getVoxelDescriptors(sm, buffer=1, version=2)
        reffeatures, refcenters, refnvoxels = np.load(os.path.join(self.testf, 'celecoxib_voxres.npy'), allow_pickle=True)
        assert np.allclose(features, reffeatures)
        assert np.array_equal(centers, refcenters)
        assert np.array_equal(nvoxels, refnvoxels)

    def test_ledipasvir(self):
        from moleculekit.smallmol.smallmol import SmallMol
        sm = SmallMol(os.path.join(self.testf, 'ledipasvir.mol2'))
        features, centers, nvoxels = getVoxelDescriptors(sm, buffer=1, version=2)
        reffeatures, refcenters, refnvoxels = np.load(os.path.join(self.testf, 'ledipasvir_voxres.npy'), allow_pickle=True)
        assert np.allclose(features, reffeatures)
        assert np.array_equal(centers, refcenters)
        assert np.array_equal(nvoxels, refnvoxels)

    def test_old_voxelization(self):
        from moleculekit.molecule import Molecule
        mol = Molecule(os.path.join(self.testf, '3ptb.pdbqt'))
        mol.element[mol.element == 'CA'] = 'Ca'
        features, centers, nvoxels = getVoxelDescriptors(mol, buffer=8, voxelsize=1, version=1)
        reffeatures, refcenters, refnvoxels = np.load(os.path.join(self.testf, '3PTB_voxres_old.npy'), allow_pickle=True)
        assert np.allclose(features, reffeatures)
        assert np.array_equal(centers, refcenters)
        assert np.array_equal(nvoxels, refnvoxels)

    def test_featC(self):
        features, centers, nvoxels = getVoxelDescriptors(self.mol, method='C', version=2)
        reffeatures, refcenters, refnvoxels = np.load(os.path.join(self.testf, '3PTB_voxres.npy'), allow_pickle=True)
        assert np.allclose(features, reffeatures)
        assert np.array_equal(centers, refcenters)
        assert np.array_equal(nvoxels, refnvoxels)

    def test_featNUMBA(self):
        features, centers, nvoxels = getVoxelDescriptors(self.mol, method='NUMBA', version=2)
        reffeatures, refcenters, refnvoxels = np.load(os.path.join(self.testf, '3PTB_voxres.npy'), allow_pickle=True)
        assert np.allclose(features, reffeatures)
        assert np.array_equal(centers, refcenters)
        assert np.array_equal(nvoxels, refnvoxels)

    def test_compare_c_numba(self):
        import numpy as np
        centers = np.load(os.path.join(self.testf, '3PTB_centers_inp.npy'), allow_pickle=True)
        coords = np.load(os.path.join(self.testf, '3PTB_coords_inp.npy'), allow_pickle=True)
        sigmas = np.load(os.path.join(self.testf, '3PTB_channels_inp.npy'), allow_pickle=True)
        centers = centers[::10, :].copy()

        res_C = _getOccupancyC(coords, centers, sigmas)
        # res_cuda = _getOccupancyCUDA(coords, centers, sigmas, 5)
        res_numba = _getOccupancyNUMBA(coords, centers, sigmas, 5)

        # assert np.abs(res_C - res_cuda).max() < 1e-4
        assert np.abs(res_C - res_numba).max() < 1e-4

if __name__ == '__main__':
    unittest.main(verbosity=2)




