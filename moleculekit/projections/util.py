# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import logging
import moleculekit.molecule

logger = logging.getLogger(__name__)


def pp_calcDistances(mol, sel1, sel2, metric='distances', threshold=8, pbc=True, gap=1, truncate=None):
    distances = _distanceArray(mol, sel1, sel2, pbc)
    distances = _postProcessDistances(distances, sel1, sel2, truncate)

    if metric == 'contacts':
        # from scipy.sparse import lil_matrix
        # metric = lil_matrix(distances <= threshold)
        metric = distances <= threshold
    elif metric == 'distances':
        metric = distances.astype(dtype=np.float32)
    else:
        raise NameError('The metric you asked for is not supported. Check spelling and documentation')
    return metric


def pp_calcMinDistances(mol, sel1, sel2, metric='distances', threshold=8, pbc=True, gap=1, truncate=None):
    import os
    import ctypes
    from moleculekit.home import home

    # Converting non-grouped boolean atomselection to group-style atomselections
    if np.ndim(sel1) != 2:
        sel1idx = tuple(np.where(sel1)[0])
        sel1 = np.zeros((len(sel1idx), len(sel1)), dtype=bool)
        sel1[range(sel1.shape[0]), sel1idx] = True
    if np.ndim(sel2) != 2:
        sel2idx = tuple(np.where(sel2)[0])
        sel2 = np.zeros((len(sel2idx), len(sel2)), dtype=bool)
        sel2[range(sel2.shape[0]), sel2idx] = True

    coords = mol.coords
    box = mol.box
    if pbc:
        if box is None or np.sum(box) == 0:
            raise RuntimeError('No periodic box dimensions given in the molecule/trajectory. '
                            'If you want to calculate distance without wrapping, set the pbc option to False')
    else:
        box = np.zeros((3, coords.shape[2]), dtype=np.float32)

    if box.shape[1] != coords.shape[2]:
        raise RuntimeError('Different number of frames in mol.coords and mol.box. '
                            'Please ensure they both have the same number of frames')

    # Converting from 2D boolean atomselect array to 2D int array where each row starts with the indexes of the boolean
    groups1 = np.ones((sel1.shape[0], mol.numAtoms), dtype=np.int32) * -1
    groups2 = np.ones((sel2.shape[0], mol.numAtoms), dtype=np.int32) * -1
    for i in range(sel1.shape[0]):
        idx = np.where(sel1[i, :])[0]
        groups1[i, 0:len(idx)] = idx
    for i in range(sel2.shape[0]):
        idx = np.where(sel2[i, :])[0]
        groups2[i, 0:len(idx)] = idx

    selfdist = np.array_equal(sel1, sel2)

    # Running the actual calculations
    lib = ctypes.cdll.LoadLibrary(os.path.join(home(libDir=True), 'mindist_ext.so'))
    mindist = np.zeros((mol.numFrames, len(groups1) * len(groups2)), dtype=np.float32)  # Preparing the return array
    if selfdist:
        mindist = np.zeros((mol.numFrames, int((len(groups1) * (len(groups2)-1))/2)), dtype=np.float32)

    #import time
    #t = time.time()
    lib.mindist_trajectory(coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           box.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           groups1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                           groups2.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                           ctypes.c_int(len(groups1)),
                           ctypes.c_int(len(groups2)),
                           ctypes.c_int(mol.numAtoms),
                           ctypes.c_int(mol.numFrames),
                           ctypes.c_int(int(pbc)),
                           ctypes.c_int(int(selfdist)),
                           mindist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    #print(time.time() - t)

    if truncate is not None:
        mindist[mindist > truncate] = truncate

    if metric == 'contacts':
        mindist = mindist <= threshold
    elif metric == 'distances':
        mindist = mindist.astype(dtype=np.float32)
    else:
        raise NameError('The metric you asked for is not supported. Check spelling and documentation')
    return mindist


def _postProcessDistances(distances, sel1, sel2, truncate):
    # distances is a list of numpy arrays. Each numpy array is numFrames x numSel1. The list is length numSel2
    # Setting upper triangle to -1 if same selections
    if np.array_equal(sel1, sel2):
        for i in range(len(distances)):
                distances[i][:, range(i + 1)] = -1

    if np.ndim(distances[0]) > 1:  # 2D data
        distances = np.concatenate(distances, axis=1)
    else:  # 1D data
        distances = np.vstack(distances).transpose()

    if np.array_equal(sel1, sel2):
        distances = distances[:, np.all(distances != -1, 0)]

    if truncate is not None:
        distances[distances > truncate] = truncate
    return np.atleast_1d(np.squeeze(distances))


def _distanceArray(mol, sel1, sel2, pbc):
    numsel1 = np.sum(sel1)
    numsel2 = np.sum(sel2)
    coords1 = mol.coords[sel1, :, :]
    coords2 = mol.coords[sel2, :, :]

    distances = []
    for j in range(numsel2):
        coo2 = coords2[j, :, :]  # 3 x numframes array
        dists = coords1 - coo2
        if pbc:
            if mol.box is None or np.sum(mol.box) == 0:
                raise NameError(
                    'No periodic box dimensions given in the molecule/trajectory. If you want to calculate distance without wrapping, set the pbc option to False')
            dists = _wrapDistances(mol.box, dists, _findDiffChain(mol, sel1, sel2, j, range(numsel1)))
        dists = np.transpose(np.sqrt(np.sum(dists * dists, 1)))
        distances.append(dists)
    return distances


def _findDiffChain(mol, sel1, sel2, i, others):
    if np.array_equal(sel1, sel2) and len(mol.chain) > 0:
        chain = mol.get('chain', sel=sel1)
        diffchain = chain[others] != chain[i]
    else:
        diffchain = None
    return diffchain


def _wrapDistances(box, dist, diffchain):
    if diffchain is not None:
        dist[diffchain, :, :] -= box * np.round(dist[diffchain, :, :] / box)
    else:
        dist = dist - box * np.round(dist / box)
    return dist

