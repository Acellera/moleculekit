# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import logging

logger = logging.getLogger(__name__)


def pp_calcDistances(
    mol, sel1, sel2, periodic, metric="distances", threshold=8, gap=1, truncate=None
):
    import os
    import ctypes
    from moleculekit.home import home

    selfdist = np.array_equal(sel1, sel2)

    sel1 = np.where(sel1)[0].astype(np.int32)
    sel2 = np.where(sel2)[0].astype(np.int32)

    coords = mol.coords
    box = mol.box
    if periodic is not None:
        if box is None or np.sum(box) == 0:
            raise RuntimeError(
                "No periodic box dimensions given in the molecule/trajectory. "
                "If you want to calculate distance without wrapping, set the periodic option to `None`"
            )
    else:
        box = np.zeros((3, coords.shape[2]), dtype=np.float32)

    if box.shape[1] != coords.shape[2]:
        raise RuntimeError(
            "Different number of frames in mol.coords and mol.box. "
            "Please ensure they both have the same number of frames"
        )

    # Digitize chains to not do PBC calculations of the same chain
    if periodic is None:  # Won't be used since box is 0
        digitized_chains = np.zeros(mol.numAtoms, dtype=np.int32)
    elif periodic == "chains":
        digitized_chains = np.unique(mol.chain, return_inverse=True)[1].astype(np.int32)
    elif periodic == "selections":
        digitized_chains = np.ones(mol.numAtoms, dtype=np.int32)
        digitized_chains[sel2] = 2

    # Running the actual calculations
    lib = ctypes.cdll.LoadLibrary(os.path.join(home(libDir=True), "dist_ext.so"))
    shape = (mol.numFrames, len(sel1) * len(sel2))
    if selfdist:
        shape = (mol.numFrames, int((len(sel1) * (len(sel2) - 1)) / 2))
    results = np.zeros(shape, dtype=np.float32)

    lib.dist_trajectory(
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        box.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sel1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        sel2.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        digitized_chains.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(sel1)),
        ctypes.c_int(len(sel2)),
        ctypes.c_int(mol.numAtoms),
        ctypes.c_int(mol.numFrames),
        ctypes.c_int(int(periodic is not None)),
        ctypes.c_int(int(selfdist)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    if truncate is not None:
        results[results > truncate] = truncate

    if metric == "contacts":
        results = results <= threshold
    elif metric == "distances":
        pass
    else:
        raise RuntimeError(
            "The metric you asked for is not supported. Check spelling and documentation"
        )
    return results


def pp_calcMinDistances(
    mol, sel1, sel2, periodic, metric="distances", threshold=8, gap=1, truncate=None
):
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
    if periodic is not None:
        if box is None or np.sum(box) == 0:
            raise RuntimeError(
                "No periodic box dimensions given in the molecule/trajectory. "
                "If you want to calculate distance without wrapping, set the `periodic` option to None"
            )
    else:
        box = np.zeros((3, coords.shape[2]), dtype=np.float32)

    if box.shape[1] != coords.shape[2]:
        raise RuntimeError(
            "Different number of frames in mol.coords and mol.box. "
            "Please ensure they both have the same number of frames"
        )

    # Converting from 2D boolean atomselect array to 2D int array where each row starts with the indexes of the boolean
    groups1 = np.ones((sel1.shape[0], mol.numAtoms), dtype=np.int32) * -1
    groups2 = np.ones((sel2.shape[0], mol.numAtoms), dtype=np.int32) * -1
    for i in range(sel1.shape[0]):
        idx = np.where(sel1[i, :])[0]
        groups1[i, 0 : len(idx)] = idx
    for i in range(sel2.shape[0]):
        idx = np.where(sel2[i, :])[0]
        groups2[i, 0 : len(idx)] = idx

    selfdist = np.array_equal(sel1, sel2)

    # Digitize chains to not do PBC calculations of the same chain
    if periodic is None:  # Won't be used since box is 0
        digitized_chains = np.zeros(mol.numAtoms, dtype=np.int32)
    elif periodic == "chains":
        digitized_chains = np.unique(mol.chain, return_inverse=True)[1].astype(np.int32)
    elif periodic == "selections":
        digitized_chains = np.ones(mol.numAtoms, dtype=np.int32)
        for i in range(sel2.shape[0]):
            digitized_chains[sel2[i, :]] = 2

    # Running the actual calculations
    lib = ctypes.cdll.LoadLibrary(os.path.join(home(libDir=True), "mindist_ext.so"))
    mindist = np.zeros(
        (mol.numFrames, len(groups1) * len(groups2)), dtype=np.float32
    )  # Preparing the return array
    if selfdist:
        mindist = np.zeros(
            (mol.numFrames, int((len(groups1) * (len(groups2) - 1)) / 2)),
            dtype=np.float32,
        )

    # import time
    # t = time.time()
    lib.mindist_trajectory(
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        box.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        groups1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        groups2.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        digitized_chains.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(groups1)),
        ctypes.c_int(len(groups2)),
        ctypes.c_int(mol.numAtoms),
        ctypes.c_int(mol.numFrames),
        ctypes.c_int(int(periodic is not None)),
        ctypes.c_int(int(selfdist)),
        mindist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    # print(time.time() - t)

    if truncate is not None:
        mindist[mindist > truncate] = truncate

    if metric == "contacts":
        mindist = mindist <= threshold
    elif metric == "distances":
        pass
    else:
        raise RuntimeError(
            "The metric you asked for is not supported. Check spelling and documentation"
        )
    return mindist


def _findDiffChain(mol, sel1, sel2, i, others):
    if np.array_equal(sel1, sel2) and len(mol.chain) > 0:
        chain = mol.get("chain", sel=sel1)
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
