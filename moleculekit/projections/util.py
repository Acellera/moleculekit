# (c) 2015-2022 Acellera Ltd http://www.acellera.com
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
    from moleculekit.distance_utils import dist_trajectory

    selfdist = np.array_equal(sel1, sel2)
    sel1 = np.where(sel1)[0].astype(np.uint32)
    sel2 = np.where(sel2)[0].astype(np.uint32)

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
        digitized_chains = np.zeros(mol.numAtoms, dtype=np.uint32)
    elif periodic == "chains":
        digitized_chains = np.unique(mol.chain, return_inverse=True)[1].astype(
            np.uint32
        )
    elif periodic == "selections":
        digitized_chains = np.ones(mol.numAtoms, dtype=np.uint32)
        digitized_chains[sel2] = 2
    else:
        raise RuntimeError(f"Invalid periodic option {periodic}")

    shape = (mol.numFrames, len(sel1) * len(sel2))
    if selfdist:
        shape = (mol.numFrames, int((len(sel1) * (len(sel2) - 1)) / 2))
    results = np.zeros(shape, dtype=np.float32)

    dist_trajectory(
        coords,
        box,
        sel1,
        sel2,
        digitized_chains,
        selfdist,
        periodic is not None,
        results,
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


def get_reduced_distances(
    mol,
    sel1,
    sel2,
    periodic,
    metric="distances",
    threshold=8,
    truncate=None,
    reduction1="closest",
    reduction2="closest",
    pairs=False,
):
    from moleculekit.distance_utils import (
        dist_trajectory_reduction,
        dist_trajectory_reduction_pairs,
    )
    from moleculekit.periodictable import periodictable

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

    selfdist = np.array_equal(sel1, sel2)

    # Digitize chains to not do PBC calculations of the same chain
    if periodic is None:  # Won't be used since box is 0
        digitized_chains = np.zeros(mol.numAtoms, dtype=np.uint32)
    elif periodic == "chains":
        digitized_chains = np.unique(mol.chain, return_inverse=True)[1].astype(
            np.uint32
        )
    elif periodic == "selections":
        digitized_chains = np.ones(mol.numAtoms, dtype=np.uint32)
        for i in range(sel2.shape[0]):
            digitized_chains[sel2[i, :]] = 2

    groups1 = [np.where(sel1[i, :])[0].tolist() for i in range(sel1.shape[0])]
    groups2 = [np.where(sel2[i, :])[0].tolist() for i in range(sel2.shape[0])]

    if pairs and len(groups1) != len(groups2):
        raise RuntimeError(
            "If `pairs=True` mode is used, the number of groups in sel1 should match the number of groups in sel2."
        )

    if selfdist:
        mindist = np.zeros(
            (mol.numFrames, int((len(groups1) * (len(groups2) - 1)) / 2)),
            dtype=np.float32,
        )
    else:
        if not pairs:
            mindist = np.zeros(
                (mol.numFrames, len(groups1) * len(groups2)), dtype=np.float32
            )  # Preparing the return array
        else:
            mindist = np.zeros(
                (mol.numFrames, len(groups1)), dtype=np.float32
            )  # Preparing the return array

    # import time
    # t = time.time()
    reduction_map = {"closest": 0, "com": 1}

    digitized_chains1 = np.array(
        [digitized_chains[gg[0]] for gg in groups1], dtype=np.uint32
    )
    digitized_chains2 = np.array(
        [digitized_chains[gg[0]] for gg in groups2], dtype=np.uint32
    )
    masses = np.array([periodictable[el].mass for el in mol.element], dtype=np.float32)
    if not pairs:
        dist_trajectory_reduction(
            coords,
            box,
            groups1,
            groups2,
            digitized_chains1,
            digitized_chains2,
            selfdist,
            periodic is not None,
            masses,
            reduction_map[reduction1.lower()],
            reduction_map[reduction2.lower()],
            mindist,
        )
    else:
        dist_trajectory_reduction_pairs(
            coords,
            box,
            groups1,
            groups2,
            digitized_chains1,
            digitized_chains2,
            periodic is not None,
            masses,
            reduction_map[reduction1.lower()],
            reduction_map[reduction2.lower()],
            mindist,
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


def pp_calcMinDistances(
    mol, sel1, sel2, periodic, metric="distances", threshold=8, truncate=None
):
    return get_reduced_distances(
        mol=mol,
        sel1=sel1,
        sel2=sel2,
        periodic=periodic,
        metric=metric,
        threshold=threshold,
        truncate=truncate,
    )


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
