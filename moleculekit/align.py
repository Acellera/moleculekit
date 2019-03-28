from numba import jit
import numpy as np


@jit('Tuple((float32[:, :], float64))(float32[:, :], float32[:, :])', nopython=True, nogil=True)
def _pp_measure_fit(P, Q):
    """
    WARNING: ASSUMES CENTERED COORDINATES!!!!

    PP_MEASURE_FIT - molecule alignment function.
    For documentation see http://en.wikipedia.org/wiki/Kabsch_algorithm
    the Kabsch algorithm is a method for calculating the optimal
    rotation matrix that minimizes the RMSD (root mean squared deviation)
    between two paired sets of points
    """
    covariance = np.dot(P.T, Q)

    (V, S, W) = np.linalg.svd(covariance)  # Matlab svd returns the W transposed compared to numpy.svd
    W = W.T

    E0 = np.sum(P * P) + np.sum(Q * Q)
    RMSD = E0 - (2 * np.sum(S.ravel()))
    RMSD = np.sqrt(np.abs(RMSD / P.shape[0]))

    d = np.sign(np.linalg.det(W) * np.linalg.det(V))
    z = np.eye(3).astype(P.dtype)
    z[2, 2] = d
    U = np.dot(np.dot(W, z), V.T)
    return U, RMSD


@jit('float32[:, :, :](float32[:, :, :], float32[:, :, :], int64[:], int64[:], int64[:], int64, boolean)', nopython=True,
     nogil=True)
def _pp_align(coords, refcoords, sel, refsel, frames, refframe, matchingframes):
    newcoords = coords.copy()
    for f in frames:
        P = coords[sel, :, f]
        if matchingframes:
            Q = refcoords[refsel, :, f]
        else:
            Q = refcoords[refsel, :, refframe]
        all1 = coords[:, :, f]

        centroidP = np.zeros(3, dtype=P.dtype)
        centroidQ = np.zeros(3, dtype=Q.dtype)
        for i in range(3):
            centroidP[i] = np.mean(P[:, i])
            centroidQ[i] = np.mean(Q[:, i])

        (rot, tmp) = _pp_measure_fit(P - centroidP, Q - centroidQ)

        all1 = all1 - centroidP
        # Rotating mol
        all1 = np.dot(all1, rot.T)
        # Translating to centroid of refmol
        all1 = all1 + centroidQ
        newcoords[:, :, f] = all1
    return newcoords