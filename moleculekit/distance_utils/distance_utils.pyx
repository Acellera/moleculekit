# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport sqrt, round
from cython.parallel import prange

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
INT32 = np.int32
INT64 = np.int64
UINT32 = np.uint32
FLOAT32 = np.float32
FLOAT64 = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t
ctypedef np.uint32_t UINT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

import cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef FLOAT32_t _dist(
        FLOAT32_t[:,:,:] coords,
        FLOAT32_t[:,:] box,
        UINT32_t[:] digitized_chains,
        int i,
        int j,
        int f,
        bool pbc,
    ):
    cdef FLOAT32_t dist2, dx, dy, dz

    dx = coords[i, 0, f] - coords[j, 0, f]
    dy = coords[i, 1, f] - coords[j, 1, f]
    dz = coords[i, 2, f] - coords[j, 2, f]
    
    if pbc and (digitized_chains[i] != digitized_chains[j]):  # Only do PBC if chains are different    
        dx = dx - box[0, f] * round(dx / box[0, f])
        dy = dy - box[1, f] * round(dy / box[1, f])
        dz = dz - box[2, f] * round(dz / box[2, f])

    return dx * dx + dy * dy + dz * dz

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def dist_trajectory(
        FLOAT32_t[:,:,:] coords,
        FLOAT32_t[:,:] box,
        UINT32_t[:] sel1,
        UINT32_t[:] sel2,
        UINT32_t[:] digitized_chains,
        bool selfdist,
        bool pbc,
        FLOAT32_t[:,:] results,
    ):
    cdef int f, i, j, bstart, idx, s1atm, s2atm
    cdef int n_atoms = coords.shape[0]
    cdef int n_frames = coords.shape[2]
    cdef int n_sel1 = sel1.shape[0]
    cdef int n_sel2 = sel2.shape[0]
    cdef FLOAT32_t dist2

    for f in range(n_frames):
        idx = 0
        for i in range(n_sel1):
            s1atm = sel1[i]
            bstart = 0
            if selfdist:
                bstart = i + 1

            for j in range(bstart, n_sel2):
                s2atm = sel2[j]
                dist2 = _dist(coords, box, digitized_chains, s1atm, s2atm, f, pbc)
                results[f, idx] = sqrt(dist2)
                idx += 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mindist_trajectory(
        FLOAT32_t[:,:,:] coords,
        FLOAT32_t[:,:] box,
        INT32_t[:,:] groups1,
        INT32_t[:,:] groups2,
        UINT32_t[:] digitized_chains,
        bool selfdist,
        bool pbc,
        FLOAT32_t[:,:] results,
    ):
    cdef int f, g1, g2, i, j, g1atm, g2atm, g2start, idx
    cdef int n_atoms = coords.shape[0]
    cdef int n_frames = coords.shape[2]
    cdef int n_groups1 = groups1.shape[0]
    cdef int n_groups2 = groups2.shape[0]
    cdef FLOAT32_t dist2, mindist

    for f in range(n_frames):
        idx = 0
        for g1 in range(n_groups1):
            g2start = 0
            if selfdist:
                g2start = g1 + 1

            for g2 in range(g2start, n_groups2):
                mindist = -1

                for i in range(n_atoms):
                    g1atm = groups1[g1, i]
                    if g1atm == -1:
                        break

                    for j in range(n_atoms):
                        g2atm = groups2[g2, j]
                        if g2atm == -1:
                            break

                        dist2 = _dist(coords, box, digitized_chains, g1atm, g2atm, f, pbc)

                        if dist2 < mindist or mindist < 0:
                            mindist = dist2

                results[f, idx] = sqrt(mindist)
                idx += 1

    return results