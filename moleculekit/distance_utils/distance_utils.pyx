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
cdef FLOAT32_t _calc_com(
        FLOAT32_t[:,:,:] coords,
        int f,
        vector[int] &group,
        FLOAT32_t[:] masses,
        FLOAT32_t[:,:] com,
        int cg,
    ):
    cdef int k, gatm
    cdef float total_mass = 0
    cdef float comx = 0
    cdef float comy = 0
    cdef float comz = 0

    # Calculate center of mass of group1
    for gatm in group:
        comx += coords[gatm, 0, f] * masses[gatm]
        comy += coords[gatm, 1, f] * masses[gatm]
        comz += coords[gatm, 2, f] * masses[gatm]
        total_mass += masses[gatm]

    com[cg, 0] = comx / total_mass
    com[cg, 1] = comy / total_mass
    com[cg, 2] = comz / total_mass


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef FLOAT32_t _dist2(
        FLOAT32_t[:] coord1,
        FLOAT32_t[:] coord2,
        FLOAT32_t[:] box,
        bool diff_chain,
        bool pbc,
    ):
    cdef FLOAT32_t dist2, dx, dy, dz

    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    
    if pbc and diff_chain:  # Only do PBC if chains are different    
        dx = dx - box[0] * round(dx / box[0])
        dy = dy - box[1] * round(dy / box[1])
        dz = dz - box[2] * round(dz / box[2])

    return dx * dx + dy * dy + dz * dz


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def dist_trajectory_reduction(
        FLOAT32_t[:,:,:] coords,
        FLOAT32_t[:,:] box,
        vector[vector[int]] &groups1,
        vector[vector[int]] &groups2,
        UINT32_t[:] digitized_chains1,
        UINT32_t[:] digitized_chains2,
        bool selfdist,
        bool pbc,
        FLOAT32_t[:] masses,
        int reduction1,
        int reduction2,
        FLOAT32_t[:,:] results,
    ):
    cdef int f, g1, g2, i, j, k, g1atm, g2atm, g2start, idx
    cdef int n_atoms = coords.shape[0]
    cdef int n_frames = coords.shape[2]
    cdef int n_groups1 = groups1.size()
    cdef int n_groups2 = groups2.size()
    cdef FLOAT32_t[:,:] com = np.zeros((2, 3), dtype=FLOAT32)
    cdef FLOAT32_t dist2, mindist
    cdef FLOAT32_t[:] curr_box
    cdef UINT32_t chain1, chain2
    cdef bool diff_chain
    cdef FLOAT32_t[:] coor1, coor2
    cdef vector[int] group1, group2
    cdef vector[int] trash
    trash.push_back(0)

    for f in range(n_frames):
        idx = 0
        curr_box = box[:, f]

        for g1 in range(n_groups1):
            g2start = 0
            if selfdist:
                g2start = g1 + 1

            group1 = groups1[g1]
            if reduction1 == 1: # COM
                _calc_com(coords, f, group1, masses, com, 0)
                group1 = trash # Assign this so it loops once over the "group"

            for g2 in range(g2start, n_groups2):
                group2 = groups2[g2]
                if reduction2 == 1: # COM
                    _calc_com(coords, f, group2, masses, com, 1)
                    group2 = trash  # Assign this so it loops once over the "group"

                mindist = -1
                diff_chain = digitized_chains1[g1] != digitized_chains2[g2]

                for g1atm in group1:
                    coor1 = coords[g1atm, :, f]
                    if reduction1 == 1:
                        coor1 = com[0, :]

                    for g2atm in group2:
                        coor2 = coords[g2atm, :, f]
                        if reduction2 == 1:
                            coor2 = com[1, :]

                        dist2 = _dist2(coor1, coor2, curr_box, diff_chain, pbc)

                        if dist2 < mindist or mindist < 0:
                            mindist = dist2

                results[f, idx] = sqrt(mindist)
                idx += 1

    return results


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def dist_trajectory_reduction_pairs(
        FLOAT32_t[:,:,:] coords,
        FLOAT32_t[:,:] box,
        vector[vector[int]] &groups1,
        vector[vector[int]] &groups2,
        UINT32_t[:] digitized_chains1,
        UINT32_t[:] digitized_chains2,
        bool pbc,
        FLOAT32_t[:] masses,
        int reduction1,
        int reduction2,
        FLOAT32_t[:,:] results,
    ):
    # Iterates over group pairs instead of doing all-vs-all groups
    cdef int f, g1, g2, i, j, k, g1atm, g2atm, g2start, idx
    cdef int n_atoms = coords.shape[0]
    cdef int n_frames = coords.shape[2]
    cdef int n_groups = groups1.size()
    cdef FLOAT32_t[:,:] com = np.zeros((2, 3), dtype=FLOAT32)
    cdef FLOAT32_t dist2, mindist
    cdef FLOAT32_t[:] curr_box
    cdef UINT32_t chain1, chain2
    cdef bool diff_chain
    cdef FLOAT32_t[:] coor1, coor2
    cdef vector[int] group1, group2
    cdef vector[int] trash
    trash.push_back(0)

    for f in range(n_frames):
        idx = 0
        curr_box = box[:, f]

        for g in range(n_groups):
            group1 = groups1[g]
            group2 = groups2[g]

            if reduction1 == 1: # COM
                _calc_com(coords, f, group1, masses, com, 0)
                group1 = trash # Assign this so it loops once over the "group"
            if reduction2 == 1: # COM
                _calc_com(coords, f, group2, masses, com, 1)
                group2 = trash  # Assign this so it loops once over the "group"

            mindist = -1
            diff_chain = digitized_chains1[g] != digitized_chains2[g]

            for g1atm in group1:
                coor1 = coords[g1atm, :, f]
                if reduction1 == 1:
                    coor1 = com[0, :]

                for g2atm in group2:
                    coor2 = coords[g2atm, :, f]
                    if reduction2 == 1:
                        coor2 = com[1, :]

                    dist2 = _dist2(coor1, coor2, curr_box, diff_chain, pbc)

                    if dist2 < mindist or mindist < 0:
                        mindist = dist2

            results[f, idx] = sqrt(mindist)
            idx += 1

    return results