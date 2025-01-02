# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round, fabs

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
UINT32 = np.uint32
FLOAT32 = np.float32
FLOAT64 = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint32_t UINT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

import cython


# https://en.wikipedia.org/wiki/Disjoint-set_data_structure
# Disjoint set implementation for finding the connected components of our molecule
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def disjoint_set_find(
        UINT32_t x,
        UINT32_t[:] parent,
    ):
    # # Path halving implementation (does not mark all parents exhaustively)
    # while parent[x] != x:
    #     parent[x] = parent[parent[x]]
    #     x = parent[x]

    # Path compression implementation
    if parent[x] != x:
        parent[x] = disjoint_set_find(parent[x], parent)
        return parent[x]
    else:
        return x

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def disjoint_set_merge(
        UINT32_t x,
        UINT32_t y,
        UINT32_t[:] parent,
        UINT32_t[:] size,
    ):
    # Replace nodes by roots
    x = disjoint_set_find(x, parent)
    y = disjoint_set_find(y, parent)

    if x == y:
        return # x and y are already in the same set

    if size[x] < size[y]:
        parent[x] = y
        size[y] += size[x]
    else:
        parent[y] = x
        size[x] += size[y]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_bonded_groups(
        UINT32_t[:,:] bonds,
        int n_atoms,
        UINT32_t[:] parent,
        UINT32_t[:] size,
    ):
    cdef int i
    cdef int n_bonds = bonds.shape[0]

    # Iterate over the bonds
    for i in range(n_bonds):
        disjoint_set_merge(bonds[i, 0], bonds[i, 1], parent, size)

    # Now everything is connected but not every node has the right parent marked

    # Iterate over the atoms to fix the final parent
    for i in range(n_atoms):
        disjoint_set_find(i, parent)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate(
        np.ndarray[UINT32_t, ndim=1] groups, 
        np.ndarray[FLOAT32_t, ndim=3] coords, 
        np.ndarray[FLOAT32_t, ndim=2] box,
        np.ndarray[UINT32_t, ndim=1] centersel, 
        np.ndarray[FLOAT32_t, ndim=1] center,
    ):
    cdef int f, i, g, a, start_idx, end_idx, k, n
    cdef int n_frames = coords.shape[2]
    cdef int n_groups = groups.shape[0]
    cdef int n_centersel = centersel.shape[0]
    cdef FLOAT32_t[:] half_box = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] box_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] grp_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t translation, diff

    if n_centersel == 0:
        for i in range(3):
            box_center[i] = center[i]

    # Wrap the coordinates
    for f in range(n_frames):
        # Calculate the geometric box center as the average of the selection atoms. Numerically stable average
        if n_centersel > 0:
            for i in range(3):
                box_center[i] = 0
            for n in range(n_centersel):
                for i in range(3):
                    box_center[i] = box_center[i] + (coords[centersel[n], i, f] - box_center[i]) / (n + 1)

        for i in range(3):
            half_box[i] = box[i, f] / 2

        for g in range(n_groups-1):
            start_idx = groups[g]
            end_idx = groups[g+1]

            # Calculate the geometric center of the group. Numerically stable average
            for i in range(3):
                grp_center[i] = 0
            n = 0
            for k in range(start_idx, end_idx):
                for i in range(3):
                    grp_center[i] = grp_center[i] + (coords[k, i, f] - grp_center[i]) / (n + 1)
                n = n + 1
            
            for i in range(3):
                diff = grp_center[i] - box_center[i]
                if fabs(diff) > half_box[i]:
                    translation = box[i, f] * round(diff / box[i, f])
                    for a in range(start_idx, end_idx):
                        coords[a, i, f] = coords[a, i, f] - translation

