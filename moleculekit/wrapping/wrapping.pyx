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
def wrap_box(
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



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def wrap_triclinic_unitcell(
        np.ndarray[UINT32_t, ndim=1] groups,
        np.ndarray[FLOAT32_t, ndim=3] coords,
        np.ndarray[FLOAT64_t, ndim=3] boxvectors,
        np.ndarray[UINT32_t, ndim=1] centersel,
        np.ndarray[FLOAT32_t, ndim=1] center,
    ):
    # Based on GROMACS pbc.cpp put_atoms_in_triclinic_unitcell function
    cdef int f, i, g, a, start_idx, end_idx, k, n, m
    cdef int n_frames = coords.shape[2]
    cdef int n_groups = groups.shape[0]
    cdef int n_centersel = centersel.shape[0]
    cdef FLOAT32_t[:] wrap_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] box_middle = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT64_t[:] shift_center = np.zeros(3, dtype=FLOAT64)
    cdef FLOAT32_t[:] grp_center_init = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] grp_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT64_t[:, :] box = np.zeros((3, 3), dtype=FLOAT64)
    cdef FLOAT64_t shm01, shm02, shm12, shift

    if n_centersel == 0:
        for i in range(3):
            wrap_center[i] = center[i]

    # Wrap the coordinates
    for f in range(n_frames):
        for i in range(3):
            for j in range(3):
                box[i, j] = boxvectors[i, j, f]

        # Calculate the geometric box center as the average of the selection atoms. Numerically stable average
        if n_centersel > 0:
            for i in range(3):
                wrap_center[i] = 0
            for n in range(n_centersel):
                for i in range(3):
                    wrap_center[i] = wrap_center[i] + (coords[centersel[n], i, f] - wrap_center[i]) / (n + 1)

        for i in range(3):
            box_middle[i] = 0 #box_center[i]
        for i in range(3):
            for j in range(3):
                box_middle[j] += 0.5 * box[i, j]

        # Calculate the elements of the shift matrix shm that transforms the coordinates
        # These factors account for the non-rectangular nature of the cell:
        # shm01 - Coupling between y and x directions
        # shm02 - Coupling between z and x directions
        # shm12 - Coupling between z and y directions
        shm01 = box[1, 0] / box[1, 1]
        shm02 = (box[1, 1] * box[2, 0] - box[2, 1] * box[1, 0]) / (box[1, 1] * box[2, 2])
        shm12 = box[2, 1] / box[2, 2]

        # Calculate the shift center - this is used to center the periodic cell
        for i in range(3):
            shift_center[i] = 0
        for i in range(3):
            for j in range(3):
                shift_center[j] = shift_center[j] + box[i, j]
        # Scale the shift center by 0.5
        for i in range(3):
            shift_center[i] = shift_center[i] * 0.5
        # Subtract the shift center from the box center
        for i in range(3):
            shift_center[i] = box_middle[i] - shift_center[i]

        # Transform the shift center using the shift matrix
        shift_center[0] = shm01 * shift_center[1] + shm02 * shift_center[2]
        shift_center[1] = shm12 * shift_center[2]
        shift_center[2] = 0

        # Center the coordinates on the wrap center
        for a in range(coords.shape[0]):
            for i in range(3):
                coords[a, i, f] = coords[a, i, f] - wrap_center[i] + box_middle[i]

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
                    grp_center_init[i] = grp_center[i]
                n = n + 1

            # Process dimensions in reverse order (z,y,x) due to triclinic cell geometry
            for m in range(2, -1, -1):
                shift = shift_center[m]
                # Apply additional shifts for x and y components due to triclinic coupling
                if m == 0: # x affected by y and z
                    shift += shm01 * grp_center[1] + shm02 * grp_center[2]
                elif m == 1: # y affected by z
                    shift += shm12 * grp_center[2]

                # If group center is outside the box in the negative direction,
                # shift it by adding box vectors until it's inside
                while grp_center[m] - shift < 0:
                    for d in range(m+1):
                        grp_center[d] += box[m, d]

                # If group center is outside the box in the positive direction,
                # shift it by subtracting box vectors until it's inside
                while grp_center[m] - shift >= box[m, m]:
                    for d in range(m+1):
                        grp_center[d] -= box[m, d]

                for a in range(start_idx, end_idx):
                    coords[a, m, f] = coords[a, m, f] - (grp_center_init[m] - grp_center[m])
