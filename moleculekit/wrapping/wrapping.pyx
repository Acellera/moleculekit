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
    cdef int n_atoms = coords.shape[0]
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
        for a in range(n_atoms):
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


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def wrap_compact_unitcell(
        np.ndarray[UINT32_t, ndim=1] groups,
        np.ndarray[FLOAT32_t, ndim=3] coords,
        np.ndarray[FLOAT64_t, ndim=3] boxvectors,
        np.ndarray[UINT32_t, ndim=1] centersel,
        np.ndarray[FLOAT32_t, ndim=1] center,
    ):
    # Based on GROMACS pbc.cpp put_atoms_in_triclinic_unitcell function
    cdef int f, i, g, a, start_idx, end_idx, k, n
    cdef int n_frames = coords.shape[2]
    cdef int n_groups = groups.shape[0]
    cdef int n_atoms = coords.shape[0]
    cdef int n_centersel = centersel.shape[0]
    cdef FLOAT32_t[:] wrap_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] box_middle = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] grp_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT64_t[:, :] box = np.zeros((3, 3), dtype=FLOAT64)
    cdef FLOAT64_t[:, :] tric_vec = np.zeros((12, 3), dtype=FLOAT64)
    cdef FLOAT64_t[3] hbox_diag
    cdef FLOAT64_t[3] dx

    if n_centersel == 0:
        for i in range(3):
            wrap_center[i] = center[i]

    # Wrap the coordinates
    for f in range(n_frames):
        for i in range(3):
            for j in range(3):
                box[i, j] = boxvectors[i, j, f]

        for i in range(3):
            hbox_diag[i] = box[i, i] * 0.5

        # Calculate the geometric box center as the average of the selection atoms. Numerically stable average
        if n_centersel > 0:
            for i in range(3):
                wrap_center[i] = 0
            for n in range(n_centersel):
                for i in range(3):
                    wrap_center[i] = wrap_center[i] + (coords[centersel[n], i, f] - wrap_center[i]) / (n + 1)

        for i in range(3):
            box_middle[i] = 0
        for i in range(3):
            for j in range(3):
                box_middle[j] += 0.5 * box[i, j]

        # Center the coordinates on the wrap center
        for a in range(n_atoms):
            for i in range(3):
                coords[a, i, f] = coords[a, i, f] - wrap_center[i] + box_middle[i]
        
        # Get the PBC vectors
        for i in range(12):
            for j in range(3):
                tric_vec[i, j] = 0
        max_cutoff2, ntric_vec = get_pbc(box, tric_vec)

        for g in range(n_groups-1):
            start_idx = groups[g]
            end_idx = groups[g+1]

            # Calculate the geometric center of the group. Numerically stable average
            for i in range(3):
                grp_center[i] = 0
            n = 0
            for a in range(start_idx, end_idx):
                for i in range(3):
                    grp_center[i] = grp_center[i] + (coords[a, i, f] - grp_center[i]) / (n + 1)
                n = n + 1

            pbc_dx(grp_center, box_middle, box, hbox_diag, tric_vec, max_cutoff2, ntric_vec, 1, dx)

            for a in range(start_idx, end_idx):
                for i in range(3):
                    coords[a, i, f] = (coords[a, i, f] - grp_center[i]) + box_middle[i] + dx[i] 

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef FLOAT64_t norm2(FLOAT64_t[:] vec) nogil:
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]

cdef FLOAT64_t cmin(FLOAT64_t a, FLOAT64_t b) nogil:
    return a if a < b else b

cdef FLOAT64_t cmax(FLOAT64_t a, FLOAT64_t b) nogil:
    return a if a > b else b


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (FLOAT64_t, int) get_pbc(FLOAT64_t[:, :] boxvectors, FLOAT64_t[:, :] tric_vec) noexcept:
    # Based on GROMACS pbc.cpp low_set_pbc function
    cdef FLOAT64_t min_hv2, min_ss, max_cutoff2, d2old, d2new, d2new_c
    cdef int ntric_vec, kk, k, jj, j, ii, i, dd, bUse, e, shift
    cdef int[3] order = [0, -1, 1] # Define possible shift orders for triclinic boxes: no shift (0), negative (-1), positive (1)
    cdef FLOAT64_t[3] fbox_diag
    cdef FLOAT64_t[3] hbox_diag
    cdef FLOAT64_t[3] trial
    cdef FLOAT64_t[3] pos
    # cdef FLOAT64_t[:, :] tric_shift = np.zeros((12, 3), dtype=FLOAT64)
    cdef FLOAT64_t skewnessMargin = 1.001 # Margin for correction when the box is too skewed

    max_cutoff2 = 0 # Maximum allowed cutoff squared (will be set later)
    ntric_vec = 0   # Number of triclinic correction vectors

    # Calculate full box diagonal and half box diagonal values for each dimension
    for i in range(3):
        fbox_diag[i] = boxvectors[i, i]       # Full box length in dimension i
        hbox_diag[i] = fbox_diag[i] * 0.5     # Half box length

    # Calculate maximum allowed cutoff based on box size
    # Physical limitation of the cut-off by half the length of the shortest box vector.
    min_hv2 = 0.25 * cmin(norm2(boxvectors[0]), norm2(boxvectors[1]))
    min_hv2 = cmin(min_hv2, 0.25 * norm2(boxvectors[2]))
    # Limitation to the smallest diagonal element due to optimizations:
    # checking only linear combinations of single box-vectors (2 in x)
    # in the grid search and pbc_dx is a lot faster
    # than checking all possible combinations.
    min_ss = cmin(boxvectors[0, 0], cmin(boxvectors[1, 1] - abs(boxvectors[2, 1]), boxvectors[2, 2]))
    max_cutoff2 = cmin(min_hv2, min_ss * min_ss)

    # For triclinic boxes, we need to calculate the shift vectors
    # Loop over all possible shift combinations
    for kk in range(3):
        k = order[kk] # Z-direction shift
        for jj in range(3):
            j = order[jj] # Y-direction shift
            for ii in range(3):
                i = order[ii] # X-direction shift

                # Only consider shifts that involve Y or Z components
                if not (j != 0 or k != 0):
                    continue

                d2old = 0
                d2new = 0

                # Calculate the trial shift vector
                for d in range(3):
                    trial[d] = i * boxvectors[0, d] + j * boxvectors[1, d] + k * boxvectors[2, d]

                    # Select position that maximizes the effect of the shift
                    if trial[d] < 0:
                        pos[d] = cmin(hbox_diag[d], -trial[d])
                    else:
                        pos[d] = cmax(-hbox_diag[d], -trial[d])
                    d2old += pos[d] ** 2
                    d2new += (pos[d] + trial[d]) ** 2

                # Check if this shift reduces the distance
                if skewnessMargin * d2new < d2old:
                    bUse = 1
                    # Verify no simpler shift gives similar reduction
                    for dd in range(3):
                        if dd == 0:
                            shift = i
                        elif dd == 1:
                            shift = j
                        else:
                            shift = k
                        if shift:
                            d2new_c = 0
                            for e in range(3):
                                d2new_c += (pos[e] + trial[e] - shift * boxvectors[dd, e]) ** 2
                            if d2new_c <= skewnessMargin * d2new:
                                bUse = 0
                                break
                    if bUse: # If this is a useful shift vector, store it
                        if ntric_vec >= 12:
                            # Too many triclinic vectors!! Should raise error
                            raise ValueError("Too many triclinic vectors!!")

                        # Store the shift vector and its components
                        for e in range(3):
                            tric_vec[ntric_vec, e] = trial[e]
                        # tric_shift[ntric_vec, 0] = i
                        # tric_shift[ntric_vec, 1] = j
                        # tric_shift[ntric_vec, 2] = k
                        ntric_vec += 1

    return max_cutoff2, ntric_vec
        

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef pbc_dx(
    FLOAT32_t[:] x1, 
    FLOAT32_t[:] x2, 
    FLOAT64_t[:, :] boxvectors, 
    FLOAT64_t[:] hbox_diag, 
    FLOAT64_t[:, :] tric_vec,
    FLOAT64_t max_cutoff2,
    int ntric_vec,
    int mode,
    FLOAT64_t[:] dx,
):
    cdef int i, j, k
    cdef FLOAT64_t[3] dx_start
    cdef FLOAT64_t[3] trial
    cdef FLOAT64_t d2min, d2trial

    for i in range(3):
        dx[i] = x1[i] - x2[i]

    if mode == 0: # Rectangular box
        for i in range(3):
            while dx[i] > hbox_diag[i]:
                dx[i] -= boxvectors[i, i]
            while dx[i] <= -hbox_diag[i]:
                dx[i] += boxvectors[i, i]
    elif mode == 1: # Triclinic box
        for i in range(2, -1, -1):
            while dx[i] > hbox_diag[i]:
                for j in range(i, -1, -1):
                    dx[j] -= boxvectors[i, j]
            while dx[i] <= -hbox_diag[i]:
                for j in range(i, -1, -1):
                    dx[j] += boxvectors[i, j]
            d2min = norm2(dx)
            if d2min > max_cutoff2:
                for j in range(3):
                    dx_start[j] = dx[j]
                k = 0
                while (d2min > max_cutoff2) and (k < ntric_vec):
                    for j in range(3):
                        trial[j] = dx_start[j] + tric_vec[k, j]
                    d2trial = norm2(trial)
                    if d2trial < d2min:
                        for j in range(3):
                            dx[j] = trial[j]
                        d2min = d2trial
                    k += 1

                        


# TODO: Make a single wrap entry point where I also validate the box dimensions
#       and then call the appropriate wrap function based on the box type
