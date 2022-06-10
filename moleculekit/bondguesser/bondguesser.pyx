# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport round, sqrt, acos, floor, fabs
from cython.parallel import prange

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


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def make_grid_neighborlist_nonperiodic(
        np.ndarray[UINT32_t, ndim=2] gridlist, 
        UINT32_t xboxes,
        UINT32_t yboxes,
        UINT32_t zboxes,
    ):
    cdef int xi, yi, zi, box_idx, xy_boxes, n

    xy_boxes = xboxes * yboxes
    box_idx = 0
    for zi in range(0, zboxes):
        for yi in range(0, yboxes):
            for xi in range(0, xboxes):
                n = 0
                gridlist[box_idx, n] = box_idx
                n += 1
                if xi < xboxes - 1:
                    gridlist[box_idx, n] = box_idx + 1
                    n += 1
                if yi < yboxes - 1:
                    gridlist[box_idx, n] = box_idx + xboxes
                    n += 1
                if zi < zboxes - 1:
                    gridlist[box_idx, n] = box_idx + xy_boxes
                    n += 1
                if xi < (xboxes - 1) and yi < (yboxes - 1):
                    gridlist[box_idx, n] = box_idx + xboxes + 1
                    n += 1
                if xi < (xboxes - 1) and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes + 1
                    n += 1
                if yi < (yboxes - 1) and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes + xboxes
                    n += 1
                if xi < (xboxes - 1) and yi > 0:
                    gridlist[box_idx, n] = box_idx - xboxes + 1
                    n += 1
                if xi > 0 and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes - 1
                    n += 1
                if yi > 0 and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes - xboxes
                    n += 1
                if xi < (xboxes - 1) and yi < (yboxes - 1) and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes + xboxes + 1
                    n += 1
                if xi > 0 and yi < (yboxes - 1) and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes + xboxes - 1
                    n += 1
                if xi < (xboxes - 1) and yi > 0 and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes - xboxes + 1
                    n += 1
                if xi > 0 and yi > 0 and zi < (zboxes - 1):
                    gridlist[box_idx, n] = box_idx + xy_boxes - xboxes - 1
                    n += 1
                box_idx += 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def grid_bonds(
        np.ndarray[FLOAT32_t, ndim=2] coords, 
        np.ndarray[FLOAT32_t, ndim=1] radii,
        np.ndarray[UINT32_t, ndim=1] is_hydrogen,
        UINT32_t num_in_box, # First num_in_box atoms belong to the center box
        float box_cutoff,
    ):
    cdef int i, j
    cdef float dist2, dx, dy, dz, cutoff2, cut  # squared dist
    cdef int n_atoms = coords.shape[0]
    cdef vector[int] results

    cutoff2 = box_cutoff * box_cutoff

    for i in range(num_in_box): # Stop iterating at num_in_box to not calculate the neighbours between them
        for j in range(i+1, n_atoms):
            if is_hydrogen[i] and is_hydrogen[j]:
                # Skip bonds between two hydrogens
                continue

            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            dist2 = dx * dx + dy * dy + dz * dz

            # Ignore also atoms with near identical coords
            if dist2 > cutoff2 or dist2 < 0.001:
                continue

            cut = 0.6 * (radii[i] + radii[j])
            if dist2 > (cut * cut):
                continue

            results.push_back(i)
            results.push_back(j)

    return results
            

