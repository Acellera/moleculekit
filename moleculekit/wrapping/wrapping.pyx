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
def calculate(
        np.ndarray[UINT32_t, ndim=1] groups, 
        np.ndarray[FLOAT32_t, ndim=3] coords, 
        np.ndarray[FLOAT32_t, ndim=2] box,
        np.ndarray[UINT32_t, ndim=1] centersel, 
    ):
    cdef int f, i, g, a, start_idx, end_idx, k, n
    cdef int n_atoms = coords.shape[0]
    cdef int n_frames = coords.shape[2]
    cdef int n_groups = groups.shape[0]
    cdef int n_centersel = centersel.shape[0]
    cdef FLOAT32_t[:] half_box = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] box_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] grp_center = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t translation, diff

    # Wrap the coordinates
    for f in range(n_frames):
        # Calculate the geometric box center as the average of the selection atoms. Numerically stable average
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

