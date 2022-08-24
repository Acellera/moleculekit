# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport sqrt, round, exp
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
def calculate_occupancy(
        FLOAT64_t[:,:] centers,
        FLOAT32_t[:,:] coords,
        FLOAT64_t[:,:] sigmas,
        FLOAT64_t[:,:] results,
    ):
    cdef int a, c, h
    cdef int n_atoms = coords.shape[0]
    cdef int n_centers = centers.shape[0]
    cdef int n_channels = sigmas.shape[1]
    cdef FLOAT64_t dist2, dx, dy, dz, atomsigmas, x, x3, x12, value

    for a in range(n_atoms):
        for c in range(n_centers):
            dx = coords[a, 0] - centers[c, 0]
            dy = coords[a, 1] - centers[c, 1]
            dz = coords[a, 2] - centers[c, 2]
            dist2 = dx * dx + dy * dy + dz * dz

            if dist2 < 25:  # At 5A the values are already very small
                for h in range(n_channels):
                    if sigmas[a, h] == 0:
                        continue
                    x = sigmas[a, h] / sqrt(dist2)
                    x3 = x * x * x
                    x12 = x3 * x3 * x3 * x3
                    value = 1.0 - exp(-x12)
                    results[c, h] = max(results[c, h], value)

                    



