# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libc.math cimport round, sqrt, acos

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
UINT32 = np.uint32
FLOAT32 = np.float32

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint32_t UINT32_t
ctypedef np.float32_t FLOAT32_t

# cdef get_plane():
    

import cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate(
        vector[vector[int]] rings,
        np.ndarray[FLOAT32_t, ndim=3] coords, 
        np.ndarray[FLOAT32_t, ndim=2] box,
    ):
    cdef int f, i, j
    cdef int n_frames = coords.shape[2]
    cdef int n_rings = rings.size()
    cdef vector[vector[int]] results
    cdef vector[vector[int]].iterator it1 = rings.begin()
    cdef vector[vector[int]].iterator it2

    for f in range(n_frames):
        results.push_back(vector[int]())
        for i in range(n_rings):
            print(rings[i])

    return results


