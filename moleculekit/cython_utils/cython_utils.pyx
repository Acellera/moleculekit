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
def calculateAnglesAndDihedrals(
        UINT32_t[:,:] bonds,
        bool cyclicdih,
        int n_atoms,
    ):
    # Same as dist_trajectory but instead of returning distances it returns index
    # pairs of atoms that are within a certain distance threshold
    cdef int i, j, k, a, b, b1, b2, min_v, max_v, n_neigh, n_angles, a1, a2
    cdef int n_bonds = bonds.shape[0]
    cdef vector[vector[UINT32_t]] neighbors
    cdef vector[vector[UINT32_t]] angles
    cdef vector[vector[UINT32_t]] dihedrals
    cdef vector[UINT32_t] buffer
    cdef vector[UINT32_t] x, y

    for i in range(n_atoms):
        for j in range(n_bonds):
            b1 = bonds[j, 0]
            b2 = bonds[j, 1]
            if b1 == i:
                buffer.push_back(b2)
            elif b2 == i:
                buffer.push_back(b1)
        neighbors.push_back(buffer)
        buffer.clear()

    for i in range(n_atoms):
        n_neigh = neighbors[i].size()
        for j in range(n_neigh):
            for k in range(j+1, n_neigh):
                a = neighbors[i][j]
                b = neighbors[i][k]
                if a != b:
                    if a < b:
                        min_v = a
                        max_v = b
                    else:
                        min_v = b
                        max_v = a
                    buffer.push_back(min_v)
                    buffer.push_back(i)
                    buffer.push_back(max_v)
                    angles.push_back(buffer)
                    buffer.clear()

    n_angles = angles.size()
    for a1 in range(n_angles):
        for a2 in range(a1 + 1, n_angles):
            x = angles[a1]
            y = angles[a2]
            if x[1] == y[0] and x[2] == y[1] and (cyclicdih or (x[0] != y[2])):
                buffer.push_back(x[0])
                buffer.push_back(x[1])
                buffer.push_back(x[2])
                buffer.push_back(y[2])
                dihedrals.push_back(buffer)
                buffer.clear()
            if x[1] == y[2] and x[2] == y[1] and (cyclicdih or (x[0] != y[0])):
                buffer.push_back(x[0])
                buffer.push_back(x[1])
                buffer.push_back(x[2])
                buffer.push_back(y[0])
                dihedrals.push_back(buffer)
                buffer.clear()
            if y[1] == x[0] and y[2] == x[1] and (cyclicdih or (y[0] != x[2])):
                buffer.push_back(y[0])
                buffer.push_back(y[1])
                buffer.push_back(y[2])
                buffer.push_back(x[2])
                dihedrals.push_back(buffer)
                buffer.clear()
            if y[1] == x[0] and y[0] == x[1] and (cyclicdih or (y[2] != x[2])):
                buffer.push_back(y[2])
                buffer.push_back(y[1])
                buffer.push_back(y[0])
                buffer.push_back(x[2])
                dihedrals.push_back(buffer)
                buffer.clear()
            
    return neighbors, angles, dihedrals