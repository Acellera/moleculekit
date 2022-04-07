# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
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

import cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate(
        np.ndarray[UINT32_t, ndim=2] donors, 
        np.ndarray[UINT32_t, ndim=1] acceptors, 
        np.ndarray[FLOAT32_t, ndim=3] coords, 
        np.ndarray[FLOAT32_t, ndim=2] box,
        np.ndarray[UINT32_t, ndim=1] sel1, 
        np.ndarray[UINT32_t, ndim=1] sel2, 
        float dist_threshold=2.5,
        float angle_threshold=120,
        bool intra=False,
        bool ignore_hs=False,
    ):
    cdef int d, a, f, i
    cdef int n_donors = donors.shape[0]
    cdef int n_acceptors = acceptors.shape[0]
    cdef int n_frames = coords.shape[2]
    cdef vector[vector[int]] results
    cdef FLOAT32_t[:] dist_vec_a = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] dist_vec_b = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t[:] half_box = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t dist2_a, dist2_b, dotprod, val, angle
    cdef UINT32_t d_idx_d, d_idx_h, d_idx, a_idx

    dist_threshold = dist_threshold * dist_threshold
    angle_threshold = angle_threshold / 57.29578

    for f in range(n_frames):
        results.push_back(vector[int]())
        for i in range(3):
            half_box[i] = box[i, f] / 2

        for d in range(n_donors):
            for a in range(n_acceptors):
                a_idx = acceptors[a]
                d_idx_d = donors[d, 0]
                d_idx_h = donors[d, 1]

                # Don't calculate h-bonds with identical donor-acceptor atom
                if a_idx == d_idx_d:
                    continue

                if intra:
                    # Both donor and acceptors must be selected
                    if sel1[a_idx] == 0 or sel1[d_idx_d] == 0:
                        continue
                else:
                    # Donor and acceptor must belong to different selections
                    if not ((sel1[a_idx] == 1 and sel2[d_idx_d] == 1) or (sel2[a_idx] == 1 and sel1[d_idx_d] == 1)):
                        continue

                dist2_a = 0
                dist2_b = 0
                d_idx = d_idx_h
                if ignore_hs: # If we ignore the hydrogen just calculate heavy to heavy distance
                    d_idx = d_idx_d
                # Calculate donor hydrogen (or heavy if ignore_hs) to acceptor vector
                for i in range(3):
                    val = coords[a_idx, i, f] - coords[d_idx, i, f] 
                    # Wrap the distance vector into the periodic box
                    if abs(val) > half_box[i] and box[i, f] != 0:
                        val = val - box[i, f] * round(val / box[i, f])
                    dist_vec_a[i] = val
                    dist2_a = dist2_a + (val * val)

                # Donor hydrogen too far from acceptor
                if dist2_a > dist_threshold:
                    continue

                # If we ignore the hydrogens skip the angle part. We have found a match already
                if ignore_hs:
                    results[f].push_back(d_idx_d)
                    results[f].push_back(-1)  # No hydrogen to add here
                    results[f].push_back(a_idx)
                    continue

                # Calculate donor heavy to donor hydrogen vector for the angle
                for i in range(3):
                    val = coords[d_idx_d, i, f] - coords[d_idx_h, i, f] 
                    # Wrap the distance vector into the periodic box
                    if abs(val) > half_box[i] and box[i, f] != 0:
                        val = val - box[i, f] * round(val / box[i, f])
                    dist_vec_b[i] = val
                    dist2_b = dist2_b + (val * val)

                # Overlapping atoms? Weird case but best exit
                if dist2_a == 0 or dist2_b == 0:
                    continue

                # Calculate the angle
                # Calculate vector dot product
                dotprod = 0
                for i in range(3):
                    dotprod = dotprod + dist_vec_a[i] * dist_vec_b[i]

                angle = dotprod / (sqrt(dist2_a) * sqrt(dist2_b))
                # Prevent numerical issues with acos
                if angle > 1:
                    angle = 1
                if angle < -1:
                    angle = -1
                angle = acos(angle)

                if angle > angle_threshold:
                    results[f].push_back(d_idx_d)
                    results[f].push_back(d_idx_h)
                    results[f].push_back(a_idx)

    return results


