# cython: cdivision=True

import numpy as np
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

import cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _calculate_ring_mean(
        int f,
        int start_idx, 
        int end_idx, 
        np.ndarray[UINT32_t, ndim=1] rings_atoms, 
        np.ndarray[FLOAT32_t, ndim=3] coords,
        np.ndarray[FLOAT32_t, ndim=1] ring_mean,
    ):
    cdef FLOAT32_t coor
    for i in range(3):
        ring_mean[i] = 0

    for rr in range(start_idx, end_idx):
        for i in range(3):
            coor = coords[rings_atoms[rr], i, f]
            ring_mean[i] = ring_mean[i] + coor

    for i in range(3):
        ring_mean[i] = ring_mean[i] / (end_idx - start_idx)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float _wrapped_dist(
        int f,
        np.ndarray[FLOAT32_t, ndim=1] point1,
        np.ndarray[FLOAT32_t, ndim=1] point2,
        np.ndarray[FLOAT32_t, ndim=1] half_box,
        np.ndarray[FLOAT32_t, ndim=2] box,
    ):
    cdef FLOAT32_t val
    cdef float dist2 = 0
    for i in range(3):
        val = point1[i] - point2[i] 
        # Wrap the distance vector into the periodic box
        if abs(val) > half_box[i] and box[i, f] != 0:
            val = val - box[i, f] * round(val / box[i, f])
        dist2 = dist2 + (val * val)
    return dist2


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _normalize(np.ndarray[FLOAT32_t, ndim=1] vec):
    cdef float vec_norm = 0 
    for i in range(3):
        vec_norm = vec_norm + (vec[i] * vec[i])
    vec_norm = sqrt(vec_norm)

    for i in range(3):
        vec[i] = vec[i] / vec_norm


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _cross_product(
        np.ndarray[FLOAT32_t, ndim=1] vec_a, 
        np.ndarray[FLOAT32_t, ndim=1] vec_b,
        np.ndarray[FLOAT32_t, ndim=1] res,
    ):
    res[0] = vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1]
    res[1] = vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2]
    res[2] = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]

    _normalize(res)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate(
        np.ndarray[UINT32_t, ndim=1] rings_atoms, 
        np.ndarray[UINT32_t, ndim=1] rings_start_indexes, 
        np.ndarray[UINT32_t, ndim=2] halogen_bond,
        np.ndarray[FLOAT32_t, ndim=3] coords, 
        np.ndarray[FLOAT32_t, ndim=2] box,
        float dist_threshold=4.5,
        float angle_threshold_min=60,
    ):
    cdef int r1, r2, f, i
    cdef int n_rings = rings_start_indexes.shape[0] - 1  # Subtract 1 for the added 0 index at the start
    cdef int n_halogens = halogen_bond.shape[0]
    cdef int n_frames = coords.shape[2]

    cdef vector[vector[int]] results
    cdef vector[vector[float]] distangles
    cdef np.ndarray[FLOAT32_t, ndim=1] ring_mean = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] ring_normal = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] half_box = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] halogen_coor = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] tmp1 = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] tmp2 = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t dist2, dot_prod
    cdef int r_start_idx, r_end_idx, halogen_idx, partner_idx

    dist_threshold = dist_threshold * dist_threshold

    for f in range(n_frames):
        results.push_back(vector[int]())
        distangles.push_back(vector[float]())
        for i in range(3):
            half_box[i] = box[i, f] / 2

        for r1 in range(n_rings):
            for r2 in range(n_halogens):
                r_start_idx = rings_start_indexes[r1]
                r_end_idx = rings_start_indexes[r1+1]
                halogen_idx = halogen_bond[r2, 0]
                partner_idx = halogen_bond[r2, 1]

                # Calculate the ring centroid
                _calculate_ring_mean(f, r_start_idx, r_end_idx, rings_atoms, coords, ring_mean)

                # Halogen coordinates
                for i in range(3):
                    halogen_coor[i] = coords[halogen_idx, i, f]

                # Calculate the wrapped distance between the ring centroid and the cation
                dist2 = _wrapped_dist(f, ring_mean, halogen_coor, half_box, box)

                # Ring centroid too far apart from cation
                if dist2 > dist_threshold:
                    continue

                # Calculate the plane normal
                for i in range(3):
                    tmp1[i] = coords[rings_atoms[r_start_idx], i, f] - coords[rings_atoms[r_start_idx+2], i, f]
                    tmp2[i] = coords[rings_atoms[r_start_idx+1], i, f] - coords[rings_atoms[r_start_idx+2], i, f]
                _cross_product(tmp1, tmp2, ring_normal)

                # Calculate halide-aryl bond vector
                for i in range(3):
                    tmp1[i] = halogen_coor[i] - coords[partner_idx, i, f]
                _normalize(tmp1)

                # Calculate angle between plane normal and ring-cation vector
                dot_prod = 0
                for i in range(3):
                    dot_prod = dot_prod + ring_normal[i] * tmp1[i]
                angle = acos(dot_prod) * 57.29578  # Convert radians to degrees
                if angle > 90:  # minimal angle to line. necessary as perpendicular can point either way
                    angle = 180 - angle
                angle = 90 - angle # We want the angle to the plane, not the normal of the plane so 90 - angle

                if angle >= angle_threshold_min:
                    results[f].push_back(r1)
                    results[f].push_back(halogen_bond[r2, 0])
                    distangles[f].push_back(sqrt(dist2))
                    distangles[f].push_back(angle)

    return results, distangles


