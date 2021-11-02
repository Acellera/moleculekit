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
cdef void _cross_product(
        np.ndarray[FLOAT32_t, ndim=1] vec_a, 
        np.ndarray[FLOAT32_t, ndim=1] vec_b,
        np.ndarray[FLOAT32_t, ndim=1] res,
    ):
    res[0] = vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1]
    res[1] = vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2]
    res[2] = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]

    cdef float vec_norm = 0 
    for i in range(3):
        vec_norm = vec_norm + (res[i] * res[i])
    vec_norm = sqrt(vec_norm)

    for i in range(3):
        res[i] = res[i] / vec_norm


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate(
        np.ndarray[UINT32_t, ndim=1] rings_atoms, 
        np.ndarray[UINT32_t, ndim=1] rings1_start_indexes, 
        np.ndarray[UINT32_t, ndim=1] rings2_start_indexes, 
        np.ndarray[FLOAT32_t, ndim=3] coords, 
        np.ndarray[FLOAT32_t, ndim=2] box,
        float dist_threshold1=4.4,
        float angle_threshold1_max=30,
        float dist_threshold2=5.5,
        float angle_threshold2_min=60,
    ):
    cdef int r1, r2, f, i
    cdef int n_rings1 = rings1_start_indexes.shape[0] - 1  # Remove one for the index which we append to the start
    cdef int n_rings2 = rings2_start_indexes.shape[0] - 1
    cdef int n_frames = coords.shape[2]

    cdef vector[vector[int]] results
    cdef vector[vector[float]] distangles
    cdef np.ndarray[FLOAT32_t, ndim=1] ring1_mean = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] ring2_mean = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] ring1_normal = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] ring2_normal = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] half_box = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] tmp1 = np.zeros(3, dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_t, ndim=1] tmp2 = np.zeros(3, dtype=FLOAT32)
    cdef FLOAT32_t dist2, dot_prod
    cdef int r1_start_idx, r2_start_idx, r1_end_idx, r2_end_idx

    dist_threshold1 = dist_threshold1 * dist_threshold1
    dist_threshold2 = dist_threshold2 * dist_threshold2

    for f in range(n_frames):
        results.push_back(vector[int]())
        distangles.push_back(vector[float]())
        for i in range(3):
            half_box[i] = box[i, f] / 2

        for r1 in range(n_rings1):
            for r2 in range(n_rings2):
                r1_start_idx = rings1_start_indexes[r1]
                r1_end_idx = rings1_start_indexes[r1+1]
                r2_start_idx = rings2_start_indexes[r2]
                r2_end_idx = rings2_start_indexes[r2+1]

                if r1_start_idx == r2_start_idx and r1_end_idx == r2_end_idx:
                    # Skip self-interactions of identical rings
                    continue
                
                # Exit early if any of the two ring's atoms are too far
                for i in range(3):
                    tmp1[i] = coords[rings_atoms[r1_start_idx], i, f]
                    tmp2[i] = coords[rings_atoms[r2_start_idx], i, f]
                dist2 = _wrapped_dist(f, tmp1, tmp2, half_box, box)
                # Rings too far apart
                if dist2 > 225: # If two randomly picked atoms are more than 15A apart early-quit
                    continue

                # Calculate the ring centroids
                _calculate_ring_mean(f, r1_start_idx, r1_end_idx, rings_atoms, coords, ring1_mean)
                _calculate_ring_mean(f, r2_start_idx, r2_end_idx, rings_atoms, coords, ring2_mean)

                # Calculate the wrapped distance between the ring centroids
                dist2 = _wrapped_dist(f, ring1_mean, ring2_mean, half_box, box)

                # Ring centroids too far apart
                if dist2 > dist_threshold2:
                    # print(r1, r2, sqrt(dist2))
                    continue

                # Calculate the plane normals
                for i in range(3):
                    tmp1[i] = coords[rings_atoms[r1_start_idx], i, f] - coords[rings_atoms[r1_start_idx+2], i, f]
                    tmp2[i] = coords[rings_atoms[r1_start_idx+1], i, f] - coords[rings_atoms[r1_start_idx+2], i, f]
                _cross_product(tmp1, tmp2, ring1_normal)
                for i in range(3):
                    tmp1[i] = coords[rings_atoms[r2_start_idx], i, f] - coords[rings_atoms[r2_start_idx+2], i, f]
                    tmp2[i] = coords[rings_atoms[r2_start_idx+1], i, f] - coords[rings_atoms[r2_start_idx+2], i, f]
                _cross_product(tmp1, tmp2, ring2_normal)

                # Calculate angle between normals
                dot_prod = 0
                for i in range(3):
                    dot_prod = dot_prod + ring1_normal[i] * ring2_normal[i]
                angle = acos(dot_prod) * 57.29578  # Convert radians to degrees
                if angle > 90:  # minimal angle to line. necessary as perpendicular can point either way
                    angle = 180-angle

                # If dist < 4.4 and angle <= 30 deg
                # If dist < 5.5 and angle in range [60, 120]
                # print(r1, r2, sqrt(dist2), angle)
                if ((dist2 < dist_threshold1 and angle <= angle_threshold1_max) or 
                    (dist2 < dist_threshold2 and angle >= angle_threshold2_min)):
                    results[f].push_back(r1)
                    results[f].push_back(r2)
                    distangles[f].push_back(sqrt(dist2))
                    distangles[f].push_back(angle)

    return results, distangles


