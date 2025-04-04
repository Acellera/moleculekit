
import os
import warnings
import numpy as np
cimport numpy as np

cimport xtclib

from libc.stdio cimport SEEK_SET, SEEK_CUR
from libc.math cimport ceil

ctypedef np.npy_int64   int64_t
ctypedef np.npy_float32 float32_t
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t
ctypedef np.uint32_t UINT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

import cython


cdef int get_xtc_nframes(char* filename):
    """ You need to pass the string with filename.encode("UTF-8") to this function """
    return xtclib.xtc_nframes(filename)

cdef int get_xtc_natoms(char* filename):
    """ You need to pass the string with filename.encode("UTF-8") to this function """
    return xtclib.xtc_natoms(filename)
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def read_xtc(char* filename):
    """ You need to pass the string with filename.encode("UTF-8") to this function """
    cdef int natoms = get_xtc_natoms(filename)
    cdef int nframes = get_xtc_nframes(filename)

    cdef FLOAT32_t[:, :, ::1] coords = np.zeros((natoms, 3, nframes), dtype=np.float32)
    cdef FLOAT32_t[:, :, ::1] box = np.zeros((3, 3, nframes), dtype=np.float32)
    cdef FLOAT32_t[::1] time = np.zeros(nframes, dtype=np.float32)
    cdef int[::1] step = np.zeros(nframes, dtype=np.int32)

    xtclib.xtc_read_new(
        filename, 
        &coords[0, 0, 0], 
        &box[0, 0, 0], 
        &time[0], 
        &step[0], 
        natoms, 
        nframes,
    )
    return np.asarray(coords), np.asarray(box), np.asarray(time), np.asarray(step)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def read_xtc_frames(char* filename, int[:] frames):
    """ You need to pass the string with filename.encode("UTF-8") to this function """
    cdef int traj_natoms = get_xtc_natoms(filename)
    cdef int traj_nframes = get_xtc_nframes(filename)
    cdef int nframes = frames.shape[0]
    cdef int f = 0

    cdef FLOAT32_t[:, :, ::1] coords = np.zeros((traj_natoms, 3, nframes), dtype=np.float32)
    cdef FLOAT32_t[:, :, ::1] box = np.zeros((3, 3, nframes), dtype=np.float32)
    cdef FLOAT32_t[::1] time = np.zeros(nframes, dtype=np.float32)
    cdef int[::1] step = np.zeros(nframes, dtype=np.int32)

    for f in range(nframes):
        xtclib.xtc_read_frame(
            filename, 
            &coords[0, 0, 0], 
            &box[0, 0, 0], 
            &time[0], 
            &step[0], 
            traj_natoms, 
            frames[f],
            nframes,
            f
        )
    return np.asarray(coords), np.asarray(box), np.asarray(time), np.asarray(step)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def write_xtc(char* filename, float[:, :, :] coords, float[:, :, :] box, float[:] time, unsigned int[:] step):
    cdef int natoms = coords.shape[0]
    cdef int nframes = coords.shape[2]
    xtclib.xtc_write(
        filename, 
        natoms, 
        nframes, 
        &step[0], 
        &time[0],
        &coords[0, 0, 0],
        &box[0, 0, 0]
    )