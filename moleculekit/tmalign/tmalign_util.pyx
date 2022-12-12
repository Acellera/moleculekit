import os
import warnings
import numpy as np
cimport numpy as np

cimport tmalignlib

from libc.stdio cimport SEEK_SET, SEEK_CUR
from libc.math cimport ceil
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free

ctypedef np.npy_int64   int64_t
ctypedef np.npy_float32 float32_t
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t
ctypedef np.uint32_t UINT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

import cython
    


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def tmalign(double[:,:,:] coords, double[:,:] ref_coords, string seqx, string seqy):
    # declare variable specific to this pair of TMalign
    # cdef double t0[3]
    # cdef double u0[3][3]
    cdef double TM1 = 0
    cdef double TM2 = 0
    cdef double TM3 = 0
    cdef double TM4 = 0
    cdef double TM5 = 0   # for a_opt, u_opt, d_opt
    cdef double d0_0 = 0
    cdef double TM_0 = 0
    cdef double d0A = 0
    cdef double d0B = 0
    cdef double d0u = 0
    cdef double d0a = 0
    cdef double d0_out = 5.0
    cdef string seqM, seqxA, seqyA # for output alignment
    cdef double rmsd0 = 0.0
    cdef int L_ali = 0 # Aligned length in standard_TMscore
    cdef double Liden = 0
    cdef double TM_ali = 0
    cdef double rmsd_ali = 0 # TMscore and rmsd in standard_TMscore
    cdef int n_ali = 0
    cdef int n_ali8 = 0
    cdef bool cp_opt = False
    cdef int i_opt = 0;        # 1 for -i, 3 for -I
    cdef int a_opt = 0;        # flag for -a, do not normalized by average length
    cdef bool u_opt = False    # flag for -u, normalized by user specified length
    cdef bool d_opt = False    # flag for -d, user specified d0
    cdef bool fast_opt = False # flags for -fast, fTM-align algorithm
    cdef double Lnorm_ass = 0
    cdef double d0_scale = 0
    cdef double TMcut = -1
    cdef vector[string] sequence
    cdef int mol_type = 0
    cdef vector[double *] coords_vec
    cdef vector[double *] ref_coords_vec

    cdef double t0[3]
    cdef double u0[3][3]

    cdef int xlen = seqx.size()
    cdef int ylen = seqy.size()

    cdef char *secx = <char *> malloc((xlen + 1) * sizeof(char))
    cdef char *secy = <char *> malloc((ylen + 1) * sizeof(char))

    cdef vector[double] TM1_all, rmsd_all
    cdef vector[int] nali_all
    cdef vector[vector[double]] translations 
    cdef vector[vector[double]] rotations 
    cdef vector[double] t0vec  # Temp variable to convert from double* to vector
    cdef vector[double] u0vec  # Temp variable to convert from double** to flat vector

    for i in range(ref_coords.shape[0]):
        ref_coords_vec.push_back(&ref_coords[i, 0])
    # Calculate secondary structure for ref structure
    tmalignlib.make_sec(ref_coords_vec.data(), ylen, secy)

    cdef int natoms = coords.shape[1]
    cdef int nframes = coords.shape[0]

    for f in range(nframes):
        for i in range(natoms):
            coords_vec.push_back(&coords[f, i, 0])            

        # Calculate secondary structure for current frame
        tmalignlib.make_sec(coords_vec.data(), xlen, secx)

        # entry function for structure alignment
        tmalignlib.TMalign_main(
            coords_vec.data(), ref_coords_vec.data(), seqx.c_str(), seqy.c_str(), secx, secy,
            &t0[0], &u0[0], TM1, TM2, TM3, TM4, TM5,
            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
            seqM, seqxA, seqyA,
            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
            xlen, ylen, sequence, Lnorm_ass, d0_scale,
            i_opt, a_opt, u_opt, d_opt, fast_opt,
            mol_type, TMcut)

        TM1_all.push_back(TM1)
        rmsd_all.push_back(rmsd0)
        nali_all.push_back(n_ali8)

        # Converting from arrays to vectors to be able to pass the data back to python
        t0vec.assign(t0, t0+3)
        translations.push_back(t0vec)
        t0vec.clear()
        for i in range(3):
            for j in range(3):
                u0vec.push_back(u0[i][j])
        rotations.push_back(u0vec)
        u0vec.clear()

        # Done! Free memory
        seqM.clear()
        seqxA.clear()
        seqyA.clear()
        coords_vec.clear()

    free(secx)
    free(secy)
    ref_coords_vec.clear()

    return TM1_all, rmsd_all, nali_all, translations, rotations
