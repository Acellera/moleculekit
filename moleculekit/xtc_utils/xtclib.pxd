cimport numpy as np
ctypedef np.npy_int64 int64_t
ctypedef np.npy_float32 float32_t

cdef extern from "include/xtc.h":
    int xtc_nframes(char *filename)
    int xtc_natoms(char *filename)
    void xtc_read_frame(char *filename, float *coords_arr, float *box_arr, float *time_arr, int *step_arr, int natoms, int frame, int nframes, int fidx)
    void xtc_read_new(char *filename, float *coords_arr, float *box_arr, float *time_arr, int *step_arr, int natoms, int nframes)
    int xtc_write(char *filename, int natoms, int nframes, int *step, float *timex, float *pos, float *box)