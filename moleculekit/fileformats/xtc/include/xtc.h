/* (c) 2015-2016 Acellera Ltd www.acellera.com
 * All rights reserved
 * Distributed under the terms of the HTMD Software License Agreement
 * No redistribution in whole or in part
 */

#ifdef PLATFORM_Linux
#if defined(__i386__) || defined(__x86_64__)
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif
#endif

#ifndef XTC
#define XTC 1
typedef struct float4
{
  float x, y, z, w;
} float4;

typedef struct double4
{
  double x, y, z, w;
} double4;

#ifdef __cplusplus
extern "C"
{
#endif

  // struct XTC_frame *xtc_read_frame(char *filename, int *natoms, int frame);
  void xtc_read_frame(char *filename, float *coords_arr, float *box_arr, float *time_arr, int *step_arr, int natoms, int frame, int nframes, int fidx);

  int xtc_write(char *filename, int natoms, int nframes, unsigned int *step, float *timex, float *pos, float *box);
  // int xtc_write( char *filename, int natoms, int step, float time, float *pos, float*  box ) ;

  int xtc_truncate_to_step(char *infile, unsigned long maxstep);

  struct XTC_frame
  {
    float box[3][3];
    int natoms;
    unsigned long step;
    double time;
    float *pos;
  }; // XTC_frame;

  struct XTC_frame *xtc_read(char *filename, int *natoms, int *Nframes, double *dt, int *dstep);

  void xtc_read_new(char *filename, float *coords_arr, float *box_arr, float *time_arr, int *step_arr, int natoms, int nframes);

  int xtc_nframes(char *filename);
  int xtc_natoms(char *filename);
#ifdef __cplusplus
}
#endif

#endif
