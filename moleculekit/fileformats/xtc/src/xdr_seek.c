/* 64 bit fileseek operations */
#define _FILE_OFFSETS_BITS 64
#include "xdr_seek.h"
#include <stdio.h>

int64_t xdr_tell(XDRFILE *xd)
{
	FILE *fptr = xd->fp;

#ifndef _WIN32
	// use posix 64 bit ftell version
	return ftello(fptr);
#elif defined(_MSVC_VER) && !__INTEL_COMPILER
	return _ftelli64(fptr);
#else
	return ftell(fptr);
#endif
}

int xdr_seek(XDRFILE *xd, int64_t pos, int whence)
{
	int result = 1;
	FILE *fptr = xd->fp;

#ifndef _WIN32
	// use posix 64 bit ftell version
	result = fseeko(fptr, pos, whence) < 0 ? exdrNR : exdrOK;
#elif _MSVC_VER && !__INTEL_COMPILER
	result = _fseeki64(fptr, pos, whence) < 0 ? exdrNR : exdrOK;
#else
	result = fseek(fptr, pos, whence) < 0 ? exdrNR : exdrOK;
#endif
	if (result != exdrOK)
		return result;

	return exdrOK;
}

int xdr_flush(XDRFILE *xdr)
{
	return fflush(xdr->fp);
}
