#ifndef _xdr_seek_h
#define _xdr_seek_h

#include <stdint.h>
#include "xdrfile.h"

int64_t xdr_tell(XDRFILE *xd);
int xdr_seek(XDRFILE *xd, int64_t pos, int whence);
int xdr_flush(XDRFILE *xd);

#endif
