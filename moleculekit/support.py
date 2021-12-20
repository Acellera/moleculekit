# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from tempfile import NamedTemporaryFile
import moleculekit.home
import numpy
import os
import ctypes as ct


def xtc_lib():
    lib = {}
    libdir = moleculekit.home.home(libDir=True)

    import platform

    if platform.system() == "Windows":
        lib["libc"] = ct.cdll.msvcrt
        ct.cdll.LoadLibrary(os.path.join(libdir, "libgcc_s_seh-1.dll"))
        if os.path.exists(os.path.join(libdir, "psprolib.dll")):
            ct.cdll.LoadLibrary(os.path.join(libdir, "psprolib.dll"))
        lib["libxtc"] = ct.cdll.LoadLibrary(os.path.join(libdir, "libxtc.dll"))
    else:
        # lib['libc'] = cdll.LoadLibrary("libc.so.6")
        lib["libc"] = ct.cdll.LoadLibrary(
            f"libc.{'so.6' if platform.uname()[0] != 'Darwin' else 'dylib'}"
        )
        lib["libxtc"] = ct.cdll.LoadLibrary(os.path.join(libdir, "libxtc.so"))
    return lib


def string_to_tempfile(content, ext):
    f = NamedTemporaryFile(delete=False, suffix="." + ext)
    f.write(content.encode("ascii", "ignore"))
    f.close()
    return f.name


def pack_string_buffer(data):
    ptr = (ct.c_char_p * len(data))()
    ptr[:] = [x.encode() for x in data]
    return ptr


def pack_int_buffer(data):
    ptr = (ct.c_int * len(data))()
    ptr[:] = data
    return ptr


def pack_ulong_buffer(data):
    ptr = (ct.c_ulong * len(data))()
    ptr[:] = data
    return ptr


def pack_double_buffer(data):
    ptr = (ct.c_double * len(data))()
    ptr[:] = data
    return ptr
