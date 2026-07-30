from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Build every extension against the CPython Stable ABI (PEP 384) so that one
# wheel per platform serves all supported interpreters instead of one wheel per
# (platform, Python version). 0x030B0000 targets 3.11, the earliest version that
# exposes Py_buffer through the limited API, which our memoryviews require.
#
# The bdist_wheel option below is what actually tags the wheel "cp311-abi3".
# Setting py_limited_api on the Extension alone produces an abi3 .so inside a
# version-specific wheel, which pip then refuses to install on other versions.
#
# Emscripten/Pyodide wheels are tied to one Pyodide runtime and are built per
# version regardless, so the Stable ABI gains nothing there. pyodide_build.yml
# sets this variable to keep that path producing the wheels it always has.
STABLE_ABI = not os.environ.get("MOLECULEKIT_DISABLE_STABLE_ABI")

LIMITED_API = (
    {
        "define_macros": [("Py_LIMITED_API", "0x030B0000")],
        "py_limited_api": True,
    }
    if STABLE_ABI
    else {}
)

extentions = [
    "moleculekit/interactions/hbonds/hbonds.pyx",
    "moleculekit/interactions/pipi/pipi.pyx",
    "moleculekit/interactions/cationpi/cationpi.pyx",
    "moleculekit/interactions/sigmahole/sigmahole.pyx",
    "moleculekit/wrapping/wrapping.pyx",
    "moleculekit/bondguesser_utils/bondguesser_utils.pyx",
    "moleculekit/atomselect_utils/atomselect_utils.pyx",
    "moleculekit/distance_utils/distance_utils.pyx",
    "moleculekit/occupancy_utils/occupancy_utils.pyx",
    "moleculekit/cython_utils/cython_utils.pyx",
]
extentions = [
    Extension(
        name=os.path.dirname(ext).replace("/", "."),
        sources=[ext],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
        # extra_link_args=["-fopenmp"],
        **LIMITED_API,
    )
    for ext in extentions
]
extentions.append(
    Extension(
        "moleculekit.xtc",
        sources=[
            "moleculekit/fileformats/xtc/src/xdrfile_xtc.cpp",
            "moleculekit/fileformats/xtc/src/xdrfile.cpp",
            "moleculekit/fileformats/xtc/src/xtc_src.cpp",
            "moleculekit/fileformats/xtc/xtc.pyx",
        ],
        include_dirs=[
            "moleculekit/fileformats/xtc/include/",
            "moleculekit/fileformats/xtc/",
            numpy.get_include(),
        ],
        language="c++",
        **LIMITED_API,
    )
)
extentions.append(
    Extension(
        "moleculekit.trr",
        sources=[
            "moleculekit/fileformats/xtc/src/xdrfile_trr.c",
            "moleculekit/fileformats/xtc/src/xdrfile.cpp",
            "moleculekit/fileformats/xtc/src/xdr_seek.c",
            "moleculekit/fileformats/xtc/trr.pyx",
        ],
        include_dirs=[
            "moleculekit/fileformats/xtc/include/",
            "moleculekit/fileformats/xtc/",
            numpy.get_include(),
        ],
        language="c",
        **LIMITED_API,
    )
)
extentions.append(
    Extension(
        "moleculekit.dcd",
        sources=[
            "moleculekit/fileformats/dcd/src/dcdplugin.c",
            "moleculekit/fileformats/dcd/dcd.pyx",
        ],
        include_dirs=[
            "moleculekit/fileformats/dcd/include/",
            "moleculekit/fileformats/dcd/",
            numpy.get_include(),
        ],
        language="c",
        **LIMITED_API,
    )
)
extentions.append(
    Extension(
        "moleculekit.binpos",
        sources=[
            "moleculekit/fileformats/binpos/src/binposplugin.c",
            "moleculekit/fileformats/binpos/binpos.pyx",
        ],
        include_dirs=[
            "moleculekit/fileformats/binpos/include/",
            "moleculekit/fileformats/binpos/",
            numpy.get_include(),
        ],
        language="c",
        **LIMITED_API,
    )
)
extentions.append(
    Extension(
        "moleculekit.tmalign",
        sources=[
            "moleculekit/tmalign/src/TMAlign.cpp",
            "moleculekit/tmalign/tmalign_util.pyx",
        ],
        include_dirs=[
            "moleculekit/tmalign/include/",
            "moleculekit/tmalign/",
            numpy.get_include(),
        ],
        extra_compile_args=["-w"],
        language="c++",
        **LIMITED_API,
    )
)
# Port of scipy.spatial.cKDTree (scipy BSD-3-Clause license).  See
# moleculekit/kdtree/README.md for licensing details.  We strip the
# scipy.sparse dependency, so only the Cython/C++ sources are needed.
extentions.append(
    Extension(
        "moleculekit.kdtree._ckdtree",
        sources=[
            "moleculekit/kdtree/src/build.cxx",
            "moleculekit/kdtree/src/count_neighbors.cxx",
            "moleculekit/kdtree/src/query.cxx",
            "moleculekit/kdtree/src/query_ball_point.cxx",
            "moleculekit/kdtree/src/query_ball_tree.cxx",
            "moleculekit/kdtree/src/query_pairs.cxx",
            "moleculekit/kdtree/src/sparse_distances.cxx",
            "moleculekit/kdtree/_ckdtree.pyx",
        ],
        include_dirs=[
            "moleculekit/kdtree/src",
            numpy.get_include(),
        ],
        extra_compile_args=["-O3", "-w"],
        language="c++",
        **LIMITED_API,
    )
)

setup(
    zip_safe=False,
    ext_modules=cythonize(extentions, language_level="3"),
    options={"bdist_wheel": {"py_limited_api": "cp311"}} if STABLE_ABI else {},
)
