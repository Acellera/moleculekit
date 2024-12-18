from setuptools import setup, Extension
from Cython.Build import cythonize
import versioneer
import numpy
import os

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
    )
)

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    ext_modules=cythonize(extentions, language_level="3"),
)
