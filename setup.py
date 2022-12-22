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
            "moleculekit/xtc_utils/src/xdrfile_xtc.cpp",
            "moleculekit/xtc_utils/src/xdrfile.cpp",
            "moleculekit/xtc_utils/src/xtc.cpp",
            "moleculekit/xtc_utils/xtc.pyx",
        ],
        include_dirs=[
            "moleculekit/xtc_utils/include/",
            "moleculekit/xtc_utils/",
            numpy.get_include(),
        ],
        language="c++",
    )
)
compiler_args = ["-w"]
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
        extra_compile_args=compiler_args,
        language="c++",
    )
)

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    ext_modules=cythonize(extentions, language_level="3"),
)
