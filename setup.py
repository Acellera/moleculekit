import setuptools
import unittest
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import subprocess


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir="moleculekit", pattern="*.py")
    return test_suite


try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except Exception:
    print("Could not get version tag. Defaulting to version 0")
    version = "0"


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

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
compiler_args = ["-Wno-unused-function", "-Wno-unreachable-code", "-Wno-sign-compare"]
extentions.append(
    Extension(
        "moleculekit.xtc",
        sources=[
            "moleculekit/xtc_utils/src/xdrfile_xtc.c",
            "moleculekit/xtc_utils/src/xdrfile.c",
            "moleculekit/xtc_utils/src/xtc.c",
            "moleculekit/xtc_utils/xtc.pyx",
        ],
        include_dirs=[
            "moleculekit/xtc_utils/include/",
            "moleculekit/xtc_utils/",
            numpy.get_include(),
        ],
        extra_compile_args=compiler_args,
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


if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="moleculekit",
        version=version,
        author="Acellera",
        author_email="info@acellera.com",
        description="A molecule reading/writing and manipulation package.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/acellera/moleculekit/",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: POSIX :: Linux",
        ],
        packages=setuptools.find_packages(
            include=["moleculekit*"],
            exclude=[
                "test-data",
                "*test*",
            ],
        ),
        package_data={
            "moleculekit": [
                "vmd_wrapper",
                "logging.ini",
                "share/*/*/*",
                "share/*/*",
                "share/*",
            ],
        },
        zip_safe=False,
        test_suite="setup.my_test_suite",
        install_requires=requirements,
        ext_modules=cythonize(extentions, language_level="3"),
    )
