import setuptools
import unittest
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir="moleculekit", pattern="*.py")
    return test_suite


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# extra_compile_args = ["-O3",]
interactions_ext = Extension(
    name="moleculekit.interactions.hbonds",
    sources=["moleculekit/interactions/hbonds/hbonds.pyx"],
    include_dirs=[numpy.get_include()],
    language="c++",
    # extra_compile_args=extra_compile_args
    # libraries=["examples"],
    # library_dirs=["lib"],
)

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="moleculekit",
        version="MOLECULEKIT_VERSION_PLACEHOLDER",
        author="Acellera",
        author_email="info@acellera.com",
        description="A molecule reading/writing and manipulation package.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/acellera/moleculekit/",
        classifiers=[
            "Programming Language :: Python :: PYTHON_VERSION_PLACEHOLDER",
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
            "moleculekit": ["lib/*/*", "vmd_wrapper", "logging.ini"],
        },
        zip_safe=False,
        test_suite="setup.my_test_suite",
        install_requires=requirements,
        ext_modules=cythonize([interactions_ext], language_level="3"),
    )
