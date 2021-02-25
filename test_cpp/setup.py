from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="hbonds",
    ext_modules=cythonize("hbonds.pyx", language="c++"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)