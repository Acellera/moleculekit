# MoleculeKit

[![Conda](https://anaconda.org/acellera/moleculekit/badges/version.svg)](https://anaconda.org/acellera/moleculekit)

A molecule manipulation library

# Getting started

### Install it into the base conda environment

#### With pip

```
pip install moleculekit
```

#### With conda

[Installation Instructions](https://software.acellera.com/moleculekit/installation.html)

### Using moleculekit in ipython

Install ipython in the correct conda enviroment using the following command.

```
pip install ipython
```

Now you can start an ipython console with

```
ipython
```

In the ipython console you can now import any of the modules of moleculekit and use it as normal.

```python
from moleculekit.molecule import Molecule

mol = Molecule('3ptb')
mol.view()
```

### API

For the official documentation of the moleculekit API head over to https://software.acellera.com/moleculekit/index.html

### Issues

For any bugs or questions on usage feel free to use the issue tracker of this github repo.

### Dev

If you are using moleculekit without installing it by using the PYTHONPATH env var you will need to compile the C++ extensions in-place with the following command:

```
python setup.py build_ext --inplace
```

#### Building for WebAssembly

Install `emscripten` https://emscripten.org/docs/getting_started/downloads.html

```
conda create -n pyodide-build
conda activate pyodide-build
conda install python=3.11
pip install pyodide-build==0.25.1

# Activate the emscripten environment
cd ../emsdk
./emsdk install 3.1.46
./emsdk activate 3.1.46
source emsdk_env.sh
cd -

# Build the package
export PYO3_CROSS_INCLUDE_DIR="HACK"
export PYO3_CROSS_LIB_DIR="HACK"
rm -rf .pyodide-xbuildenv
pyodide build -o dist_pyodide
cp dist_pyodide/*.whl test_wasm/wheels/
cd test_wasm
python3 -m http.server
```

If you get an error at building about numpy missing, check this issue https://github.com/pyodide/pyodide/issues/4347

#### Debugging segmentation faults in Cython part

1. Put a reproducible script in a file like `segfault.py`
2. Modify setup.py to have `-g` as compile flag instead of `-O3`
3. Recompile extensions with `python setup.py build_ext --inplace`
4. Execute script with gdb like `gdb --args python segfault.py`
5. Execute `run` and then `bt 10` to show the backtrace
6. Have fun

## Citing MoleculeKit

If you use this software in your publication please cite:

Stefan Doerr, Matthew J. Harvey, Frank Noé, and Gianni De Fabritiis.
**HTMD: High-throughput molecular dynamics for molecular discovery.**
_Journal of Chemical Theory and Computation_, **2016**, _12_ (4), pp 1845–1852.
[doi:10.1021/acs.jctc.6b00049](http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00049)
