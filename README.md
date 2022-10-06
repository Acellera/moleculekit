# MoleculeKit

[![Build Status](https://dev.azure.com/stefdoerr/moleculekit/_apis/build/status/Acellera.moleculekit?branchName=master)](https://dev.azure.com/stefdoerr/moleculekit/_build/latest?definitionId=1&branchName=master)
[![Language Grade: Python](https://img.shields.io/lgtm/grade/python/g/Acellera/moleculekit.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Acellera/moleculekit/context:python)
[![Conda](https://anaconda.org/acellera/moleculekit/badges/version.svg)](https://anaconda.org/acellera/moleculekit)
[![codecov](https://codecov.io/gh/Acellera/moleculekit/branch/master/graph/badge.svg)](https://codecov.io/gh/Acellera/moleculekit)

A molecule manipulation library

# Getting started

We recommend installing Miniconda on your machine to better manage python packages and environments.

You can install moleculekit either in the "base" conda environment or in a new conda environment. We recommend the second.

### Install it into the base conda environment

#### With conda

[Installation Instructions](https://software.acellera.com/moleculekit/installation.html)

#### With pip

The pip version of moleculekit is VERY limited and not officially supported. Use at your own risk.

```
(base) user@computer:~$ pip install moleculekit
```

### Optional dependencies of moleculekit

Moleculekit has a small number of optional dependencies which are needed for some of it's functionalities. They were not added to the default dependencies to keep moleculekit a fast and small installation and to avoid unnecessary conflicts with other software. However if you want to leverage all of it's functionality you can install the rest of the dependencies with the following command:

```
(moleculekit) user@computer:~$ wget https://raw.githubusercontent.com/Acellera/moleculekit/master/extra_requirements.txt
(moleculekit) user@computer:~$ conda install --file extra_requirements.txt -c acellera
```

### Using moleculekit in ipython

Install ipython in the correct conda enviroment using the following command. If you have installed the extra dependencies as above, you can skip this step since it already installs ipython.

```
(moleculekit) user@computer:~$ conda install ipython
```

Now you can start an ipython console with

```
(moleculekit) user@computer:~$ ipython
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
conda install python=3.10
pip install pyodide-build==0.22.0.dev0

# Activate the emscripten environment
cd ../../emsdk
source emsdk_env.sh
cd -

# Build the package
pyodide build # Requires some hacking around the pip installation
cp dist/moleculekit-*wasm32.whl test_wasm/moleculekit-X-cp310-cp310-emscripten_3_1_14_wasm32.whl
cd test_wasm
python3 -m http.server
```

## Citing MoleculeKit

If you use this software in your publication please cite:

Stefan Doerr, Matthew J. Harvey, Frank Noé, and Gianni De Fabritiis.
**HTMD: High-throughput molecular dynamics for molecular discovery.**
_Journal of Chemical Theory and Computation_, **2016**, _12_ (4), pp 1845–1852.
[doi:10.1021/acs.jctc.6b00049](http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00049)
