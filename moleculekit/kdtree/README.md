# KDTree

The implementation was taken from scipy commit
f82b3f9d3c30cc6e7f906b0d00b8b43bfea02e5c
(`scipy/spatial/_ckdtree.pyx`, `scipy/spatial/_kdtree.py`,
`scipy/spatial/ckdtree/src/*`) so that `moleculekit` does not need to
depend on SciPy.

The following modifications were made to `_ckdtree.pyx`:

- Dropped `import scipy.sparse` and the `coo_array`/`coo_matrix`/
  `dok_array`/`dok_matrix` output types of
  `cKDTree.sparse_distance_matrix`.  Only `'dict'` and `'ndarray'` are
  supported -- build a sparse matrix externally if you need one.
- Inlined `scipy._lib._util.copy_if_needed` as `None` (the NumPy >=2
  convention for "copy only when necessary").
- Replaced `libcpp.mutex.py_safe_call_once` / `py_safe_once_flag`
  (requires Cython >= 3.2.5) with a plain `threading.Lock` guarding
  lazy initialisation of the `cKDTree.tree` property.

The files in this folder follow the scipy license.


Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.