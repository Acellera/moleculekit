# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool
from libc.math cimport round, sqrt, acos, floor, fabs
from cython.parallel import prange

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
UINT32 = np.uint32
FLOAT32 = np.float32
FLOAT64 = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint32_t UINT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

import cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_bonded_groups(
        UINT32_t[:,:] bonds,
        UINT32_t[:] has_lower_bond,
        int n_atoms,
        UINT32_t[:] group,
    ):
    cdef int i, j, k
    cdef int n_bonds = bonds.shape[0]
    cdef vector[int] results

    # Mark atoms which have bonds to lower indexes
    for i in range(n_bonds):
        has_lower_bond[max(bonds[i, 0], bonds[i, 1])] = 1

    # Atoms which don't have bonds to lower indexes are the start of a new fragment
    k = 0
    for i in range(n_atoms):
        if not has_lower_bond[i]:
            k += 1
            results.push_back(i)
        group[i] = k
            
    results.push_back(n_atoms)

    return results


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _mark_residues(
        bool[:] protein,
        bool[:] nucleic,
        bool[:] segcrossers,
        int p_bb_count,
        int n_bb_count,
        int res_start,
        int i
    ):
    if p_bb_count >= 4:
        # If we have 4 or more protein backbone atoms in the residue it's a protein
        for j in range(res_start, i):
            # if segcrossers[j] == 1:
            #     # If the atom has a bond to another segment don't mark as protein
            #     continue
            protein[j] = 1
    elif n_bb_count >= 4:
        # If we have 4 or more nucleic backbone atoms in the residue it's a nucleic
        for j in range(res_start, i):
            # if segcrossers[j] == 1:
            #     # If the atom has a bond to another segment don't mark as nucleic
            #     continue
            nucleic[j] = 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def find_protein_nucleic_ext(
        bool[:] protein,
        bool[:] nucleic,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        bool[:] segcrossers,
        UINT32_t[:] resseq,
    ):
    cdef int i, j, k, p_bb_count, n_bb_count, curr_res, res_start
    cdef int n_atoms = protein.shape[0]

    curr_res = -1
    p_bb_count = 0
    n_bb_count = 0
    for i in range(n_atoms):
        if curr_res != resseq[i]:
            _mark_residues(protein, nucleic, segcrossers, p_bb_count, n_bb_count, res_start, i)
            # New residue. Reset vars
            res_start = i
            p_bb_count = 0
            n_bb_count = 0

        curr_res = resseq[i]
        if protein_bb[i] == 1:
            p_bb_count += 1
        elif nucleic_bb[i] == 1:
            n_bb_count += 1

    _mark_residues(protein, nucleic, segcrossers, p_bb_count, n_bb_count, res_start, i+1)
            

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef vector[vector[int]] _make_uniq_resids(
        int n_atoms,
        vector[vector[int]] atom_bonds,
        UINT32_t[:] resid,
        UINT32_t[:] insertion,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        UINT32_t[:] uq_resid,
        UINT32_t[:] flgs
    ):
    cdef stack[int] st
    cdef vector[vector[int]] residue_atoms
    cdef int i, j, bi, res, ins, num_residues

    residue_atoms.push_back(vector[int]())
    
    num_residues = 0
    for i in range(n_atoms):
        if flgs[i] != 1:  # Not numbered yet
            # Find atoms connected to i with same resid and give new uniq_resid
            res = resid[i]
            ins = insertion[i]

            st.push(i)

            while not st.empty():
                j = st.top()
                st.pop()
                if flgs[j]:
                    continue
                    
                uq_resid[j] = num_residues
                residue_atoms[num_residues].push_back(j)
                flgs[j] = 1

                for bi in atom_bonds[j]:
                    if flgs[bi] != 1:
                        if chain_id[i] == chain_id[bi] and seg_id[i] == chain_id[bi] and resid[bi] == res and insertion[bi] == ins:
                            st.push(bi)
            num_residues += 1
            residue_atoms.push_back(vector[int]())

    return residue_atoms

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef vector[vector[int]] _get_atom_bonds(
        int n_atoms,
        UINT32_t[:, :] bonds
    ):
    cdef vector[vector[int]] atom_bonds
    cdef int n_bonds, b1, b2

    # Create nested list of bonded atoms
    for i in range(n_atoms):
        atom_bonds.push_back(vector[int]())

    n_bonds = bonds.shape[0]
    for i in range(n_bonds):
        b1 = bonds[i, 0]
        b2 = bonds[i, 1]
        atom_bonds[b1].push_back(b2)
        atom_bonds[b2].push_back(b1)

    return atom_bonds

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def analyze_molecule(
        int n_atoms,
        UINT32_t[:, :] bonds,
        UINT32_t[:] resid,
        UINT32_t[:] insertion,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        bool[:] protein,
        bool[:] nucleic,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
    ):
    cdef vector[vector[int]] atom_bonds
    cdef vector[vector[int]] residue_atoms
    cdef UINT32_t[:] uq_resid = np.zeros(n_atoms, dtype=UINT32)
    cdef UINT32_t[:] flgs = np.zeros(n_atoms, dtype=UINT32)
    cdef int n_residues

    atom_bonds = _get_atom_bonds(n_atoms, bonds)

    # Create unique residue IDs
    residue_atoms = _make_uniq_resids(n_atoms, atom_bonds, resid, insertion, chain_id, seg_id, uq_resid, flgs)

    # Create list of atoms in each unique residue

    return atom_bonds, uq_resid, residue_atoms

            