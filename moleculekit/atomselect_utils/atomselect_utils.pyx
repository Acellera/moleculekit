# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool
from libc.math cimport round, sqrt, acos, floor, fabs
from cython.parallel import prange
from cpython cimport array
import array

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
UINT32 = np.uint32
INT64 = np.int64
FLOAT32 = np.float32
FLOAT64 = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint32_t UINT32_t
ctypedef np.int64_t INT64_t
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
cdef vector[vector[int]] _make_uniq_resids(
        int n_atoms,
        vector[vector[int]] atom_bonds,
        INT64_t[:] resid,
        UINT32_t[:] insertion,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        UINT32_t[:] uq_resid,
        bool[:] flgs
    ):
    cdef stack[int] st
    cdef vector[vector[int]] residue_atoms
    cdef int i, j, bi, res, num_residues
    cdef UINT32_t ins

    residue_atoms.push_back(vector[int]())
    
    num_residues = 0
    for i in range(n_atoms):
        if not flgs[i]:  # Not numbered yet
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
                flgs[j] = True

                for bi in atom_bonds[j]:
                    if flgs[bi]:
                        continue
                    if chain_id[j] == chain_id[bi] and seg_id[j] == seg_id[bi] and resid[bi] == res and insertion[bi] == ins:
                        st.push(bi)
            num_residues += 1
            residue_atoms.push_back(vector[int]())

    residue_atoms.pop_back()
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
cdef bool _has_n_backbone(
        bool[:] bb,
        INT64_t[:] resid,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        int i,
        UINT32_t tmpid,
        UINT32_t[:] flgs,
        vector[vector[int]] atom_bonds,
        int n,
        int residueid,
    ):
    cdef stack[int] st
    cdef int j, count
    count = 0

    # find n backbone atoms connected together with the given residueid
    if flgs[i] != 0:
        return False # Already seen
    
    if not bb[i] or resid[i] != residueid:
        return False # Not a backbone atom

    st.push(i)

    while not st.empty():
        i = st.top()
        st.pop()

        flgs[i] = tmpid
        count += 1
        if count >= n:
            return True

        for j in atom_bonds[i]:
            if flgs[j] == 0:
                if chain_id[i] != chain_id[j] or seg_id[i] != seg_id[j]:
                    continue
                if bb[j] and residueid == resid[j]:
                    st.push(j)
    return False


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _clean_up_connection(
        int i,
        UINT32_t tmpid,
        vector[vector[int]] atom_bonds,
        UINT32_t[:] flgs,
    ):
    cdef stack[int] st
    cdef int j

    if flgs[i] != tmpid: # Been here before
        return False

    st.push(i)
    while not st.empty():
        i = st.top()
        st.pop()
        flgs[i] = 0
        for j in atom_bonds[i]:
            if flgs[j] == tmpid:
                st.push(j)

    return True


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_connected_atoms_in_resid(
        int i,
        UINT32_t tmpid,
        vector[vector[int]] atom_bonds,
        UINT32_t[:] flgs,
        INT64_t[:] resid,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        bool[:] polymer, 
        int residueid,
    ):
    cdef stack[int] st
    cdef int j

    if flgs[i] != 0 or resid[i] != residueid:
        return False

    st.push(i)
    while not st.empty():
        i = st.top()
        st.pop()
        flgs[i] = tmpid
        polymer[i] = True

        for j in atom_bonds[i]:
            if flgs[j] == 0 and chain_id[i] == chain_id[j] and seg_id[i] == seg_id[j] and residueid == resid[j]:
                st.push(j)
    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_and_mark(
        int n, 
        bool[:] bb,
        bool[:] polymer, 
        UINT32_t[:] flgs,
        vector[vector[int]] atom_bonds,
        INT64_t[:] resid,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
    ):
    cdef stack[int] st
    cdef int n_atoms, i
    cdef UINT32_t tmpid
    n_atoms = bb.shape[0]
    tmpid = 1

    for i in range(n_atoms):
        if bb[i] and flgs[i] == 0:
            if _has_n_backbone(bb, resid, chain_id, seg_id, i, tmpid, flgs, atom_bonds, n, resid[i]):
                _clean_up_connection(i, tmpid, atom_bonds, flgs)
                _find_connected_atoms_in_resid(i, tmpid, atom_bonds, flgs, resid, chain_id, seg_id, polymer, resid[i])
                tmpid += 1
            else:
                _clean_up_connection(i, tmpid, atom_bonds, flgs)

    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_residues(
        int n_atoms,
        bool[:] protein,
        bool[:] nucleic,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        INT64_t[:] resid,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        vector[vector[int]] atom_bonds,
        UINT32_t[:] flgs
    ):
    # TODO: Replace with pure C: cdef UINT32_t[:] flgs = array_new[UINT32](n_atoms)
    _find_and_mark(4, protein_bb, protein, flgs, atom_bonds, resid, chain_id, seg_id)
    _find_and_mark(4, nucleic_bb, nucleic, flgs, atom_bonds, resid, chain_id, seg_id)
    return True


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_connected_fragments(
        int n_residues,
        int n_atoms,
        vector[vector[int]] residue_atoms,
        vector[vector[int]] atom_bonds,
        UINT32_t[:] uq_resid,
        UINT32_t[:] fragments,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        bool[:] sg_atoms,
        bool[:] res_flgs,
    ):
    cdef stack[int] st
    cdef int i, j, ba, ra, ri, count
    count = 0

    for i in range(n_residues):
        if not res_flgs[i]:
            st.push(i)

            while not st.empty():
                j = st.top()
                st.pop()

                for ra in residue_atoms[j]:
                    fragments[ra] = count

                    for ba in atom_bonds[ra]:
                        ri = uq_resid[ba]

                        if ri != i and res_flgs[ri] == 0 and chain_id[ra] == chain_id[ba] and seg_id[ra] == seg_id[ba] and not (sg_atoms[ra] and sg_atoms[ba]):
                            res_flgs[ri] = True
                            st.push(ri)
            count += 1
    return True


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_fragments(
        int n_residues,
        int n_atoms,
        vector[vector[int]] residue_atoms,
        vector[vector[int]] atom_bonds,
        UINT32_t[:] uq_resid,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        bool[:] sg_atoms,
        bool[:] res_flgs,
        UINT32_t[:] fragments,
        # UINT32_t[:] protein_frag,
        # UINT32_t[:] nucleic_frag,
        # UINT32_t[:] protein_frag_cyclic,
        # UINT32_t[:] nucleic_frag_cyclic,
    ):
    _find_connected_fragments(n_residues, n_atoms, residue_atoms, atom_bonds, uq_resid, fragments, chain_id, seg_id, sg_atoms, res_flgs)
    # _find_subfragments()
    # find_subfragments_topologically()
    # find_cyclic_subfragments(protein_frag, protein_frag_cyclic)
    # find_cyclic_subfragments(nucleic_frag, nucleic_frag_cyclic)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def analyze_molecule(
        int n_atoms,
        UINT32_t[:, :] bonds,
        INT64_t[:] resid,
        UINT32_t[:] insertion,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        bool[:] protein,
        bool[:] nucleic,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        UINT32_t[:] uq_resid,
        bool[:] sg_atoms,
        UINT32_t[:] fragments,
        # UINT32_t[:] protein_frag,
        # UINT32_t[:] nucleic_frag,
        # UINT32_t[:] protein_frag_cyclic,
        # UINT32_t[:] nucleic_frag_cyclic,
    ):
    cdef bool[:] flgs = np.zeros(n_atoms, dtype=np.bool)
    cdef int n_residues

    atom_bonds = _get_atom_bonds(n_atoms, bonds)

    # Create unique residue IDs and residue list
    residue_atoms = _make_uniq_resids(n_atoms, atom_bonds, resid, insertion, chain_id, seg_id, uq_resid, flgs)

    # Mark atoms as protein or nucleic
    cdef UINT32_t[:] atm_flgs = np.zeros(n_atoms, dtype=UINT32)
    _find_residues(n_atoms, protein, nucleic, protein_bb, nucleic_bb, resid, chain_id, seg_id, atom_bonds, atm_flgs)

    n_residues = uq_resid[n_atoms-1] + 1
    cdef bool[:] res_flgs = np.zeros(n_residues, dtype=np.bool)
    _find_fragments(n_residues, n_atoms, residue_atoms, atom_bonds, uq_resid, chain_id, seg_id, sg_atoms, res_flgs, fragments)#, protein_frag, nucleic_frag, protein_frag_cyclic, nucleic_frag_cyclic)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def analyze_molecule2(
        int n_atoms,
        UINT32_t[:, :] bonds
    ):
    cdef vector[int] atom_bonds
    _get_atom_bonds2(n_atoms, bonds, atom_bonds) 

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _get_atom_bonds2(
        int n_atoms,
        UINT32_t[:, :] bonds,
        vector[int] atom_bonds,
    ):
    
    cdef int n_bonds, b1, b2, i

    # Create nested list of bonded atoms
    for i in range(n_atoms):
        atom_bonds.push_back(i)

    # n_bonds = bonds.shape[0]
    # for i in range(n_bonds):
    #     b1 = bonds[i, 0]
    #     b2 = bonds[i, 1]
    #     atom_bonds[b1].push_back(b2)
    #     atom_bonds[b2].push_back(b1)
    return True