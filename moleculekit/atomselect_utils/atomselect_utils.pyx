# cython: cdivision=True

import numpy as np
from math import sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool
from libcpp.string cimport string
from libc.string cimport strcmp
from libc.stdio cimport printf
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
cdef bool _make_uniq_resids(
        int n_atoms,
        vector[vector[int]] &atom_bonds,
        vector[vector[int]] &residue_atoms,
        INT64_t[:] resid,
        UINT32_t[:] insertion,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        UINT32_t[:] uq_resid,
        bool[:] flgs
    ):
    cdef stack[int] st
    cdef int i, j, bi, res
    cdef UINT32_t ins, num_residues

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
    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _get_atom_bonds(
        int n_atoms,
        UINT32_t[:, :] bonds,
        vector[vector[int]] &atom_bonds,
    ):
    
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

    return True


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
        vector[vector[int]] &atom_bonds,
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
        vector[vector[int]] &atom_bonds,
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
        vector[vector[int]] &atom_bonds,
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
        vector[vector[int]] &atom_bonds,
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
        vector[vector[int]] &atom_bonds,
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
        vector[vector[int]] &residue_atoms,
        vector[vector[int]] &atom_bonds,
        UINT32_t[:] uq_resid,
        UINT32_t[:] fragments,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        vector[string] &atomnames,
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

                        if (ri != i and res_flgs[ri] == 0 and 
                            chain_id[ra] == chain_id[ba] and 
                            seg_id[ra] == seg_id[ba] and 
                            not (strcmp(atomnames[ra].c_str(), "SG") == 0 and strcmp(atomnames[ba].c_str(), "SG") == 0)):
                            res_flgs[ri] = True
                            st.push(ri)
            count += 1
    return True


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_connected_subfragment(
        int resnum,
        int fragnum,
        const char* endatom,
        const char* altendatom,
        const char* alt2endatom,
        bool[:] res_flgs,
        bool[:] polymer,
        bool[:] backbone,
        vector[string] &atomnames,
        UINT32_t[:] uq_resid,
        vector[vector[int]] &residue_atoms,
        vector[vector[int]] &atom_bonds,
        vector[vector[int]] &pfragList,
    ):
    cdef int ra, ba

    if res_flgs[resnum] or not polymer[residue_atoms[resnum][0]]:
        return False

    pfragList[fragnum].push_back(resnum) # Add residue to protein fragment list
    res_flgs[resnum] = True # Avoid repeats

    for ra in residue_atoms[resnum]:
        if (strcmp(atomnames[ra].c_str(), endatom) != 0 and 
            strcmp(atomnames[ra].c_str(), altendatom) != 0 and 
            strcmp(atomnames[ra].c_str(), alt2endatom) != 0):
            continue

        for ba in atom_bonds[ra]: # Look at the bonds
            if (backbone[ra] or backbone[ba]) and uq_resid[ba] != resnum and not res_flgs[uq_resid[ba]]:
                _find_connected_subfragment(uq_resid[ba], fragnum, endatom, altendatom, alt2endatom, res_flgs, polymer, backbone, atomnames, uq_resid, residue_atoms, atom_bonds, pfragList)
                return True # Only find one, assume no branching
    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_subfragments(
        int n_residues,
        vector[vector[int]] &residue_atoms,
        vector[vector[int]] &atom_bonds,
        UINT32_t[:] uq_resid,
        bool[:] res_flgs,
        bool[:] protein_bb,
        vector[string] &atomnames,
        bool[:] protein,
        vector[vector[int]] &pfragList,
    ):
    cdef bool off_res_bonds = False

    for i in range(n_residues):
        if res_flgs[i] or not protein[residue_atoms[i][0]]: # Has been seen before or isn't protein
            continue

        for ra in residue_atoms[i]:  # Does the residue have a matching starting atom?
            if strcmp(atomnames[ra].c_str(), "N") != 0:
                continue

            off_res_bonds = False
            for ba in atom_bonds[ra]:
                if uq_resid[ba] != i and protein[ba]: # Are there off-residue bonds to protein?
                    off_res_bonds = True
                    break

            if not off_res_bonds:
                pfragList.push_back(vector[int]())
                _find_connected_subfragment(i, pfragList.size() - 1, "C", "", "", res_flgs, protein, protein_bb, atomnames, uq_resid, residue_atoms, atom_bonds, pfragList)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_subfragments_topologically(
        int n_residues,
        vector[vector[int]] &residue_atoms,
        vector[vector[int]] &atom_bonds,
        UINT32_t[:] uq_resid,
        bool[:] res_flgs,
        bool[:] nucleic_bb,
        vector[string] &atomnames,
        bool[:] nucleic,
        vector[vector[int]] &nfragList,
    ):
    cdef int offresbondcount = 0

    for i in range(n_residues):
        if res_flgs[i] or not nucleic[residue_atoms[i][0]]: # Has been seen before or isn't nucleic
            continue
        
        for ra in residue_atoms[i]:  # Does the residue have a matching starting atom?
            offresbondcount = 0
            for ba in atom_bonds[ra]:
                if uq_resid[ba] != i and nucleic[ba]: # Are there off-residue bonds to protein?
                    offresbondcount +=1
                    
            if offresbondcount == 1: # Valid fragment start atom. Find residues downchain
                nfragList.push_back(vector[int]())
                _find_connected_subfragment(i, nfragList.size() - 1, "O3'", "O3*", "H3T", res_flgs, nucleic, nucleic_bb, atomnames, uq_resid, residue_atoms, atom_bonds, nfragList)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_fragments(
        int n_residues,
        int n_atoms,
        vector[vector[int]] &residue_atoms,
        vector[vector[int]] &atom_bonds,
        vector[vector[int]] &pfragList,
        vector[vector[int]] &nfragList,
        UINT32_t[:] uq_resid,
        UINT32_t[:] chain_id,
        UINT32_t[:] seg_id,
        vector[string] &atomnames,
        bool[:] res_flgs,
        bool[:] protein,
        bool[:] nucleic,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        UINT32_t[:] fragments,
    ):
    _find_connected_fragments(n_residues, n_atoms, residue_atoms, atom_bonds, uq_resid, fragments, chain_id, seg_id, atomnames, res_flgs)
    for r in range(n_residues):  # Zero the residue flags
        res_flgs[r] = 0
    _find_subfragments(n_residues, residue_atoms, atom_bonds, uq_resid, res_flgs, protein_bb, atomnames, protein, pfragList)
    _find_subfragments_topologically(n_residues, residue_atoms, atom_bonds, uq_resid, res_flgs, nucleic_bb, atomnames, nucleic, nfragList)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _fix_backbones(
        bool[:] protein,
        bool[:] nucleic,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
    ):
    # Fix BB atoms by unmarking them if they are not polymers
    # This is necessary since we use just N CA C O names and other
    # molecules such as waters or ligands might have them
    cdef int n_atoms = protein.shape[0]
    for i in range(n_atoms):
        if protein_bb[i] and not protein[i]:
            protein_bb[i] = False
        if nucleic_bb[i] and not nucleic[i]:
            nucleic_bb[i] = False

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
        UINT32_t[:] fragments,
        FLOAT32_t[:] masses,
        UINT32_t[:] sidechain,
        vector[string] &atomnames,
    ):
    cdef bool[:] flgs = np.zeros(n_atoms, dtype=np.bool_)
    cdef vector[vector[int]] atom_bonds
    cdef vector[vector[int]] residue_atoms
    cdef vector[vector[int]] pfragList
    cdef vector[vector[int]] nfragList
    cdef int n_residues

    _get_atom_bonds(n_atoms, bonds, atom_bonds)

    # Create unique residue IDs and residue list
    _make_uniq_resids(n_atoms, atom_bonds, residue_atoms, resid, insertion, chain_id, seg_id, uq_resid, flgs)

    # Mark atoms as protein or nucleic
    cdef UINT32_t[:] atm_flgs = np.zeros(n_atoms, dtype=UINT32)
    _find_residues(n_atoms, protein, nucleic, protein_bb, nucleic_bb, resid, chain_id, seg_id, atom_bonds, atm_flgs)

    n_residues = uq_resid[n_atoms-1] + 1
    cdef bool[:] res_flgs = np.zeros(n_residues, dtype=np.bool_)
    _find_fragments(n_residues, n_atoms, residue_atoms, atom_bonds, pfragList, nfragList, uq_resid, chain_id, seg_id, atomnames, res_flgs, protein, nucleic, protein_bb, nucleic_bb, fragments)#, protein_frag, nucleic_frag, protein_frag_cyclic, nucleic_frag_cyclic)

    _fix_backbones(protein, nucleic, protein_bb, nucleic_bb)

    _atomsel_sidechain(residue_atoms, atom_bonds, pfragList, protein_bb, nucleic_bb, atomnames, masses, sidechain)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _recursive_find_sidechain_atoms(
        int atom_index, 
        vector[vector[int]] &atom_bonds,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        UINT32_t[:] sidechain,
    ):
    cdef int ba

    # Have we been here?
    if sidechain[atom_index] == 2:
        return False

    # Is it a backbone atom?
    if protein_bb[atom_index] or nucleic_bb[atom_index]:
        return False

    sidechain[atom_index] = 2

    # Try the atoms it's bonded to
    for ba in atom_bonds[atom_index]:
        _recursive_find_sidechain_atoms(ba, atom_bonds, protein_bb, nucleic_bb, sidechain)

    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _find_sidechain_atoms(
        int n_atoms, 
        vector[vector[int]] &atom_bonds,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        UINT32_t[:] sidechain,
    ):
    cdef int i

    for i in range(n_atoms):
        if sidechain[i]:
            _recursive_find_sidechain_atoms(i, atom_bonds, protein_bb, nucleic_bb, sidechain)

    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef bool _atomsel_sidechain(
        vector[vector[int]] &residue_atoms,
        vector[vector[int]] &atom_bonds,
        vector[vector[int]] &pfragList,
        bool[:] protein_bb,
        bool[:] nucleic_bb,
        vector[string] &atomnames,
        FLOAT32_t[:] masses,
        UINT32_t[:] sidechain,
    ):
    cdef int ca_idx, b1, b2, c1, c2
    cdef FLOAT32_t m1, m2
    cdef int n_atoms = protein_bb.shape[0]
    cdef int n_fragments = pfragList.size()

    for f in range(n_fragments):
        for res in pfragList[f]:

            ca_idx = -1
            for i in residue_atoms[res]:
                if strcmp(atomnames[i].c_str(), "CA") == 0:
                    ca_idx = i

            if ca_idx < 0:
                # printf("Atomselection: sidechain: cannot find CA atom\n")
                continue

            # Find at most two neighbours of CA which are not on the backbone
            b1 = -1
            b2 = -1
            for ba in atom_bonds[ca_idx]:
                if not protein_bb[ba] and not nucleic_bb[ba]:
                    if b1 == -1:
                        b1 = ba
                    else:
                        if b2 == -1:
                            b2 = ba
                        else:
                            # printf("Atomselection: sidechain: protein residue index %i, CA atom idx %i has more than two non-backbone bonds. Ignoring the others\n", res, ca_idx)
                            pass

            if b1 == -1:
                continue

            if b2 != -1: # Find the right one. Check number of atoms to see if they have a lone H
                c1 = atom_bonds[b1].size()
                c2 = atom_bonds[b2].size()

                if c1 == 1 and c2 > 1:
                    b1 = b2
                elif c2 == 1 and c1 > 1:
                    b1 = b1
                elif c1 == 1 and c2 == 1:
                    # check the masses
                    m1 = masses[b1]
                    m2 = masses[b2]
                    if m1 > 2.3 and m2 <= 2.3:
                        b1 = b2
                    elif m2 > 2.3 and m1 <= 2.3:
                        b1 = b1
                    elif m1 <= 2.0 and m2 <= 2.3:
                        # should have two H's, find the "first" of those
                        if strcmp(atomnames[b1].c_str(), atomnames[b2].c_str()) > 0:
                            b1 = b2
                    else:
                        # printf("Atomselect: sidechain: protein residue index %i, CA atom idx %i has sidechain-like atom (indices %i and %i) and we cannot determine which to call a sidechain. Taking a guess...\n", res, ca_idx, b1, b2)
                        if strcmp(atomnames[b1].c_str(), atomnames[b2].c_str()) > 0:
                            b1 = b2
            sidechain[b1] = 1

    _find_sidechain_atoms(n_atoms, atom_bonds, protein_bb, nucleic_bb, sidechain)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def within_distance(
        FLOAT32_t[:, :] coords,
        float cutoff,
        UINT32_t[:] source,
        FLOAT32_t[:] source_min,
        FLOAT32_t[:] source_max,
        bool[:] mask,
    ):
    # TODO: Improve with neighbour list / cell list implementation
    cdef int n_atoms = coords.shape[0]
    cdef int n_source = source.shape[0]
    cdef bool close_enough
    cdef int i, j, k
    cdef FLOAT32_t diff, sq_cutoff

    sq_cutoff = cutoff * cutoff

    for i in range(n_atoms):
        # Trivial check if it's too far from all source atoms
        close_enough = False
        for k in range(3):
            if coords[i, k] >= (source_min[k] - cutoff):
                close_enough = True
                break
            if coords[i, k] <= (source_max[k] + cutoff):
                close_enough = True
                break

        if not close_enough:
            continue

        for j in range(n_source):
            diff = 0
            for k in range(3):
                diff += (coords[i, k] - coords[source[j], k]) ** 2
            if diff < sq_cutoff:
                mask[i] = True
                break

