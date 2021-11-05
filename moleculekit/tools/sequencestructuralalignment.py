# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _get_sequence(mol, segs, protein_mask, nucleic_mask):
    from moleculekit.util import ensurelist

    if segs is None:
        if any(protein_mask) and any(nucleic_mask):
            raise RuntimeError(
                "mol contains both protein and nucleic residues. You need to specify which segments to align."
            )
        molseg = "protein" if any(protein_mask) else "nucleic"
        seqmol = mol.sequence(noseg=True)[molseg]
        segment_type = molseg
    else:
        seqs = mol.sequence()
        seqmol = ""
        has_prot = False
        has_nucl = False
        for seg in ensurelist(segs):
            seg_is_prot = any(protein_mask & (mol.segid == seg))
            seg_is_nucl = any(nucleic_mask & (mol.segid == seg))
            if seg_is_prot and seg_is_nucl:
                raise RuntimeError(
                    f"Segment {seg} of mol contains both protein and nucleic atoms. Please reassign segments."
                )
            has_prot = has_prot | seg_is_prot
            has_nucl = has_nucl | seg_is_nucl
            seqmol += seqs[seg]
        if has_prot and has_nucl:
            raise RuntimeError(
                "The segments you provided contain both protein and nucleic atoms. Please choose different segments"
            )
        segment_type = "protein" if has_prot else "nucleic"
    return seqmol, segment_type


def sequenceStructureAlignment(
    mol, ref, molseg=None, refseg=None, maxalignments=10, nalignfragment=1
):
    """Aligns two structures by their longests sequences alignment

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The Molecule we want to align
    ref : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The reference Molecule to which we want to align
    molseg : str
        The segment of `mol` we want to align
    refseg : str
        The segment of `ref` we want to align to
    maxalignments : int
        The maximum number of alignments we want to produce
    nalignfragment : int
        The number of fragments used for the alignment.

    Returns
    -------
    mols : list
        A list of Molecules each containing a different alignment.
    masks : list
        A list of boolean mask pairs of which atoms were aligned. [[molmask1, refmask1], [molmask2, refmask2], ...]
    """
    from moleculekit.util import ensurelist

    try:
        from Bio import pairwise2
    except ImportError:
        raise ImportError(
            "You need to install the biopython package to use this function. Try using `conda install biopython`."
        )
    from Bio.SubsMat import MatrixInfo as matlist

    if len([x for x in np.unique(mol.altloc) if len(x)]) > 1:
        raise RuntimeError(
            "Alternative atom locations detected in `mol`. Please remove these before calling this function."
        )
    if len([x for x in np.unique(ref.altloc) if len(x)]) > 1:
        raise RuntimeError(
            "Alternative atom locations detected in `ref`. Please remove these before calling this function."
        )

    mol_protein = mol.atomselect("protein")
    mol_nucleic = mol.atomselect("nucleic")
    ref_protein = ref.atomselect("protein")
    ref_nucleic = ref.atomselect("nucleic")

    seqmol, segment_type_mol = _get_sequence(mol, molseg, mol_protein, mol_nucleic)
    seqref, segment_type_ref = _get_sequence(ref, refseg, ref_protein, ref_nucleic)
    if segment_type_mol != segment_type_ref:
        raise RuntimeError(
            f"The segments of mol are of type {segment_type_mol} while for ref they are of type {segment_type_ref}. Please choose different segments to align."
        )
    segment_type = segment_type_mol

    for seg in seqref:
        seg_is_prot = any(ref_protein & (ref.segid == seg))
        seg_is_nucl = any(ref_nucleic & (ref.segid == seg))
        if seg_is_prot and seg_is_nucl:
            raise RuntimeError(
                f"Segment {seg} of ref contains both protein and nucleic atoms. Please reassign segments."
            )

    def getSegIdx(m, seg, segment_type):
        # Calculate the atoms which belong to the selected segments
        if seg is None and segment_type == "protein":
            msegidx = m.atomselect("protein and name CA")
        elif seg is None and segment_type == "nucleic":
            msegidx = m.atomselect("nucleic and backbone and name P")
        else:
            msegidx = np.zeros(m.numAtoms, dtype=bool)
            atom = "CA" if segment_type == "protein" else "P"
            for seg in ensurelist(seg):
                msegidx |= (m.segid == seg) & (m.name == atom)
        return np.where(msegidx)[0]

    molsegidx = getSegIdx(mol, molseg, segment_type)
    refsegidx = getSegIdx(ref, refseg, segment_type)

    # Create fake residue numbers for the selected segment
    from moleculekit.util import sequenceID

    molfakeresid = sequenceID(
        (mol.resid[molsegidx], mol.insertion[molsegidx], mol.chain[molsegidx])
    )
    reffakeresid = sequenceID(
        (ref.resid[refsegidx], ref.insertion[refsegidx], ref.chain[refsegidx])
    )

    if segment_type == "protein":
        alignments = pairwise2.align.globaldx(seqref, seqmol, matlist.blosum62)
    elif segment_type == "nucleic":
        alignments = pairwise2.align.globalxx(seqref, seqmol)
    numaln = len(alignments)

    if numaln > maxalignments:
        logger.warning(
            f"{numaln} alignments found. Limiting to {maxalignments} as specified in the `maxalignments` argument."
        )

    alignedstructs = []
    masks = []
    for i in range(min(maxalignments, numaln)):
        refaln = np.array(list(alignments[i][0]))
        molaln = np.array(list(alignments[i][1]))

        # By doing cumsum we calculate how many letters were before the current letter (i.e. residues before current)
        residref = np.cumsum(refaln != "-") - 1  # Start them from 0
        residmol = np.cumsum(molaln != "-") - 1  # Start them from 0

        # Find the region of maximum alignment between the molecules
        dsig = np.hstack(([False], (refaln != "-") & (molaln != "-"), [False])).astype(
            int
        )
        dsigdiff = np.diff(dsig)
        startIndex = np.where(dsigdiff > 0)[0]
        endIndex = np.where(dsigdiff < 0)[0]
        duration = endIndex - startIndex
        duration_sorted = np.sort(duration)[::-1]

        _list_starts = []
        _list_finish = []
        for n in range(nalignfragment):
            if n == len(duration):
                break
            idx = np.where(duration == duration_sorted[n])[0]
            start = startIndex[idx][0]
            finish = endIndex[idx][0]
            _list_starts.append(start)
            _list_finish.append(finish)

        # Get the "resids" of the aligned residues only
        refalnresid = np.concatenate(
            [
                residref[start:finish]
                for start, finish in zip(_list_starts, _list_finish)
            ]
        )
        molalnresid = np.concatenate(
            [
                residmol[start:finish]
                for start, finish in zip(_list_starts, _list_finish)
            ]
        )
        refidx = []
        for r in refalnresid:
            refidx += list(refsegidx[reffakeresid == r])
        molidx = []
        for r in molalnresid:
            molidx += list(molsegidx[molfakeresid == r])

        molmask = np.zeros(mol.numAtoms, dtype=bool)
        molmask[molidx] = True
        refmask = np.zeros(ref.numAtoms, dtype=bool)
        refmask[refidx] = True

        start_residues = np.concatenate(
            [mol.resid[molsegidx[molfakeresid == residmol[r]]] for r in _list_starts]
        )
        finish_residues = np.concatenate(
            [
                mol.resid[molsegidx[molfakeresid == residmol[r - 1]]]
                for r in _list_finish
            ]
        )
        joined = ", ".join(
            [f"{s}-{f}" for s, f in zip(start_residues, finish_residues)]
        )
        logger.info(
            f"Alignment #{i} was done on {len(refalnresid)} residues: mol segid {np.unique(mol.segid[molidx])[0]} resid {joined}"
        )

        alignedmol = mol.copy()
        alignedmol.align(sel=molmask, refmol=ref, refsel=refmask)
        alignedstructs.append(alignedmol)
        masks.append([molmask, refmask])

    return alignedstructs, masks
