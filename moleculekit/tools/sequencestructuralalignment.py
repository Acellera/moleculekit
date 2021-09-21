# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import logging

logger = logging.getLogger(__name__)


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

    seqmol = mol.sequence()
    seqref = ref.sequence()

    if molseg is None:
        msg = "No segment was specified by the user for `mol`"
        if len(seqmol) > 1:
            msg += f" and multiple segments ({list(seqmol.keys())}) were detected. "
        else:
            msg += ". "
        msg += "Alignment will be done on all protein segments."
        logger.info(msg)
        seqmol = mol.sequence(noseg=True)
        molseg = list(seqmol.keys())[0]
    if refseg is None:
        msg = "No segment was specified by the user for `ref`"
        if len(seqref) > 1:
            msg += f" and multiple segments ({list(seqref.keys())}) were detected. "
        else:
            msg += ". "
        msg += "Alignment will be done on all protein segments."
        logger.info(msg)
        seqref = ref.sequence(noseg=True)
        refseg = list(seqref.keys())[0]

    def getSegIdx(m, mseg):
        # Calculate the atoms which belong to the selected segments
        if isinstance(mseg, str) and mseg == "protein":
            msegidx = m.atomselect("protein and name CA")
        else:
            msegidx = np.zeros(m.numAtoms, dtype=bool)
            for seg in ensurelist(mseg):
                msegidx |= (m.segid == seg) & (m.name == "CA")
        return np.where(msegidx)[0]

    molsegidx = getSegIdx(mol, molseg)
    refsegidx = getSegIdx(ref, refseg)

    # Create fake residue numbers for the selected segment
    from moleculekit.util import sequenceID

    molfakeresid = sequenceID(
        (mol.resid[molsegidx], mol.insertion[molsegidx], mol.chain[molsegidx])
    )
    reffakeresid = sequenceID(
        (ref.resid[refsegidx], ref.insertion[refsegidx], ref.chain[refsegidx])
    )

    # TODO: Use BLOSUM62?
    alignments = pairwise2.align.globaldx(
        seqref[refseg], seqmol[molseg], matlist.blosum62
    )
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
