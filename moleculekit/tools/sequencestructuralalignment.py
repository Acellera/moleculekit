import numpy as np
import logging
logger = logging.getLogger(__name__)


def sequenceStructureAlignment(mol, ref, molseg=None, refseg=None, maxalignments=10, nalignfragment=1):
    """ Aligns two structures by their longests sequences alignment

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
    """
    from moleculekit.util import ensurelist
    try:
        from Bio import pairwise2
    except ImportError as e:
        raise ImportError('You need to install the biopython package to use this function. Try using `conda install biopython`.')
    from Bio.SubsMat import MatrixInfo as matlist

    if len([x for x in np.unique(mol.altloc) if len(x)]) > 1:
        raise RuntimeError('Alternative atom locations detected in `mol`. Please remove these before calling this function.')
    if len([x for x in np.unique(ref.altloc) if len(x)]) > 1:
        raise RuntimeError('Alternative atom locations detected in `ref`. Please remove these before calling this function.')

    seqmol = mol.sequence()
    seqref = ref.sequence()

    if molseg is None and len(seqmol) > 1:
        logger.info('Multiple segments ({}) detected in `mol`. Alignment will be done on all. Otherwise please specify which segment to align.'.format(list(seqmol.keys())))
        seqmol = mol.sequence(noseg=True)
        molseg = list(seqmol.keys())[0]
    if refseg is None and len(seqref) > 1:
        logger.info('Multiple segments ({}) detected in `ref`. Alignment will be done on all. Otherwise please specify which segment to align.'.format(list(seqref.keys())))
        seqref = ref.sequence(noseg=True)
        refseg = list(seqref.keys())[0]

    def getSegIdx(m, mseg):
        # Calculate the atoms which belong to the selected segments
        if isinstance(mseg, str) and mseg == 'protein':
            msegidx = m.atomselect('protein and name CA')
        else:
            msegidx = np.zeros(m.numAtoms, dtype=bool)
            for seg in ensurelist(mseg):
                msegidx |= (m.segid == seg) & (m.name == 'CA')
        return np.where(msegidx)[0]
    molsegidx = getSegIdx(mol, molseg)
    refsegidx = getSegIdx(ref, refseg)

    # Create fake residue numbers for the selected segment
    from moleculekit.util import sequenceID
    molfakeresid = sequenceID((mol.resid[molsegidx], mol.insertion[molsegidx], mol.chain[molsegidx]))
    reffakeresid = sequenceID((ref.resid[refsegidx], ref.insertion[refsegidx], ref.chain[refsegidx]))

    # TODO: Use BLOSUM62?
    alignments = pairwise2.align.globaldx(seqref[refseg], seqmol[molseg], matlist.blosum62)
    numaln = len(alignments)

    if numaln > maxalignments:
        logger.warning('{} alignments found. Limiting to {} as specified in the `maxalignments` argument.'.format(numaln, maxalignments))

    alignedstructs = []
    for i in range(min(maxalignments, numaln)):
        refaln = np.array(list(alignments[i][0]))
        molaln = np.array(list(alignments[i][1]))
 
        # By doing cumsum we calculate how many letters were before the current letter (i.e. residues before current)
        residref = np.cumsum(refaln != '-') - 1  # Start them from 0
        residmol = np.cumsum(molaln != '-') - 1  # Start them from 0

        # Find the region of maximum alignment between the molecules
        dsig = np.hstack(([False], (refaln != '-') & (molaln != '-'), [False])).astype(int)
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
        refalnresid = np.concatenate([ residref[start:finish] for start, finish in zip(_list_starts,_list_finish)])
        molalnresid = np.concatenate([ residmol[start:finish] for start, finish in zip(_list_starts, _list_finish) ])
        refidx = []
        for r in refalnresid:
            refidx += list(refsegidx[reffakeresid == r])
        molidx = []
        for r in molalnresid:
            molidx += list(molsegidx[molfakeresid == r])        

        molboolidx = np.zeros(mol.numAtoms, dtype=bool)
        molboolidx[molidx] = True
        refboolidx = np.zeros(ref.numAtoms, dtype=bool)
        refboolidx[refidx] = True

        start_residues = np.concatenate([ mol.resid[molsegidx[molfakeresid == residmol[r]]] for r in _list_starts])
        finish_residues = np.concatenate([ mol.resid[molsegidx[molfakeresid == residmol[r-1]]] for r in _list_finish])
        logger.info('Alignment #{} was done on {} residues: mol segid {} resid {}'.format(
            i, len(refalnresid), np.unique(mol.segid[molidx])[0], ', '.join(['{}-{}'.format(s,f) for s, f in zip(start_residues,finish_residues)])  ))

        alignedmol = mol.copy()
        alignedmol.align(molboolidx, ref, refboolidx)
        alignedstructs.append(alignedmol)

    return alignedstructs