# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import logging
from moleculekit.molecule import Molecule
import unittest

logger = logging.getLogger(__name__)


def _get_sequence(mol: Molecule, sel):
    sel_mask = mol.atomselect(sel)
    protein_mask = mol.atomselect("protein")
    nucleic_mask = mol.atomselect("nucleic")
    if any(protein_mask & sel_mask) and any(nucleic_mask & sel_mask):
        raise RuntimeError(
            "Your selection contains both protein and nucleic residues. You need to clarify which selection to align."
        )
    molseg = "protein" if any(protein_mask) else "nucleic"
    seqmol, seqidx = mol.sequence(noseg=True, return_idx=True, sel=sel, _logger=False)
    seqidx = seqidx[molseg]
    seqmol = seqmol[molseg]
    segment_type = molseg
    return seqmol, seqidx, segment_type


def _get_max_align_region(molaln, refaln, nalignfragment):
    dsig = np.hstack(([False], (refaln != "-") & (molaln != "-"), [False])).astype(int)
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
    return _list_starts, _list_finish


def _align_by_sequence_alignment(
    mol, molidx, ref, refidx, segment_type, alignments, maxalignments, nalignfragment
):
    numaln = len(alignments)

    if numaln > maxalignments:
        logger.warning(
            f"{numaln} alignments found. Limiting to {maxalignments} as specified in the `maxalignments` argument."
        )

    def _find_common_atom(mol, ref, idxmol, idxref):
        moln = mol.name[idxmol].tolist()
        refn = ref.name[idxref].tolist()
        prot_names = ["CA", "C", "O", "N"]
        nucl_names = ["C3'", "C4'", "C1'", "P"]
        names = prot_names if segment_type == "protein" else nucl_names
        for name in names:
            if name in moln and name in refn:
                return idxmol[moln.index(name)], idxref[refn.index(name)]
        raise RuntimeError(
            f"Could not find any of {names} in both mol and ref residues"
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
        starts, ends = _get_max_align_region(molaln, refaln, nalignfragment)

        _refidx = []
        _molidx = []
        startends = []
        for ss, ff in zip(starts, ends):
            for k, (rr, mr) in enumerate(zip(residref[ss:ff], residmol[ss:ff])):
                mi, ri = _find_common_atom(mol, ref, molidx[mr], refidx[rr])
                _refidx.append(ri)
                _molidx.append(mi)
                # The rest is just for pretty printing
                if k == 0:
                    start_resid = mol.resid[mi]
                if k == ff - ss - 1:
                    end_resid = mol.resid[mi]
            startends.append(f"{start_resid}-{end_resid}")

        molmask = np.zeros(mol.numAtoms, dtype=bool)
        molmask[_molidx] = True
        refmask = np.zeros(ref.numAtoms, dtype=bool)
        refmask[_refidx] = True

        logger.info(
            f"Alignment #{i} was done on {len(_refidx)} residues: {' ,'.join(startends)}"
        )

        alignedmol = mol.copy()
        alignedmol.align(sel=molmask, refmol=ref, refsel=refmask)
        alignedstructs.append(alignedmol)
        masks.append([molmask, refmask])

    return alignedstructs, masks


def sequenceStructureAlignment(
    mol,
    ref,
    molseg=None,
    refseg=None,
    molsel="all",
    refsel="all",
    maxalignments=10,
    nalignfragment=1,
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
    if molseg is not None:
        logger.warning(
            'molseg argument is deprecated. Please use molsel="segid X" instead.'
        )
        molsel = f"segid {molseg}"
    if refseg is not None:
        logger.warning(
            'refseg argument is deprecated. Please use refsel="segid X" instead.'
        )
        refsel = f"segid {refseg}"

    try:
        from Bio import pairwise2
    except ImportError:
        raise ImportError(
            "You need to install the biopython package to use this function. Try using `conda install biopython`."
        )
    from Bio.Align import substitution_matrices

    blosum62 = substitution_matrices.load("BLOSUM62")

    if len([x for x in np.unique(mol.altloc) if len(x)]) > 1:
        raise RuntimeError(
            "Alternative atom locations detected in `mol`. Please remove these before calling this function."
        )
    if len([x for x in np.unique(ref.altloc) if len(x)]) > 1:
        raise RuntimeError(
            "Alternative atom locations detected in `ref`. Please remove these before calling this function."
        )

    seqmol, molidx, segment_type_mol = _get_sequence(mol, molsel)
    seqref, refidx, segment_type_ref = _get_sequence(ref, refsel)
    if segment_type_mol != segment_type_ref:
        raise RuntimeError(
            f"The segments of mol are of type {segment_type_mol} while for ref they are of type {segment_type_ref}. Please choose different segments to align."
        )
    segment_type = segment_type_mol

    if segment_type == "protein":
        # -11 is gap creation penalty. -1 is gap extension penalty. Taken from https://www.arabidopsis.org/Blast/BLASToptions.jsp BLASTP options
        alignments = pairwise2.align.globalds(seqref, seqmol, blosum62, -11.0, -1.0)
    elif segment_type == "nucleic":
        alignments = pairwise2.align.globalxx(seqref, seqmol)

    alignedstructs, masks = _align_by_sequence_alignment(
        mol, molidx, ref, refidx, segment_type, alignments, maxalignments, nalignfragment
    )
    return alignedstructs, masks



class _TestSequenceStructuralAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.home import home

        self.home = home(dataDir="test-sequence-alignment")

    def test_align_protein(self):
        import os

        mol1 = Molecule(os.path.join(self.home, "4OBE.pdb"))
        mol2 = Molecule(os.path.join(self.home, "6OB2.pdb"))
        refmol = Molecule(os.path.join(self.home, "4OBE_aligned.pdb"))
        mol3, _ = sequenceStructureAlignment(
            mol1, mol2, molsel="protein", refsel="protein"
        )

        assert np.allclose(refmol.coords, mol3[0].coords, atol=1e-3)

    def test_align_rna(self):
        import os

        mol1 = Molecule(os.path.join(self.home, "5C45.pdb"))
        mol2 = Molecule(os.path.join(self.home, "5C45_sim.pdb"))
        refmol = Molecule(os.path.join(self.home, "5C45_aligned.pdb"))
        mol3, _ = sequenceStructureAlignment(mol1, mol2)

        assert np.allclose(refmol.coords, mol3[0].coords, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
