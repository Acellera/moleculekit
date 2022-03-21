# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import string
import numpy as np
import logging


logger = logging.getLogger(__name__)


chain_alphabet = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
segid_alphabet = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)


def autoSegment(
    mol,
    sel="all",
    basename="P",
    spatial=True,
    spatialgap=4.0,
    fields=("segid",),
    field=None,
    _logger=True,
):
    """Detects resid gaps in a selection and assigns incrementing segid to each fragment

    !!!WARNING!!! If you want to use atom selections like 'protein' or 'fragment',
    use this function on a Molecule containing only protein atoms, otherwise the protein selection can fail.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The Molecule object
    sel : str
        Atom selection string on which to check for gaps.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    basename : str
        The basename for segment ids. For example if given 'P' it will name the segments 'P1', 'P2', ...
    spatial : bool
        Only considers a discontinuity in resid as a gap if the CA atoms have distance more than `spatialgap` Angstrom
    spatialgap : float
        The size of a spatial gap which validates a discontinuity (A)
    fields : list
        Fields in which to set the segments. Must be a combination of "chain", "segid" or only one of them.

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A new Molecule object with modified segids

    Example
    -------
    >>> newmol = autoSegment(mol, "chain B", "P", fields=("chain", "segid"))
    """
    from moleculekit.util import sequenceID

    if field is not None and isinstance(field, str):
        if field == "both":
            fields = ("chain", "segid")
        else:
            fields = (field,)

    mol = mol.copy()

    idx = mol.atomselect(sel, indexes=True)
    rid = mol.resid[idx].copy()
    residiff = np.diff(rid)
    # Points to the index before the gap!
    gappos = np.where((residiff != 1) & (residiff != 0))[0]

    idxstartseg = [idx[0]] + idx[gappos + 1].tolist()
    idxendseg = idx[gappos].tolist() + [idx[-1]]

    # Letters to be used for chains, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    sel_mask = mol.atomselect(sel)
    used_chains = set(mol.chain[~sel_mask])
    available_chains = [x for x in chain_alphabet if x not in used_chains]
    used_segids = set([x[0] for x in mol.segid[~sel_mask] if x != ""])
    available_segids = [x for x in [basename] + segid_alphabet if x not in used_segids]
    basename = available_segids[0]

    if len(gappos) == 0:
        if "chain" in fields:
            mol.set("chain", available_chains[0], sel)
        if "segid" in fields:
            mol.set("segid", basename + "0", sel)
        return mol

    if spatial:
        residbackup = mol.resid.copy()
        # Assigning unique resids to be able to do the distance selection
        mol.set("resid", sequenceID(mol.resid))

        todelete = []
        i = 0
        for s, e in zip(idxstartseg[1:], idxendseg[:-1]):
            # Get the carbon alphas of both residues  ('coords', sel='resid "{}" "{}" and name CA'.format(mol.resid[e], mol.resid[s]))
            ca1coor = mol.coords[(mol.resid == mol.resid[e]) & (mol.name == "CA")]
            ca2coor = mol.coords[(mol.resid == mol.resid[s]) & (mol.name == "CA")]
            if len(ca1coor) and len(ca2coor):
                dist = np.sqrt(np.sum((ca1coor.squeeze() - ca2coor.squeeze()) ** 2))
                if dist < spatialgap:
                    todelete.append(i)
            i += 1
        todelete = np.array(todelete, dtype=int)
        # Join the non-real gaps into segments
        idxstartseg = np.delete(idxstartseg, todelete + 1)
        idxendseg = np.delete(idxendseg, todelete)

        mol.set("resid", residbackup)  # Restoring the original resids

    for i, (s, e) in enumerate(zip(idxstartseg, idxendseg)):
        if "chain" in fields:
            newchainid = available_chains[i % len(available_chains)]
            if _logger:
                logger.info(
                    f"Set chain {newchainid} between resid {mol.resid[s]} and {mol.resid[e]}."
                )
            mol.chain[s : e + 1] = newchainid
        if "segid" in fields:
            newsegid = basename + str(i)
            if _logger:
                logger.info(
                    f"Created segment {newsegid} between resid {mol.resid[s]} and {mol.resid[e]}."
                )
            mol.segid[s : e + 1] = newsegid

    return mol


def autoSegment2(
    mol,
    sel="(protein or resname ACE NME)",
    basename="P",
    fields=("segid",),
    residgaps=False,
    residgaptol=1,
    chaingaps=True,
    _logger=True,
):
    """Detects bonded segments in a selection and assigns incrementing segid to each segment

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The Molecule object
    sel : str
        Atom selection string on which to check for gaps.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    basename : str
        The basename for segment ids. For example if given 'P' it will name the segments 'P1', 'P2', ...
    fields : tuple of strings
        Field to fix. Can be "segid" (default) or any other Molecule field or combinations thereof.
    residgaps : bool
        Set to True to consider gaps in resids as structural gaps. Set to False to ignore resids
    residgaptol : int
        Above what resid difference is considered a gap. I.e. with residgaptol 1, 235-233 = 2 > 1 hence is a gap. We set
        default to 2 because in many PDBs single residues are missing in the proteins without any gaps.
    chaingaps : bool
        Set to True to consider changes in chains as structural gaps. Set to False to ignore chains

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A new Molecule object with modified segids

    Example
    -------
    >>> newmol = autoSegment2(mol)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if isinstance(fields, str):
        fields = (fields,)

    orig_sel = sel
    sel = f"({sel}) and backbone or (resname NME ACE and name N C O CH3)"  # Looking for bonds only over the backbone of the protein
    # Keep the original atom indexes to map from submol to mol
    idx = mol.atomselect(sel, indexes=True)
    # We filter out everything not on the backbone to calculate only those bonds
    submol = mol.copy()
    submol.filter(sel, _logger=False)
    bonds = submol._getBonds()  # Calculate both file and guessed bonds

    if residgaps:
        # Remove bonds between residues without continuous resids
        bondresiddiff = np.abs(submol.resid[bonds[:, 0]] - submol.resid[bonds[:, 1]])
        bonds = bonds[bondresiddiff <= residgaptol, :]
    else:
        # Warning about bonds bonding non-continuous resids
        bondresiddiff = np.abs(submol.resid[bonds[:, 0]] - submol.resid[bonds[:, 1]])
        if _logger and np.any(bondresiddiff > 1):
            for i in np.where(bondresiddiff > residgaptol)[0]:
                logger.warning(
                    "Bonds found between resid gaps: resid {} and {}".format(
                        submol.resid[bonds[i, 0]], submol.resid[bonds[i, 1]]
                    )
                )
    if chaingaps:
        # Remove bonds between residues without same chain
        bondsamechain = submol.chain[bonds[:, 0]] == submol.chain[bonds[:, 1]]
        bonds = bonds[bondsamechain, :]
    else:
        # Warning about bonds bonding different chains
        bondsamechain = submol.chain[bonds[:, 0]] == submol.chain[bonds[:, 1]]
        if _logger and np.any(bondsamechain == False):
            for i in np.where(bondsamechain == False)[0]:
                logger.warning(
                    "Bonds found between chain gaps: resid {}/{} and {}/{}".format(
                        submol.resid[bonds[i, 0]],
                        submol.chain[bonds[i, 0]],
                        submol.resid[bonds[i, 1]],
                        submol.chain[bonds[i, 1]],
                    )
                )

    # Calculate connected components using the bonds
    sparsemat = csr_matrix(
        (
            np.ones(bonds.shape[0] * 2),  # Values
            (
                np.hstack((bonds[:, 0], bonds[:, 1])),  # Rows
                np.hstack((bonds[:, 1], bonds[:, 0])),
            ),
        ),
        shape=[submol.numAtoms, submol.numAtoms],
    )  # Columns
    numcomp, compidx = connected_components(sparsemat, directed=False)

    # Letters to be used for chains, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    sel_mask = mol.atomselect(orig_sel)
    used_chains = set(mol.chain[~sel_mask])
    available_chains = [x for x in chain_alphabet if x not in used_chains]
    used_segids = set([x[0] for x in mol.segid[~sel_mask] if x != ""])
    available_segids = [x for x in [basename] + segid_alphabet if x not in used_segids]
    basename = available_segids[0]

    mol = mol.copy()
    prevsegres = None
    for i in range(numcomp):  # For each connected component / segment
        segid = basename + str(i)
        backboneSegIdx = idx[compidx == i]  # The backbone atoms of the segment
        segres = mol.atomselect(
            f"same residue as index {' '.join(map(str, backboneSegIdx))}"
        )  # Get whole residues

        # Warning about separating segments with continuous resids
        if (
            _logger
            and i > 0
            and (np.min(mol.resid[segres]) - np.max(mol.resid[prevsegres])) == 1
        ):
            logger.warning(
                f"Separated segments {basename + str(i - 1)} and {segid}, despite continuous resids, due to lack of bonding."
            )

        # Add the new segment ID to all fields the user specified
        for f in fields:
            if f != "chain":
                mol.__dict__[f][segres] = segid  # Assign the segid to the correct atoms
            else:
                mol.__dict__[f][segres] = available_chains[i % len(available_chains)]
        if _logger:
            logger.info(
                f"Created segment {segid} between resid {mol.resid[segres].min()} and {mol.resid[segres].max()}."
            )
        prevsegres = segres  # Store old segment atom indexes for the warning about continuous resids

    return mol


import unittest


class _TestPreparation(unittest.TestCase):
    def test_autoSegment(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from os import path

        p = Molecule(path.join(home(dataDir="test-autosegment"), "4dkl.pdb"))
        p.filter("(chain B and protein) or water")
        p = autoSegment(p, "protein", "P")
        m = Molecule(path.join(home(dataDir="test-autosegment"), "membrane.pdb"))
        print(np.unique(m.get("segid")))

        mol = Molecule(path.join(home(dataDir="test-autosegment"), "1ITG_clean.pdb"))
        ref = Molecule(path.join(home(dataDir="test-autosegment"), "1ITG.pdb"))
        mol = autoSegment(mol, sel="protein")
        assert np.all(mol.segid == ref.segid)

        mol = Molecule(path.join(home(dataDir="test-autosegment"), "3PTB_clean.pdb"))
        ref = Molecule(path.join(home(dataDir="test-autosegment"), "3PTB.pdb"))
        mol = autoSegment(mol, sel="protein")
        assert np.all(mol.segid == ref.segid)

    def test_autoSegment2(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from os import path

        mol = Molecule(path.join(home(dataDir="test-autosegment"), "3segments.pdb"))
        smol1 = autoSegment2(mol, sel="nucleic or protein or resname ACE NME")
        smol2 = autoSegment2(mol)
        assert np.array_equal(smol1.segid, smol2.segid)
        vals, counts = np.unique(smol1.segid, return_counts=True)
        assert len(vals) == 3
        assert np.array_equal(counts, np.array([331, 172, 1213]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
