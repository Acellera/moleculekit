import string
import numpy as np
import logging


logger = logging.getLogger(__name__)


def _getChainAlphabet(mode):
    if mode == 'numeric':
        chain_alphabet = list(string.digits)
    elif mode == 'alphabetic':
        chain_alphabet = list(string.ascii_uppercase + string.ascii_lowercase)
    elif mode == 'alphanumeric':
        chain_alphabet = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
    else:
        raise RuntimeError('Invalid mode {}. Use \'numeric\', \'alphabetic\' or \'alphanumeric\''.format(mode))
    return chain_alphabet


def autoSegment(mol, sel='all', basename='P', spatial=True, spatialgap=4.0, field="segid", mode='alphanumeric', _logger=True):
    """ Detects resid gaps in a selection and assigns incrementing segid to each fragment

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
        Only considers a discontinuity in resid as a gap of the CA atoms have distance more than `spatialgap` Angstrom
    spatialgap : float
        The size of a spatial gap which validates a discontinuity (A)
    field : str
        Field to fix. Can be "segid" (default), "chain", or "both"
    mode : str
        If set to 'numeric' it will use numbers for segment IDs.
        If set to 'alphabetic' it will use letters for segment IDs.
        If set to 'alphanumeric' it will use both numbers and letters for segment IDs.

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A new Molecule object with modified segids

    Example
    -------
    >>> newmol = autoSegment(mol,'chain B','P')
    """
    from moleculekit.util import sequenceID
    mol = mol.copy()

    idx = mol.atomselect(sel, indexes=True)
    rid = mol.resid[idx].copy()
    residiff = np.diff(rid)
    gappos = np.where((residiff != 1) & (residiff != 0))[0]  # Points to the index before the gap!

    # Letters to be used for chains, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    used_chains = set(mol.chain)
    chain_alphabet = _getChainAlphabet(mode)

    available_chains = [x for x in chain_alphabet if x not in used_chains]

    idxstartseg = [idx[0]] + idx[gappos + 1].tolist()
    idxendseg = idx[gappos].tolist() + [idx[-1]]

    mol.set('segid', basename, sel)

    if len(gappos) == 0:
        mol.set('segid', basename+chain_alphabet[0], sel)
        return mol

    if spatial:
        residbackup = mol.resid.copy()
        mol.set('resid', sequenceID(mol.resid))  # Assigning unique resids to be able to do the distance selection

        todelete = []
        i = 0
        for s, e in zip(idxstartseg[1:], idxendseg[:-1]):
            # Get the carbon alphas of both residues  ('coords', sel='resid "{}" "{}" and name CA'.format(mol.resid[e], mol.resid[s]))
            ca1coor = mol.coords[(mol.resid == mol.resid[e]) & (mol.name == 'CA')]
            ca2coor = mol.coords[(mol.resid == mol.resid[s]) & (mol.name == 'CA')]
            if len(ca1coor) and len(ca2coor):
                dist = np.sqrt(np.sum((ca1coor.squeeze() - ca2coor.squeeze()) ** 2))
                if dist < spatialgap:
                    todelete.append(i)
            i += 1
        todelete = np.array(todelete, dtype=int)
        # Join the non-real gaps into segments
        idxstartseg = np.delete(idxstartseg, todelete+1)
        idxendseg = np.delete(idxendseg, todelete)

        mol.set('resid', residbackup)  # Restoring the original resids

    i = 0
    for s, e in zip(idxstartseg, idxendseg):
        # Fixup segid
        if field in ['segid', 'both']:
            newsegid = basename + str(i)
            if np.any(mol.segid == newsegid):
                raise RuntimeError('Segid {} already exists in the molecule. Please choose different prefix.'.format(newsegid))
            if _logger: logger.info('Created segment {} between resid {} and {}.'.format(newsegid, mol.resid[s], mol.resid[e]))
            mol.segid[s:e+1] = newsegid
        # Fixup chain
        if field in ['chain', 'both']:
            newchainid = available_chains[i]
            if _logger: logger.info('Set chain {} between resid {} and {}.'.format(newchainid, mol.resid[s], mol.resid[e]))
            mol.chain[s:e+1] = newchainid

        i += 1

    return mol


def autoSegment2(mol, sel='(protein or resname ACE NME)', basename='P', fields=('segid',), residgaps=False, residgaptol=1, chaingaps=True, mode='alphanumeric', _logger=True):
    """ Detects bonded segments in a selection and assigns incrementing segid to each segment

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
    mode : str
        If set to 'numeric' it will use numbers for segment IDs.
        If set to 'alphabetic' it will use letters for segment IDs.
        If set to 'alphanumeric' it will use both numbers and letters for segment IDs.

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

    sel += ' and backbone or (resname NME ACE and name N C O CH3)'  # Looking for bonds only over the backbone of the protein
    idx = mol.atomselect(sel, indexes=True)  # Keep the original atom indexes to map from submol to mol
    submol = mol.copy()  # We filter out everything not on the backbone to calculate only those bonds
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
                logger.warning('Bonds found between resid gaps: resid {} and {}'.format(submol.resid[bonds[i, 0]],
                                                                                        submol.resid[bonds[i, 1]]))
    if chaingaps:
        # Remove bonds between residues without same chain
        bondsamechain = submol.chain[bonds[:, 0]] == submol.chain[bonds[:, 1]]
        bonds = bonds[bondsamechain, :]
    else:
        # Warning about bonds bonding different chains
        bondsamechain = submol.chain[bonds[:, 0]] == submol.chain[bonds[:, 1]]
        if _logger and np.any(bondsamechain == False):
            for i in np.where(bondsamechain == False)[0]:
                logger.warning('Bonds found between chain gaps: resid {}/{} and {}/{}'.format(submol.resid[bonds[i, 0]],
                                                                                              submol.chain[bonds[i, 0]],
                                                                                              submol.resid[bonds[i, 1]],
                                                                                              submol.chain[bonds[i, 1]]
                                                                                              ))

    # Calculate connected components using the bonds
    sparsemat = csr_matrix((np.ones(bonds.shape[0] * 2),  # Values
                            (np.hstack((bonds[:, 0], bonds[:, 1])),  # Rows
                             np.hstack((bonds[:, 1], bonds[:, 0])))), shape=[submol.numAtoms, submol.numAtoms])  # Columns
    numcomp, compidx = connected_components(sparsemat, directed=False)

    # Letters to be used for chains, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    used_chains = set(mol.chain)
    chain_alphabet = _getChainAlphabet(mode)
    available_chains = [x for x in chain_alphabet if x not in used_chains]

    mol = mol.copy()
    prevsegres = None
    for i in range(numcomp):  # For each connected component / segment
        segid = basename + str(i)
        backboneSegIdx = idx[compidx == i]  # The backbone atoms of the segment
        segres = mol.atomselect('same residue as index {}'.format(' '.join(map(str, backboneSegIdx)))) # Get whole residues

        # Warning about separating segments with continuous resids
        if _logger and i > 0 and (np.min(mol.resid[segres]) - np.max(mol.resid[prevsegres])) == 1:
            logger.warning('Separated segments {} and {}, despite continuous resids, due to lack of bonding.'.format(
                            basename + str(i-1), segid))

        # Add the new segment ID to all fields the user specified
        for f in fields:
            if f != 'chain':
                if np.any(mol.__dict__[f] == segid):
                    raise RuntimeError('Segid {} already exists in the molecule. Please choose different prefix.'.format(segid))
                mol.__dict__[f][segres] = segid  # Assign the segid to the correct atoms
            else:
                mol.__dict__[f][segres] = available_chains[i % len(available_chains)]
        if _logger:
            logger.info('Created segment {} between resid {} and {}.'.format(segid, np.min(mol.resid[segres]),
                                                                            np.max(mol.resid[segres])))
        prevsegres = segres  # Store old segment atom indexes for the warning about continuous resids

    return mol

import unittest
class _TestPreparation(unittest.TestCase):
    def test_autoSegment(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from os import path

        p = Molecule(path.join(home(dataDir='test-autosegment'), '4dkl.pdb'))
        p.filter('(chain B and protein) or water')
        p = autoSegment(p, 'protein', 'P')
        m = Molecule(path.join(home(dataDir='test-autosegment'), 'membrane.pdb'))
        print(np.unique(m.get('segid')))

        mol = Molecule(path.join(home(dataDir='test-autosegment'), '1ITG_clean.pdb'))
        ref = Molecule(path.join(home(dataDir='test-autosegment'), '1ITG.pdb'))
        mol = autoSegment(mol, sel='protein')
        assert np.all(mol.segid == ref.segid)

        mol = Molecule(path.join(home(dataDir='test-autosegment'), '3PTB_clean.pdb'))
        ref = Molecule(path.join(home(dataDir='test-autosegment'), '3PTB.pdb'))
        mol = autoSegment(mol, sel='protein')
        assert np.all(mol.segid == ref.segid)

if __name__ == "__main__":
    unittest.main(verbosity=2)