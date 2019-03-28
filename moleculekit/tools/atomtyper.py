import os
import string
from tempfile import NamedTemporaryFile
import numpy as np
from moleculekit.tools.autosegment import autoSegment2
from moleculekit.molecule import Molecule
from moleculekit.writers import _deduce_PDB_atom_name, checkTruncations
from moleculekit.util import ensurelist
import logging


logger = logging.getLogger(__name__)


def getPDBQTAtomType(atype, aidx, mol, aromaticNitrogen=False):
    tmptype = ''
    # carbons
    if atype == 'Car':
        tmptype = 'A'
    elif atype.startswith('C'):
        tmptype = 'C'
    # nitrogens
    if atype.startswith('N'):
        tmptype = 'N'
        if atype in ['Nam', 'Npl', 'Ng+']:
            bs, bo = np.where(mol.bonds == aidx)
            if len(bs) == 2:
                tmptype += 'A'
        elif atype == 'Nar':
            # if mol.resname[aidx] == 'HIE':
            bs, bo = np.where(mol.bonds == aidx)
            if len(bs) == 2:
                tmptype += 'a' if aromaticNitrogen else 'A'
            else:
                tmptype += 'n' if aromaticNitrogen else ''
        elif atype[-1] != '+':
            tmptype += 'A'
    # elif atype.startswith('N'):
    #   print(atype, aidx)
    #  tmptype = 'NA'
    # oxygens
    if atype.startswith('O'):
        tmptype = 'OA'
    # sulfurs
    if atype.startswith('S'):
        tmptype = 'S'
        if atype not in ['Sox', 'Sac']:
            tmptype += 'A'

    # hydrogens
    if atype.startswith('H'):
        tmptype = 'H'
        # print(aidx)
        # print(np.where(mol.bonds == aidx))
        bond = np.where(mol.bonds == aidx)[0][0]
        oidx = [a for a in mol.bonds[bond] if a != aidx][0]
        if mol.element[oidx] not in ['C', 'A']:
            tmptype += 'D'
    if tmptype == '':
        tmptype = atype[0]

    return tmptype


def getProperties(mol):
    try:
        import pybel
    except ImportError:
        raise ImportError('Could not import openbabel. The atomtyper requires this dependency so please install it with `conda install openbabel -c acellera`')
    name = NamedTemporaryFile(suffix='.pdb').name
    mol.write(name)
    mpybel = next(pybel.readfile('pdb', name))
    # print(name)
    residues = pybel.ob.OBResidueIter(mpybel.OBMol)
    atoms = [[r.GetName(), r.GetNum(), r.GetAtomID(at), at.GetType(), round(at.GetPartialCharge(), 3)]
             for r in residues
             for at in pybel.ob.OBResidueAtomIter(r)]
    return atoms

def prepareProteinForAtomtyping(mol, guessBonds=True, protonate=True, pH=7, segment=True, verbose=True):
    """ Prepares a Molecule object for atom typing.

    Parameters
    ----------
    mol : Molecule object
        The protein to prepare
    guessBonds : bool
        Drops the bonds in the molecule and guesses them from scratch
    protonate : bool
        Protonates the protein for the given pH and optimizes hydrogen networks
    pH : float
        The pH for protonation
    segment : bool
        Automatically guesses the segments of a protein by using the guessed bonds
    verbose : bool
        Set to False to turn of the printing

    Returns
    -------
    mol : Molecule object
        The prepared Molecule
    """
    mol = mol.copy()
    protsel = mol.atomselect('protein')

    if not np.any(protsel):
        raise RuntimeError('No protein atoms found in Molecule')

    if np.any(~protsel):
        resnames = np.unique(mol.resname[~protsel])
        raise RuntimeError('Found non-protein atoms with resnames {} in the Molecule. Please make sure to only pass protein atoms.'.format(resnames))

    if protonate:
        from moleculekit.tools.preparation import proteinPrepare
        mol = proteinPrepare(mol, pH=pH, verbose=verbose, _loggerLevel='INFO' if verbose else 'ERROR')

    if guessBonds:
        mol.bonds = mol._guessBonds()

    if segment:
        from moleculekit.tools.autosegment import autoSegment2
        mol = autoSegment2(mol, fields=('segid', 'chain'), _logger=verbose) 
    return mol


def atomtypingValidityChecks(mol):
    logger.info('Checking validity of Molecule before atomtyping. ' \
                'If it gives incorrect results or to improve performance disable it with validitychecks=False. ' \
                'Most of these checks can be passed by using the moleculekit.atomtyper.prepareProteinForAtomtyping function. ' \
                'But make sure you understand what you are doing.')
    protsel = mol.atomselect('protein')

    if not np.any(protsel):
        raise RuntimeError('No protein atoms found in Molecule')

    if np.any(~protsel):
        resnames = np.unique(mol.resname[~protsel])
        raise RuntimeError('Found non-protein atoms with resnames {} in the Molecule. Please make sure to only pass protein atoms.'.format(resnames))

    if mol.bonds.shape[0] < mol.numAtoms:
        raise ValueError('The protein has less bonds than atoms. This seems incorrect. Assign them with `mol.bonds = mol._getBonds()`')

    if np.all(mol.segid == '') or np.all(mol.chain == ''):
        raise RuntimeError('Please assign segments to the segid and chain fields of the molecule using autoSegment2')

    from moleculekit.tools.autosegment import autoSegment2
    mm = mol.copy()
    mm.segid[:] = ''  # Set segid and chain to '' to avoid name clashes in autoSegment2
    mm.chain[:] = ''
    refmol = autoSegment2(mm, fields=('chain', 'segid'), _logger=False)
    numsegsref = len(np.unique(refmol.segid))
    numsegs = len(np.unique(mol.segid))
    if numsegs != numsegsref:
        raise RuntimeError('The molecule contains {} segments while we predict {}. Make sure you used autoSegment2 on the protein'.format(numsegs, numsegsref))

    if not np.any(mol.element == 'H'):
        raise RuntimeError('No hydrogens found in the Molecule. Make sure to use proteinPrepare before passing it to voxelization. Also you might need to recalculate the bonds after this.')


def getPDBQTAtomTypesAndCharges(mol, aromaticNitrogen=False, validitychecks=True):
    if validitychecks:
        atomtypingValidityChecks(mol)

    atomsProp = getProperties(mol)
    for n, a in enumerate(atomsProp):
        if a[0] == 'HIP':
            if a[2].strip().startswith('C') and a[2].strip() not in ['CA', 'C', 'CB']:
                a[3] = 'Car'

            #print(n, a)
    charges = ['{0:.3f}'.format(a[-1]) for a in atomsProp]
    pdbqtATypes = [getPDBQTAtomType(a[3], n, mol, aromaticNitrogen)
                   for n, a in enumerate(atomsProp)]

    return np.array(pdbqtATypes, dtype='O'), np.array(charges, dtype='float32')


def _getHydrophobic(atypes):
    return atypes == 'C'


def _getAromatic(atypes):
    return (atypes == 'A') | (atypes == 'Na') | (atypes == 'Nn')


def _getAcceptor(atypes):
    return (atypes == 'OA') | (atypes == 'NA') | (atypes == 'SA') | (atypes == 'Na')


def _getDonors(atypes, bonds):
    donors = np.zeros(len(atypes), dtype=bool)
    hydrogens = np.where((atypes == 'HD') | (atypes == 'HS'))[0]
    for h in hydrogens:
        partners = bonds[bonds[:, 0] == h, 1]
        partners = np.hstack((partners, bonds[bonds[:, 1] == h, 0]))
        for p in partners:
            if atypes[p][0] in ('N', 'O', 'S'):
                donors[p] = True
    return donors


def _getPosIonizable(mol):
    # arginine, lysine and histidine
    posIonizables = np.zeros(mol.numAtoms, dtype=bool)

    # ARG
    n_idxs = np.where(((mol.resname == 'ARG') | (mol.resname == 'AR0')) & (
        mol.atomtype == 'N') & (mol.name != 'N'))
    allc_idxs = np.where((mol.resname == 'ARG') & (
        mol.atomtype == 'C') & (mol.name != 'C'))[0]
    c_idxs = []
    for c in allc_idxs:
        bs = np.where(mol.bonds == c)[0]
        if len(bs) == 3:
            c_idxs.append(c)

    aidxs = n_idxs[0].tolist() + c_idxs

    # LYS
    n_idxs = np.where(((mol.resname == 'LYS') | (mol.resname == 'LYN')) & (
        mol.atomtype == 'N') & (mol.name != 'N'))
    aidxs += n_idxs[0].tolist()

    # HIS, HID, HIE, HIP, HSD, HSE
    n_idxs = np.where(((mol.resname == 'HIS') | (mol.resname == 'HID') |
                       (mol.resname == 'HIE') | (mol.resname == 'HIP') |
                       (mol.resname == 'HSE') | (mol.resname == 'HSD') |
                       (mol.resname == 'HSP')) &
                      ((mol.atomtype == 'N') | (mol.atomtype == 'NA') | (mol.atomtype == 'Nn') | (mol.atomtype == 'Na')) &
                      (mol.name != 'N'))

    c_idxs = np.where(((mol.resname == 'HIS') | (mol.resname == 'HID') |
                       (mol.resname == 'HIE') | (mol.resname == 'HIP') |
                       (mol.resname == 'HSE') | (mol.resname == 'HSD') |
                       (mol.resname == 'HSP')) &
                      (mol.atomtype == 'A'))

    aidxs += n_idxs[0].tolist() + c_idxs[0].tolist()

    posIonizables[aidxs] = 1

    return posIonizables


def _getNegIonizable(mol):
    # aspartic and glutamate
    negIonizables = np.zeros(mol.numAtoms, dtype=bool)

    # ASP
    o_idxs = np.where(((mol.resname == 'ASP') | (mol.resname == 'ASH')) &
                      (mol.atomtype == 'OA') & (mol.name != 'O'))
    allc_idxs = np.where(((mol.resname == 'ASP') | (mol.resname == 'ASH')) &
                         (mol.atomtype == 'C') & (mol.name != 'C'))[0]
    c_idxs = []
    for c in allc_idxs:
        bs = np.where(mol.bonds == c)[0]
        if len(bs) == 3:
            c_idxs.append(c)
    aidxs = o_idxs[0].tolist() + c_idxs

    # Glutamate
    o_idxs = np.where(((mol.resname == 'GLU') | (mol.resname == 'GLH')) &
                      (mol.atomtype == 'OA') & (mol.name != 'O'))

    allc_idxs = np.where(((mol.resname == 'GLU') | (mol.resname == 'GLH')) &
                         (mol.atomtype == 'C') & (mol.name != 'C'))[0]
    c_idxs = []
    for c in allc_idxs:
        bs = np.where(mol.bonds == c)[0]
        if len(bs) == 3:
            c_idxs.append(c)
    aidxs += o_idxs[0].tolist() + c_idxs

    negIonizables[aidxs] = 1

    return negIonizables


def _getOccupancy(elements):
    return np.array(elements) != 'H'


def _getMetals(atypes):
    return (atypes == 'MG') | (atypes == 'ZN') | (atypes == 'MN') | \
           (atypes == 'CA') | (atypes == 'FE') | (atypes == 'HG') | \
           (atypes == 'CD') | (atypes == 'NI') | (atypes == 'CO') | \
           (atypes == 'CU') | (atypes == 'K') | (atypes == 'LI') | \
           (atypes == 'Mg') | (atypes == 'Zn') | (atypes == 'Mn') | \
           (atypes == 'Ca') | (atypes == 'Fe') | (atypes == 'Hg') | \
           (atypes == 'Cd') | (atypes == 'Ni') | (atypes == 'Co') | \
           (atypes == 'Cu') | (atypes == 'Li')


def getFeatures(mol):
    atypes = mol.atomtype
    elements = [el[0] for el in atypes]

    hydr = _getHydrophobic(atypes)
    arom = _getAromatic(atypes)
    acc = _getAcceptor(atypes)
    don = _getDonors(atypes, mol.bonds)
    pos = _getPosIonizable(mol)
    neg = _getNegIonizable(mol)
    metals = _getMetals(atypes)
    occ = _getOccupancy(elements)

    return np.vstack((hydr, arom, acc, don, pos, neg, metals, occ)).T.copy()


def parallel(func, listobj, n_cpus=-1, *args):
    from tqdm import tqdm
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_cpus)(delayed(func)(ob, *args)
                                      for ob in tqdm(listobj))
    return results