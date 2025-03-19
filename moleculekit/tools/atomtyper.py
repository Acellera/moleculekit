# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import logging


logger = logging.getLogger(__name__)

metal_atypes = (
    "MG",
    "ZN",
    "MN",
    "CA",
    "FE",
    "HG",
    "CD",
    "NI",
    "CO",
    "CU",
    "K",
    "LI",
    "Mg",
    "Zn",
    "Mn",
    "Ca",
    "Fe",
    "Hg",
    "Cd",
    "Ni",
    "Co",
    "Cu",
    "Li",
)


def getPDBQTAtomType(atype, aidx, mol, aromaticNitrogen=False):
    if atype in metal_atypes:  # Save metals from treated as protein atoms
        return atype

    tmptype = ""
    # carbons
    if atype == "Car":
        tmptype = "A"
    elif atype.startswith("C"):
        tmptype = "C"
    # nitrogens
    if atype.startswith("N"):
        tmptype = "N"
        if atype in ["Nam", "Npl", "Ng+"]:
            bs, bo = np.where(mol.bonds == aidx)
            if len(bs) == 2:
                tmptype += "A"
        elif atype == "Nar":
            # if mol.resname[aidx] == 'HIE':
            bs, bo = np.where(mol.bonds == aidx)
            if len(bs) == 2:
                tmptype += "a" if aromaticNitrogen else "A"
            else:
                tmptype += "n" if aromaticNitrogen else ""
        elif atype[-1] != "+":
            tmptype += "A"
    # elif atype.startswith('N'):
    #   print(atype, aidx)
    #  tmptype = 'NA'
    # oxygens
    if atype.startswith("O"):
        tmptype = "OA"
    # sulfurs
    if atype.startswith("S"):
        tmptype = "S"
        if atype not in ["Sox", "Sac"]:
            tmptype += "A"

    # hydrogens
    if atype.startswith("H"):
        tmptype = "H"
        # print(aidx)
        # print(np.where(mol.bonds == aidx))
        try:
            bond = np.where(mol.bonds == aidx)[0][0]
        except Exception:
            raise RuntimeError(
                f"Could not atomtype hydrogen atom with index {aidx} due to no bonding partners."
            )

        oidx = [a for a in mol.bonds[bond] if a != aidx][0]
        if mol.element[oidx] not in ["C", "A"]:
            tmptype += "D"

    if tmptype == "":
        tmptype = atype[0]

    return tmptype


def prepareProteinForAtomtyping(
    mol, guessBonds=True, protonate=True, pH=7.4, segment=True, verbose=True
):
    """Prepares a Molecule object for atom typing.

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
    from moleculekit.tools.autosegment import autoSegment2
    from moleculekit.util import sequenceID

    mol = mol.copy()
    if (
        guessBonds
    ):  # Need to guess bonds at the start for atom selection and for autoSegment
        mol.bondtype = np.array([], dtype=object)
        mol.bonds = mol._guessBonds()

    protsel = mol.atomselect("protein")
    metalsel = mol.atomselect(f"element {' '.join(metal_atypes)}")
    watersel = mol.atomselect("water")
    notallowed = ~(protsel | metalsel | watersel)

    if not np.any(protsel):
        raise RuntimeError("No protein atoms found in Molecule")

    if np.any(notallowed):
        resnames = np.unique(mol.resname[notallowed])
        raise RuntimeError(
            "Found atoms with resnames {} in the Molecule which can cause issues with the voxelization. Please make sure to only pass protein atoms and metals.".format(
                resnames
            )
        )

    protmol = mol.copy()
    protmol.filter(protsel, _logger=False)
    metalmol = mol.copy()
    metalmol.filter(metalsel, _logger=False)
    watermol = mol.copy()
    watermol.filter(watersel, _logger=False)

    if protonate:
        from moleculekit.tools.preparation import systemPrepare

        if np.all(protmol.segid == "") and np.all(protmol.chain == ""):
            protmol = autoSegment2(
                protmol, fields=("segid", "chain"), basename="K", _logger=verbose
            )  # We need segments to prepare the protein
        protmol = systemPrepare(
            protmol,
            pH=pH,
            verbose=verbose,
            _logger_level="INFO" if verbose else "ERROR",
        )

    if guessBonds:
        protmol.bonds = protmol._guessBonds()
        # TODO: Should we remove bonds between metals and protein?

    if segment:
        protmol = autoSegment2(
            protmol, fields=("segid", "chain"), _logger=verbose
        )  # Reassign segments after preparation

        # Assign separate segment to the metals just in case pybel takes that into account
        if np.any(protmol.chain == "Z") or np.any(protmol.segid == "ME"):
            raise AssertionError(
                "Report this issue on the moleculekit github issue tracker. Too many chains in the protein."
            )
        metalmol.segid[:] = "ME"
        metalmol.chain[:] = "Z"
        metalmol.resid[:] = (
            np.arange(0, 2 * metalmol.numAtoms, 2) + protmol.resid.max() + 1
        )  # Just in case, let's put a residue gap between the metals so that they are considered separate chains no matter what happens

        if watermol.numAtoms != 0:
            if np.any(protmol.chain == "W") or np.any(protmol.segid == "WX"):
                raise AssertionError(
                    "Report this issue on the moleculekit github issue tracker. Too many chains in the protein."
                )
            watermol.resid[:] = sequenceID(
                (watermol.resid, watermol.segid, watermol.chain), step=2
            )
            watermol.segid[:] = "WX"
            watermol.chain[:] = "W"

    mol = protmol.copy()
    mol.append(metalmol)
    mol.append(watermol)
    return mol


def atomtypingValidityChecks(mol):
    logger.info(
        "Checking validity of Molecule before atomtyping. "
        "If it gives incorrect results or to improve performance disable it with validitychecks=False. "
        "Most of these checks can be passed by using the moleculekit.atomtyper.prepareProteinForAtomtyping function. "
        "But make sure you understand what you are doing."
    )
    protsel = mol.atomselect("protein")
    metals = mol.atomselect(f"element {' '.join(metal_atypes)}")
    notallowed = ~(protsel | metals)

    if not np.any(protsel):
        raise RuntimeError("No protein atoms found in Molecule")

    if np.any(notallowed):
        resnames = np.unique(mol.resname[notallowed])
        raise RuntimeError(
            "Found atoms with resnames {} in the Molecule which can cause issues with the voxelization. Please make sure to only pass protein atoms and metals.".format(
                resnames
            )
        )

    if mol.bonds.shape[0] < (mol.numAtoms - 1):
        raise ValueError(
            "The protein has less bonds than (number of atoms - 1). This seems incorrect. You can assign bonds with `mol.bonds = mol._getBonds()`"
        )

    from moleculekit.molecule import calculateUniqueBonds

    uqbonds, _ = calculateUniqueBonds(mol.bonds, mol.bondtype)
    if uqbonds.shape[0] != mol.bonds.shape[0]:
        raise RuntimeError(
            "The protein has duplicate bond information. This will mess up atom typing. Please keep only unique bonds in the molecule. If you want you can use moleculekit.molecule.calculateUniqueBonds for this."
        )

    if np.all(mol.segid == "") or np.all(mol.chain == ""):
        raise RuntimeError(
            "Please assign segments to the segid and chain fields of the molecule using autoSegment2"
        )

    from moleculekit.tools.autosegment import autoSegment2

    mm = mol.copy()
    mm.segid[:] = ""  # Set segid and chain to '' to avoid name clashes in autoSegment2
    mm.chain[:] = ""
    refmol = autoSegment2(mm, fields=("chain", "segid"), _logger=False)
    numsegsref = len(np.unique(refmol.segid))
    numsegs = len(np.unique(mol.segid))
    if numsegs != numsegsref:
        raise RuntimeError(
            "The molecule contains {} segments while we predict {}. Make sure you used autoSegment2 on the protein".format(
                numsegs, numsegsref
            )
        )

    if not np.any(mol.element == "H"):
        raise RuntimeError(
            "No hydrogens found in the Molecule. Make sure to use systemPrepare before passing it to voxelization. Also you might need to recalculate the bonds after this."
        )


def getPDBQTAtomTypesAndCharges(mol, aromaticNitrogen=False, validitychecks=True):
    from moleculekit.tools.obabel_tools import getOpenBabelProperties

    if validitychecks:
        atomtypingValidityChecks(mol)

    atomsProp = getOpenBabelProperties(mol)
    pdbqtATypes = [""] * mol.numAtoms
    charges = [np.nan] * mol.numAtoms

    for idx, resname, resid, name, attype, charge in atomsProp:
        if resname == "HIP":
            if name.strip().startswith("C") and name.strip() not in ["CA", "C", "CB"]:
                attype = "Car"
        charges[idx] = f"{charge:.3f}"
        pdbqtATypes[idx] = getPDBQTAtomType(attype, idx, mol, aromaticNitrogen)

    return np.array(pdbqtATypes, dtype="O"), np.array(charges, dtype="float32")


def _getHydrophobic(atypes):
    return atypes == "C"


def _getAromatic(atypes):
    return (atypes == "A") | (atypes == "Na") | (atypes == "Nn")


def _getAcceptor(atypes):
    return (atypes == "OA") | (atypes == "NA") | (atypes == "SA") | (atypes == "Na")


def _getDonors(atypes, bonds):
    donors = np.zeros(len(atypes), dtype=bool)
    hydrogens = np.where((atypes == "HD") | (atypes == "HS"))[0]
    for h in hydrogens:
        partners = bonds[bonds[:, 0] == h, 1]
        partners = np.hstack((partners, bonds[bonds[:, 1] == h, 0]))
        for p in partners:
            if atypes[p][0] in ("N", "O", "S"):
                donors[p] = True
    return donors


def _getPosIonizable(mol):
    # arginine, lysine and histidine
    posIonizables = np.zeros(mol.numAtoms, dtype=bool)

    # ARG
    n_idxs = np.where(
        ((mol.resname == "ARG") | (mol.resname == "AR0"))
        & (mol.atomtype == "N")
        & (mol.name != "N")
    )
    allc_idxs = np.where(
        (mol.resname == "ARG") & (mol.atomtype == "C") & (mol.name != "C")
    )[0]
    c_idxs = []
    for c in allc_idxs:
        bs = np.where(mol.bonds == c)[0]
        if len(bs) == 3:
            c_idxs.append(c)

    aidxs = n_idxs[0].tolist() + c_idxs

    # LYS
    n_idxs = np.where(
        ((mol.resname == "LYS") | (mol.resname == "LYN"))
        & (mol.atomtype == "N")
        & (mol.name != "N")
    )
    aidxs += n_idxs[0].tolist()

    # HIS, HID, HIE, HIP, HSD, HSE
    n_idxs = np.where(
        (
            (mol.resname == "HIS")
            | (mol.resname == "HID")
            | (mol.resname == "HIE")
            | (mol.resname == "HIP")
            | (mol.resname == "HSE")
            | (mol.resname == "HSD")
            | (mol.resname == "HSP")
        )
        & (
            (mol.atomtype == "N")
            | (mol.atomtype == "NA")
            | (mol.atomtype == "Nn")
            | (mol.atomtype == "Na")
        )
        & (mol.name != "N")
    )

    c_idxs = np.where(
        (
            (mol.resname == "HIS")
            | (mol.resname == "HID")
            | (mol.resname == "HIE")
            | (mol.resname == "HIP")
            | (mol.resname == "HSE")
            | (mol.resname == "HSD")
            | (mol.resname == "HSP")
        )
        & (mol.atomtype == "A")
    )

    aidxs += n_idxs[0].tolist() + c_idxs[0].tolist()

    posIonizables[aidxs] = 1

    return posIonizables


def _getNegIonizable(mol):
    # aspartic and glutamate
    negIonizables = np.zeros(mol.numAtoms, dtype=bool)

    # ASP
    o_idxs = np.where(
        ((mol.resname == "ASP") | (mol.resname == "ASH"))
        & (mol.atomtype == "OA")
        & (mol.name != "O")
    )
    allc_idxs = np.where(
        ((mol.resname == "ASP") | (mol.resname == "ASH"))
        & (mol.atomtype == "C")
        & (mol.name != "C")
    )[0]
    c_idxs = []
    for c in allc_idxs:
        bs = np.where(mol.bonds == c)[0]
        if len(bs) == 3:
            c_idxs.append(c)
    aidxs = o_idxs[0].tolist() + c_idxs

    # Glutamate
    o_idxs = np.where(
        ((mol.resname == "GLU") | (mol.resname == "GLH"))
        & (mol.atomtype == "OA")
        & (mol.name != "O")
    )

    allc_idxs = np.where(
        ((mol.resname == "GLU") | (mol.resname == "GLH"))
        & (mol.atomtype == "C")
        & (mol.name != "C")
    )[0]
    c_idxs = []
    for c in allc_idxs:
        bs = np.where(mol.bonds == c)[0]
        if len(bs) == 3:
            c_idxs.append(c)
    aidxs += o_idxs[0].tolist() + c_idxs

    negIonizables[aidxs] = 1

    return negIonizables


def _getOccupancy(elements):
    return np.array(elements) != "H"


def _getMetals(atypes):
    return np.isin(atypes, metal_atypes)


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

    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "Please install joblib to use the parallel function with `conda install joblib`."
        )

    results = Parallel(n_jobs=n_cpus)(delayed(func)(ob, *args) for ob in tqdm(listobj))
    return results
