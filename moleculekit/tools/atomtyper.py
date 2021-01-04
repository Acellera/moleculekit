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
        except Exception as e:
            raise RuntimeError(
                f"Could not atomtype hydrogen atom with index {aidx} due to no bonding partners."
            )

        oidx = [a for a in mol.bonds[bond] if a != aidx][0]
        if mol.element[oidx] not in ["C", "A"]:
            tmptype += "D"

    if tmptype == "":
        tmptype = atype[0]

    return tmptype


def getProperties(mol):
    try:
        from openbabel import pybel
    except ImportError:
        raise ImportError(
            "Could not import openbabel. The atomtyper requires this dependency so please install it with `conda install openbabel -c conda-forge`"
        )

    name = NamedTemporaryFile(suffix=".pdb").name
    mol.write(name)
    mpybel = next(pybel.readfile("pdb", name))

    # print(name)
    residues = pybel.ob.OBResidueIter(mpybel.OBMol)
    atoms = [
        [
            r.GetName(),
            r.GetNum(),
            r.GetAtomID(at),
            at.GetType(),
            round(at.GetPartialCharge(), 3),
        ]
        for r in residues
        for at in pybel.ob.OBResidueAtomIter(r)
    ]

    os.remove(name)

    return atoms


def prepareProteinForAtomtyping(
    mol, guessBonds=True, protonate=True, pH=7, segment=True, verbose=True
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
    metalsel = mol.atomselect("element {}".format(" ".join(metal_atypes)))
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
        from moleculekit.tools.preparation import proteinPrepare

        if np.all(protmol.segid == "") and np.all(protmol.chain == ""):
            protmol = autoSegment2(
                protmol, fields=("segid", "chain"), basename="K", _logger=verbose
            )  # We need segments to prepare the protein
        protmol = proteinPrepare(
            protmol, pH=pH, verbose=verbose, _loggerLevel="INFO" if verbose else "ERROR"
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
    metals = mol.atomselect("element {}".format(" ".join(metal_atypes)))
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
            "No hydrogens found in the Molecule. Make sure to use proteinPrepare before passing it to voxelization. Also you might need to recalculate the bonds after this."
        )


def getPDBQTAtomTypesAndCharges(mol, aromaticNitrogen=False, validitychecks=True):
    if validitychecks:
        atomtypingValidityChecks(mol)

    atomsProp = getProperties(mol)
    for n, a in enumerate(atomsProp):
        if a[0] == "HIP":
            if a[2].strip().startswith("C") and a[2].strip() not in ["CA", "C", "CB"]:
                a[3] = "Car"

            # print(n, a)
    charges = ["{0:.3f}".format(a[-1]) for a in atomsProp]
    pdbqtATypes = [
        getPDBQTAtomType(a[3], n, mol, aromaticNitrogen)
        for n, a in enumerate(atomsProp)
    ]

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
    return np.in1d(atypes, metal_atypes)


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

    results = Parallel(n_jobs=n_cpus)(delayed(func)(ob, *args) for ob in tqdm(listobj))
    return results


def getProteinAtomFeatures(mol):
    from moleculekit.home import home
    from moleculekit.tools.atomtyper_utils import getAtomLabel, getAtomBonds
    from moleculekit.tools.voxeldescriptors import _order
    import networkx as nx
    import yaml

    mol = mol.copy()

    atomtypes_f = os.path.join(
        home(shareDir=True), "atomtyper", "protein_atomtypes.yaml"
    )
    resmap_f = os.path.join(
        home(shareDir=True), "atomtyper", "protein_residue_map.yaml"
    )
    atmmap_f = os.path.join(home(shareDir=True), "atomtyper", "protein_atom_map.yaml")
    with open(atomtypes_f, "r") as f:
        atomtypes = yaml.load(f, Loader=yaml.BaseLoader)
    # Convert all properties to integers for fast indexing
    for resn in atomtypes:
        for atmn in atomtypes[resn]:
            props = atomtypes[resn][atmn]["properties"]
            int_props = np.array([_order.index(prop) for prop in props])
            atomtypes[resn][atmn]["properties"] = int_props
    with open(resmap_f, "r") as f:
        resname_map = yaml.load(f, Loader=yaml.BaseLoader)
    with open(atmmap_f, "r") as f:
        atmlabel_map = yaml.load(f, Loader=yaml.BaseLoader)
    if atmlabel_map is None:
        atmlabel_map = {}
    for resn in atomtypes:
        resname_map[resn] = resn

    atoms_to_compute = mol.element != "H"

    features = np.zeros((mol.numAtoms, len(_order)), dtype=bool)
    # Set all occupancies to True
    features[atoms_to_compute, _order.index("occupancies")] = True

    failed = {"residues": [], "atoms": [], "labels": []}

    # Iterate over residues and find non-existent or try to map them
    failed = fixResidueNames(mol, atoms_to_compute, resname_map, failed)

    # Create bond graph to generate atom labels
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(np.arange(mol.numAtoms))
    bond_graph.add_edges_from(mol.bonds)

    # Create labels for all heavy atoms of the protein
    mol.label = np.array(["" for _ in range(mol.numAtoms)], dtype=object)
    for idx in np.where(atoms_to_compute)[0]:
        bond_elem = mol.element[list(bond_graph.adj[idx])]
        heavy_bonds = sum(bond_elem != "H")
        label = getAtomLabel(mol.name[idx], heavy_bonds, len(bond_elem) - heavy_bonds)
        mol.label[idx] = label

    # Fix some labels which are commonly wrong
    fixAtomLabels(mol, atoms_to_compute, atmlabel_map)

    uqresn = np.unique(mol.resname)
    resnames = np.intersect1d(list(atomtypes.keys()), uqresn)

    # Keep track which atoms were atom-typed and which not
    all_matched = np.zeros(mol.numAtoms, dtype=bool)
    all_matched[~atoms_to_compute] = True  # Ignore the non-compute atoms

    # Perform the atom-typing
    for resn in resnames:
        for label in atomtypes[resn]:
            matched = (mol.resname == resn) & (mol.label == label)
            all_matched[matched] = True

            props = atomtypes[resn][label]["properties"]
            if len(props):
                features[matched, props[:, None]] = True

    # Atom-type the terminals
    for resn in ["NTERM", "CTERM"]:
        for label in atomtypes[resn]:
            matched = (mol.label == label) & ~all_matched
            all_matched[matched] = True

            props = atomtypes[resn][label]["properties"]
            if len(props):
                features[matched, props[:, None]] = True

    # Print and report all atoms which were not atom-typed
    failed["labels"] = np.vstack(
        (
            mol.resname[~all_matched],
            mol.label[~all_matched],
            mol.resid[~all_matched],
            mol.chain[~all_matched],
        )
    ).T
    if failed["labels"].ndim == 1:
        failed["labels"] = failed["labels"][None, :]

    for resn, label in np.vstack(list({tuple(row[:2]) for row in failed["labels"]})):
        atmn, heavy_bonds, h_bonds = label.split("_")
        logger.error(
            f"Could not find residue {resn} atom {atmn} with bonding partners (heavy: {heavy_bonds} hydrogen: {h_bonds}) in the atomtypes library. No features will be calculated for this atom."
        )

    return features, failed


def fixResidueNames(mol, atoms_to_compute, resname_map, failed):
    from moleculekit.tools.atomtyper_utils import getAtomBonds

    for resn in np.unique(mol.resname[atoms_to_compute]):
        if resn not in resname_map:
            if resn in failed["residues"]:
                continue
            logger.error(
                f"No atomtypes defined for residue {resn}. No features will be calculated for this residue."
            )
            failed["residues"].append(resn)
            continue

        res_atoms = mol.resname == resn

        # Get unique residues (resid, chain, segid) which match that residue name
        resid_chain_seg = np.vstack(
            (mol.resid[res_atoms], mol.chain[res_atoms], mol.segid[res_atoms])
        ).T
        if resid_chain_seg.ndim == 1:
            resid_chain_seg = resid_chain_seg[None, :]
        resid_chain_seg = np.vstack(list({tuple(row) for row in resid_chain_seg}))

        potential_residues = resname_map[resn]
        if isinstance(potential_residues, list):  # fancy residue detection by hydrogens
            # Loop over unique residues
            for resid, chain, segid in resid_chain_seg:
                # All atoms of the current residue
                atoms_match = (
                    (mol.resid == int(resid))
                    & (mol.chain == chain)
                    & (mol.segid == segid)
                )
                found = False
                for pot_res in potential_residues:
                    res_name = pot_res["residue_name"]
                    conditions = pot_res["conditions"]

                    # Loop over all conditions which have to be true
                    for condition in conditions:
                        idx = None
                        failed_condition = True
                        if "atom_name" in condition:
                            atom_name = condition["atom_name"]
                            match = atoms_match & (mol.name == atom_name)
                            idx = np.where(match)[0]
                            if len(idx):
                                idx = idx[0]
                            else:
                                idx = None
                        if "h_bonds" in condition and idx is not None:
                            h_bonds = int(condition["h_bonds"])
                            _, h_bonds_atom = getAtomBonds(mol, idx)
                            if h_bonds == h_bonds_atom:
                                failed_condition = False
                        if failed_condition:
                            break

                    if not failed_condition:
                        mol.resname[atoms_match] = res_name
                        logger.info(
                            f"Converted residue {resn}/{resid}/{chain}/{segid} to type {res_name}"
                        )
                        found = True
                        break
                if not found:
                    logger.error(
                        f"Could not match residue {resn}/{resid}/{chain}/{segid} to any potential residue. No features will be calculated for this."
                    )
                    failed["residues"].append((resn, resid, chain, segid))
        else:
            mol.resname[mol.resname == resn] = potential_residues

    return failed


def fixAtomLabels(mol, atoms_to_compute, atmname_map):
    for resn in atmname_map:
        for label in atmname_map[resn]:
            match = (mol.resname == resn) & (mol.label == label)
            mol.label[match] = atmname_map[resn][label]


import unittest


class _TestAtomTyper(unittest.TestCase):
    def test_preparation(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule, mol_equal
        from os import path

        mol = Molecule(path.join(home(dataDir="test-voxeldescriptors"), "1ATL.pdb"))
        ref = Molecule(
            path.join(home(dataDir="test-voxeldescriptors"), "1ATL_prepared.pdb")
        )
        mol2 = prepareProteinForAtomtyping(mol, verbose=False)

        assert mol_equal(mol2, ref, exceptFields=("coords",))

    def test_atomtyping(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule, mol_equal
        from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
        from os import path
        import numpy as np

        mol = Molecule("3PTB")
        mol.filter("protein or water")
        pmol = prepareProteinForAtomtyping(mol, verbose=False)

        feats, failed = getProteinAtomFeatures(pmol)
        ref_file = path.join(home(dataDir="test-atomtyper"), "3ptb_atomtypes.npy")
        ref_feats = np.load(ref_file)

        # This is the wrongly bonded oxygen in LEU
        assert failed["labels"].shape[0] == 1
        assert np.array_equal(feats, ref_feats)


if __name__ == "__main__":
    unittest.main(verbosity=2)

    # from moleculekit.tools.atomtyper import (
    #     prepareProteinForAtomtyping,
    #     getProteinAtomFeatures,
    # )
    # from moleculekit.molecule import Molecule
    # from glob import glob
    # from tqdm import tqdm

    # all_failed = {}
    # for protf in tqdm(
    #     glob("/home/sdoerr/Datasets/scPDB_2017/scPDB/*/protein_prepared.pdb")
    # ):
    #     mol = Molecule(protf)
    #     # mol.filter("protein or water")
    #     # pmol = prepareProteinForAtomtyping(mol, verbose=False)
    #     features, failed = getProteinAtomFeatures(mol)
    #     all_failed[protf] = failed
