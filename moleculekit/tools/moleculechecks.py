# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from numpy import pi


def closestDistance(mol1, mol2):
    from moleculekit.projections.util import pp_calcMinDistances

    mol1 = mol1.copy()
    mol1.segid[:] = "X"
    mol2 = mol2.copy()
    mol2.segid[:] = "Y"

    mol1.append(mol2)

    sel1 = mol1.atomselect("segid X")[None, :]
    sel2 = mol1.atomselect("segid Y")[None, :]

    return pp_calcMinDistances(mol1, sel1, sel2, periodic=None).squeeze()


def isProteinProtonated(mol):
    prot = mol.atomselect("protein")
    numberHs = (prot & (mol.element == "H")).sum()
    numberProt = prot.sum()
    numberHeavy = numberProt - numberHs

    # Rough heuristic of what is a significant number of hydrogens
    if numberHs < (numberHeavy * 0.7):
        return False
    return True


def isLigandOptimized(mol, atol=1e-06):
    """Checks if a ligand is optimized. If all dihedral angles are 0 it means it's flat."""
    from moleculekit.util import guessAnglesAndDihedrals

    if mol.numAtoms <= 3:  # Can't check for planarity with <=3 atoms
        return True

    if not len(mol.bonds):
        mol.bonds = mol._guessBonds()

    _, dihedrals = guessAnglesAndDihedrals(mol.bonds)
    for dih in dihedrals:
        radians = mol.getDihedral(dih)
        if not (abs(radians) < atol or abs(abs(radians) - pi) < atol):
            return True
    return False


# def isLigandOptimized(mol, num_planes=3):
#     import numpy as np
#     import random

#     if mol.numAtoms < 4:
#         # Can't tell easily if it's optimized or not
#         return True

#     def random_combination(iterable, r):
#         "Random selection from itertools.combinations(iterable, r)"
#         pool = tuple(iterable)
#         n = len(pool)
#         indices = sorted(random.sample(range(n), r))
#         return tuple(pool[i] for i in indices)

#     coords = mol.coords[:, :, 0]

#     for _ in range(num_planes):
#         atom_triplet = random_combination(range(mol.numAtoms), 3)
#         atom_coords = coords[list(atom_triplet)]
#         vec1 = atom_coords[1] - atom_coords[0]
#         vec2 = atom_coords[2] - atom_coords[0]
#         plane_normal = np.cross(vec1, vec2)
#         plane_normal /= np.linalg.norm(plane_normal)

#         all_vectors = coords - coords[atom_triplet[0]]
#         projections = np.abs(np.dot(all_vectors, plane_normal))

#         atoms_on_plane = np.where(projections < 1e-4)[0]
#         # print(f"{atoms_on_plane} atoms lie on the plane defined by {atom_triplet}")
#         # print(projections)

#         # Arbitrary threshold. 3 Atoms will always be on the plane they define so I assume if two more are exactly on the same plane it's no chance
#         if len(atoms_on_plane) > 5:
#             return False

#     return True


def isLigandDocked(prot, lig, threshold=10):
    dist = closestDistance(prot, lig)
    return dist < threshold


def areLigandsDocked(prot_file, sdf_file, threshold=10, max_check=None):
    from moleculekit.smallmol.smallmollib import SmallMolLib
    from moleculekit.molecule import Molecule

    not_docked = []
    prot = Molecule(prot_file)
    for i, lig in enumerate(SmallMolLib(sdf_file)):
        if max_check is not None and i >= max_check:
            break

        ligname = lig.ligname
        lig = lig.toMolecule()

        if not isLigandDocked(prot, lig, threshold):
            not_docked.append(ligname)

    return len(not_docked) == 0, not_docked


def areLigandsOptimized(sdf_file, max_check=None):
    from moleculekit.smallmol.smallmollib import SmallMolLib

    not_optimized = []
    for i, lig in enumerate(SmallMolLib(sdf_file)):
        if max_check is not None and i >= max_check:
            break

        ligname = lig.ligname
        lig = lig.toMolecule()

        if not isLigandOptimized(lig):
            not_optimized.append(ligname)

    return len(not_optimized) == 0, not_optimized


def proteinHasBonds(mol):
    import numpy as np

    prot_idx = mol.atomselect("protein", indexes=True)
    num_prot_bonds = np.sum(np.all(np.isin(mol.bonds, prot_idx), axis=1))

    return num_prot_bonds >= len(prot_idx) - 1
