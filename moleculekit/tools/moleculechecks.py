# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from typing import TYPE_CHECKING

from numpy import pi

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


def closestDistance(mol1: "Molecule", mol2: "Molecule"):
    """Compute the minimum distance between the atoms of two molecules.

    The two molecules are temporarily combined and the smallest pairwise
    distance between any atom of ``mol1`` and any atom of ``mol2`` is returned.

    Parameters
    ----------
    mol1 : moleculekit.molecule.Molecule
        The first molecule.
    mol2 : moleculekit.molecule.Molecule
        The second molecule.

    Returns
    -------
    dist : float
        The shortest distance, in Angstrom, between any atom of ``mol1`` and
        any atom of ``mol2``.
    """
    from moleculekit.projections.util import pp_calcMinDistances

    mol1 = mol1.copy()
    mol1.segid[:] = "X"
    mol2 = mol2.copy()
    mol2.segid[:] = "Y"

    mol1.append(mol2)

    sel1 = mol1.atomselect("segid X")[None, :]
    sel2 = mol1.atomselect("segid Y")[None, :]

    return pp_calcMinDistances(mol1, sel1, sel2, periodic=None).squeeze()


def isProteinProtonated(mol: "Molecule") -> bool:
    """Heuristically check whether a protein carries its hydrogen atoms.

    Counts the protein hydrogens versus protein heavy atoms and decides that
    the protein is protonated only if there is a significant number of
    hydrogens relative to the heavy atoms.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to inspect.

    Returns
    -------
    protonated : bool
        True if the protein appears to be protonated, False otherwise.
    """
    prot = mol.atomselect("protein")
    numberHs = (prot & (mol.element == "H")).sum()
    numberProt = prot.sum()
    numberHeavy = numberProt - numberHs

    # Rough heuristic of what is a significant number of hydrogens
    if numberHs < (numberHeavy * 0.7):
        return False
    return True


def isLigandOptimized(mol: "Molecule", atol: float = 1e-06) -> bool:
    """Check whether a ligand has a 3D-optimized (non-flat) geometry.

    All dihedral angles being either 0 or +/-pi means the ligand is planar,
    which indicates that it has not been optimized into a 3D conformation.
    Ligands with three or fewer atoms cannot be planar and are always
    considered optimized.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The ligand molecule. If it has no bonds, they are guessed in place.
    atol : float
        Absolute tolerance, in radians, for deciding whether a dihedral angle
        is effectively 0 or +/-pi (i.e. flat).

    Returns
    -------
    optimized : bool
        True if at least one dihedral angle deviates from a planar value
        (or the ligand has three or fewer atoms), False if the ligand is flat.
    """
    from moleculekit.util import calculateAnglesAndDihedrals

    if mol.numAtoms <= 3:  # Can't check for planarity with <=3 atoms
        return True

    if not len(mol.bonds):
        mol.guessBonds()

    _, dihedrals = calculateAnglesAndDihedrals(mol.bonds)
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


def isLigandDocked(prot: "Molecule", lig: "Molecule", threshold: float = 10) -> bool:
    """Check whether a ligand is docked close to a protein.

    Parameters
    ----------
    prot : moleculekit.molecule.Molecule
        The protein molecule.
    lig : moleculekit.molecule.Molecule
        The ligand molecule.
    threshold : float
        Maximum allowed distance, in Angstrom, between the closest protein and
        ligand atoms for the ligand to be considered docked.

    Returns
    -------
    docked : bool
        True if the closest protein-ligand distance is below ``threshold``,
        False otherwise.
    """
    dist = closestDistance(prot, lig)
    return dist < threshold


def areLigandsDocked(prot_file, sdf_file, threshold: float = 10, max_check: int | None = None):
    """Check whether all ligands in an SDF file are docked to a protein.

    Each ligand in ``sdf_file`` is tested against the protein with
    :func:`isLigandDocked`.

    Parameters
    ----------
    prot_file : str
        Path to the protein structure file.
    sdf_file : str
        Path to the SDF file containing one or more ligands.
    threshold : float
        Maximum allowed closest protein-ligand distance, in Angstrom, for a
        ligand to count as docked.
    max_check : int, optional
        If given, only the first ``max_check`` ligands are checked.

    Returns
    -------
    all_docked : bool
        True if every checked ligand is docked, False otherwise.
    not_docked : list of str
        The names of the ligands that were found not to be docked.
    """
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


def areLigandsOptimized(sdf_file, max_check: int | None = None):
    """Check whether all ligands in an SDF file have optimized 3D geometries.

    Each ligand in ``sdf_file`` is tested with :func:`isLigandOptimized`.

    Parameters
    ----------
    sdf_file : str
        Path to the SDF file containing one or more ligands.
    max_check : int, optional
        If given, only the first ``max_check`` ligands are checked.

    Returns
    -------
    all_optimized : bool
        True if every checked ligand is optimized, False otherwise.
    not_optimized : list of str
        The names of the ligands that were found to be flat (not optimized).
    """
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


def proteinHasBonds(mol: "Molecule") -> bool:
    """Check whether the protein atoms of a molecule are bonded together.

    Counts the bonds whose both atoms are protein atoms and compares against
    the number of protein atoms. A fully connected protein chain of ``n``
    atoms has at least ``n - 1`` bonds.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to inspect.

    Returns
    -------
    has_bonds : bool
        True if the protein atoms appear to be bonded together, False otherwise.
    """
    import numpy as np

    prot_idx = mol.atomselect("protein", indexes=True)
    num_prot_bonds = np.sum(np.all(np.isin(mol.bonds, prot_idx), axis=1))

    return num_prot_bonds >= len(prot_idx) - 1
