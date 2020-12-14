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


# def isProteinProtonated(mol):
#     from moleculekit.tools.autosegment import autoSegment
#     from moleculekit.tools.preparation import proteinPrepare
#     import logging

#     mol = mol.copy()
#     mol.filter("protein")

#     numberHs = mol.atomselect("hydrogen").sum()
#     pmol = proteinPrepare(mol, _loggerLevel=logging.ERROR)
#     prepNumberHs = pmol.atomselect("hydrogen").sum()
#     addedHs = prepNumberHs - numberHs

#     if addedHs > (
#         mol.numAtoms / 3
#     ):  # Rough heuristic of what is a significant number of hydrogens to add
#         return False
#     return True


def isProteinProtonated(mol):
    numberHs = mol.atomselect("protein and hydrogen").sum()
    numberHeavy = mol.numAtoms - numberHs

    # Rough heuristic of what is a significant number of hydrogens
    if numberHs < (numberHeavy * 0.7):
        return False
    return True


def isLigandOptimized(mol, num_planes=3):
    from itertools import combinations
    import numpy as np
    import random

    if mol.numAtoms < 4:
        # Can't tell easily if it's optimized or not
        return True

    def random_combination(iterable, r):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)

    coords = mol.coords[:, :, 0]

    for _ in range(num_planes):
        atom_triplet = random_combination(range(mol.numAtoms), 3)
        atom_coords = coords[list(atom_triplet)]
        vec1 = atom_coords[1] - atom_coords[0]
        vec2 = atom_coords[2] - atom_coords[0]
        plane_normal = np.cross(vec1, vec2)
        plane_normal /= np.linalg.norm(plane_normal)

        all_vectors = coords - coords[atom_triplet[0]]
        projections = np.abs(np.dot(all_vectors, plane_normal))

        atoms_on_plane = np.where(projections < 1e-4)[0]
        # print(f"{atoms_on_plane} atoms lie on the plane defined by {atom_triplet}")
        # print(projections)

        # Arbitrary threshold. 3 Atoms will always be on the plane they define so I assume if two more are exactly on the same plane it's no chance
        if len(atoms_on_plane) > 5:
            return False

    return True


def isLigandDocked(prot, lig, threshold=10):
    dist = closestDistance(prot, lig)
    return dist < threshold


def areLigandsDocked(prot_file, sdf_file, threshold=10, maxCheck=None):
    from moleculekit.smallmol.smallmollib import SmallMolLib
    from moleculekit.molecule import Molecule

    not_docked = []
    prot = Molecule(prot_file)
    for i, lig in enumerate(SmallMolLib(sdf_file)):
        if maxCheck is not None and i >= maxCheck:
            break

        ligname = lig.ligname
        lig = lig.toMolecule()

        if not isLigandDocked(prot, lig, threshold):
            not_docked.append(ligname)

    return len(not_docked) == 0, not_docked


def areLigandsOptimized(sdf_file):
    from moleculekit.smallmol.smallmollib import SmallMolLib

    not_optimized = []
    for lig in SmallMolLib(sdf_file):
        ligname = lig.ligname
        lig = lig.toMolecule()

        if not isLigandOptimized(lig):
            not_optimized.append(ligname)

    return len(not_optimized) == 0, not_optimized


import unittest


class _TestMoleculeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from moleculekit.tools.preparation import proteinPrepare

        self.mol = Molecule("3ptb")

        self.prot = self.mol.copy()
        self.prot.filter("protein")

        self.prot_protonated = self.prot.copy()
        self.prot_protonated = proteinPrepare(self.prot_protonated)

        self.lig = self.mol.copy()
        self.lig.filter("resname BEN")

        self.lig_far = self.lig.copy()
        self.lig_far.moveBy([0, 20, 0])

        self.lig_flat = self.lig.copy()
        self.lig_flat.coords[:, 0, :] = -1.7  # Flatten on x axis

    def test_optimized_ligands(self):
        assert isLigandOptimized(self.lig)
        assert not isLigandOptimized(self.lig_flat)

    def test_docked_ligands(self):
        assert isLigandDocked(self.prot, self.lig)
        assert not isLigandDocked(self.prot, self.lig_far)

    def test_protonated_protein(self):
        assert not isProteinProtonated(self.prot)
        assert isProteinProtonated(self.prot_protonated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
