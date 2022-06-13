import numpy as np
from moleculekit.bondguesser_utils import make_grid_neighborlist_nonperiodic, grid_bonds
import unittest

# VdW radii are from A. Bondi J. Phys. Chem., 68, 441 - 452, 1964
# H is overriden to 1A from 1.2A in J.Phys.Chem., 100, 7384 - 7391, 1996
# Unavailable radii are set to 2A
# Ion radii are taken from CHARMM27 parameters
vdw_radii = {
    "H": 1,
    "He": 1.4,
    "Li": 1.82,
    "Be": 2.0,
    "B": 2.0,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 1.36,
    "Mg": 1.18,
    "Al": 2.0,
    "Si": 2.1,
    "P": 1.8,
    "S": 1.8,
    "Cl": 2.27,
    "Ar": 1.88,
    "K": 1.76,
    "Ca": 1.37,
    "Sc": 2.0,
    "Ti": 2.0,
    "V": 2.0,
    "Cr": 2.0,
    "Mn": 2.0,
    "Fe": 2.0,
    "Co": 2.0,
    "Ni": 1.63,
    "Cu": 1.4,
    "Zn": 1.39,
    "Ga": 1.07,
    "Ge": 2.0,
    "As": 1.85,
    "Se": 1.9,
    "Br": 1.85,
    "Kr": 2.02,
    "Rb": 2.0,
    "Sr": 2.0,
    "Y": 2.0,
    "Zr": 2.0,
    "Nb": 2.0,
    "Mo": 2.0,
    "Tc": 2.0,
    "Ru": 2.0,
    "Rh": 2.0,
    "Pd": 1.63,
    "Ag": 1.72,
    "Cd": 1.58,
    "In": 1.93,
    "Sn": 2.17,
    "Sb": 2.0,
    "Te": 2.06,
    "I": 1.98,
    "Xe": 2.16,
    "Cs": 2.1,
    "Ba": 2.0,
    "La": 2.0,
    "Ce": 2.0,
    "Pr": 2.0,
    "Nd": 2.0,
    "Pm": 2.0,
    "Sm": 2.0,
    "Eu": 2.0,
    "Gd": 2.0,
    "Tb": 2.0,
    "Dy": 2.0,
    "Ho": 2.0,
    "Er": 2.0,
    "Tm": 2.0,
    "Yb": 2.0,
    "Lu": 2.0,
    "Hf": 2.0,
    "Ta": 2.0,
    "W": 2.0,
    "Re": 2.0,
    "Os": 2.0,
    "Ir": 2.0,
    "Pt": 1.72,
    "Au": 1.66,
    "Hg": 1.55,
    "Tl": 1.96,
    "Pb": 2.02,
    "Bi": 2.0,
    "Po": 2.0,
    "At": 2.0,
    "Rn": 2.0,
    "Fr": 2.0,
    "Ra": 2.0,
    "Ac": 2.0,
    "Th": 2.0,
    "Pa": 2.0,
    "U": 1.86,
    "Np": 2.0,
    "Pu": 2.0,
    "Am": 2.0,
    "Cm": 2.0,
    "Bk": 2.0,
    "Cf": 2.0,
    "Es": 2.0,
    "Fm": 2.0,
    "Md": 2.0,
    "No": 2.0,
    "Lr": 2.0,
    "Rf": 2.0,
    "Db": 2.0,
    "Sg": 2.0,
    "Bh": 2.0,
    "Hs": 2.0,
    "Mt": 2.0,
    "Ds": 2.0,
    "Rg": 2.0,
}


def guess_bonds(mol, num_processes=6):
    # from moleculekit.periodictable import periodictable

    # exceptions = {
    #     "H": 1,
    #     "Na": 1.36,
    #     "Mg": 1.18,
    #     "Cl": 2.27,
    #     "K": 1.76,
    #     "Ca": 1.37,
    #     "Ni": 1.63,
    #     "Cu": 1.4,
    #     "Zn": 1.39,
    #     "Ga": 1.07,
    # }

    coords = mol.coords[:, :, mol.frame].copy()

    radii = []
    for el in mol.element:
        if el not in vdw_radii:
            raise RuntimeError(f"Unknown element '{el}'")
        # Default radius for None radius is 2A
        radii.append(vdw_radii[el])
    radii = np.array(radii)

    is_hydrogen = mol.element == "H"

    # Setting the grid box distance to 1.2 times the largest VdW radius
    grid_cutoff = np.max(radii) * 1.2

    bonds = bond_grid_search(
        coords, grid_cutoff, is_hydrogen, radii, num_processes=num_processes
    )

    # bonds = guess_bonds(mol.coords, radii, is_hydrogen, grid_cutoff)
    return bonds


def bond_grid_search(
    coords,
    grid_cutoff,
    is_hydrogen,
    radii,
    max_boxes=4e6,
    cutoff_incr=1.26,
    num_processes=1,
):
    from collections import defaultdict
    from multiprocessing import Pool

    # Get min and max coords
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    xyzrange = max_c - min_c

    # Increase grid_cutoff until it won't generate more than max_boxes
    newpairdist = grid_cutoff
    pairdist = grid_cutoff
    xyz_boxes = np.floor(xyzrange / pairdist).astype(np.uint32) + 1
    num_boxes = xyz_boxes[0] * xyz_boxes[1] * xyz_boxes[2]
    while num_boxes > max_boxes or num_boxes < 1:
        pairdist = newpairdist
        xyz_boxes = (xyzrange / pairdist) + 1
        num_boxes = xyz_boxes[0] * xyz_boxes[1] * xyz_boxes[2]
        newpairdist = pairdist * cutoff_incr  # sqrt(2) ~= 1.26

    # Compute box index for all atoms
    box_idx = np.floor((coords - min_c) / pairdist).astype(int)
    # Clip box indexes within range
    box_idx[:, 0] = np.clip(box_idx[:, 0], 0, xyz_boxes[0] - 1)
    box_idx[:, 1] = np.clip(box_idx[:, 1], 0, xyz_boxes[1] - 1)
    box_idx[:, 2] = np.clip(box_idx[:, 2], 0, xyz_boxes[2] - 1)

    # Convert to single box index
    xy_boxes = xyz_boxes[0] * xyz_boxes[1]
    box_idx = box_idx[:, 2] * xy_boxes + box_idx[:, 1] * xyz_boxes[0] + box_idx[:, 0]

    atoms_in_box = defaultdict(list)
    for i in range(len(box_idx)):
        atoms_in_box[box_idx[i]].append(i)

    gridlist = np.full((num_boxes, 14), num_boxes).astype(np.uint32)
    make_grid_neighborlist_nonperiodic(
        gridlist, xyz_boxes[0], xyz_boxes[1], xyz_boxes[2]
    )

    args = []
    fixedargs = [
        atoms_in_box,
        gridlist,
        num_boxes,
        coords,
        radii,
        is_hydrogen,
        pairdist,
    ]
    for boxidx in atoms_in_box.keys():
        args.append([boxidx, *fixedargs])

    with Pool(processes=num_processes) as p:
        results = p.starmap(_thread_func, args)
    results = np.vstack([rr for rr in results if len(rr)])

    return results


# One of these is executed per box
def _thread_func(
    boxidx, atoms_in_box, gridlist, num_boxes, coords, radii, is_hydrogen, pairdist
):
    if boxidx not in atoms_in_box:
        return []

    box_atoms = np.array(atoms_in_box[boxidx])
    neigh_boxes = [x for x in gridlist[boxidx] if x != num_boxes]
    neigh_atoms = np.hstack([atoms_in_box[nb] for nb in neigh_boxes[1:]])
    neigh_atoms = neigh_atoms.astype(int)
    all_atoms = np.hstack((box_atoms, neigh_atoms)).astype(int)
    # First come the coords of atoms in the box, then neighbouring cells
    sub_coords = np.concatenate([coords[box_atoms], coords[neigh_atoms]], axis=0)
    num_in_box = len(box_atoms)

    sub_radii = radii[all_atoms].astype(np.float32)
    sub_is_hydrogen = is_hydrogen[all_atoms].astype(np.uint32)
    pairs = grid_bonds(sub_coords, sub_radii, sub_is_hydrogen, num_in_box, pairdist)
    real_pairs = []
    for i in range(int(len(pairs) / 2)):
        real_pairs.append([all_atoms[pairs[i * 2]], all_atoms[pairs[i * 2 + 1]]])
    return real_pairs


class _TestBondGuesser(unittest.TestCase):
    def test_bond_guessing(self):
        from moleculekit.molecule import Molecule, calculateUniqueBonds
        from moleculekit.home import home
        import os
        import time

        pdbids = [
            "3ptb",
            "3hyd",
            "6a5j",
            "5vbl",
            "7q5b",
            "1unc",
            "3zhi",
            "1a25",
            "1u5u",
            "1gzm",
            "6va1",
            "1bna",
            "3wbm",
            "1awf",
            "5vav",
        ]

        for pi in pdbids:
            with self.subTest(pdb=pi):
                mol = Molecule(pi)
                bonds = guess_bonds(mol)

                reff = os.path.join(home(dataDir="test-bondguesser"), f"{pi}.csv")

                bonds, _ = calculateUniqueBonds(bonds.astype(np.uint32), [])
                bondsref, _ = calculateUniqueBonds(mol._guessBonds(), [])

                x1 = time.time()
                _ = guess_bonds(mol)
                x1 = time.time() - x1
                x2 = time.time()
                _ = mol._guessBonds()
                x2 = time.time() - x2
                print(f"Times {x1}, {x2}")

                with open(reff, "w") as f:
                    for b in range(bondsref.shape[0]):
                        f.write(f"{bondsref[b, 0]},{bondsref[b, 1]}\n")

                assert np.array_equal(bonds, bondsref)


if __name__ == "__main__":
    unittest.main(verbosity=2)
