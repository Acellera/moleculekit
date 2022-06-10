import numpy as np
from moleculekit.bondguesser import make_grid_neighborlist_nonperiodic, grid_bonds


def guess_bonds(mol, num_processes=6):
    from moleculekit.periodictable import periodictable

    coords = mol.coords[:, :, mol.frame].copy()

    radii = []
    for el in mol.element:
        if el not in periodictable:
            raise RuntimeError(f"Unknown element '{el}'")
        if el == "H":
            radii.append(1.0)  # Override hydrogen radius to 1A
            continue
        vdw = periodictable[el].vdw_radius
        # Default radius for None radius is 2A
        radii.append(vdw if vdw is not None else 2)
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
    num_processes=6,
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
