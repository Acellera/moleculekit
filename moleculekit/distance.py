def find_clashes(
    mol,
    sel1=None,
    sel2=None,
    overlap=0.6,
    exclude_bonded=True,
    exclude_14=True,
    guess_bonds=True,
):
    """Find pairs of atoms that sterically clash with each other.

    A clash is defined as a pair of atoms whose interatomic distance is less
    than ``r_vdw_1 + r_vdw_2 - overlap`` where VdW radii come from
    ``moleculekit.periodictable``.  Uses the bundled :class:`cKDTree`
    (ported from SciPy) for fast neighbor lookup.

    Parameters
    ----------
    mol : Molecule
        The molecule to analyze.
    sel1 : str, ndarray of bool, or None, optional
        First selection.  If None, all atoms are used.
    sel2 : str, ndarray of bool, or None, optional
        Second selection.  If None, uses ``sel1`` (self-clashes).
    overlap : float, optional
        How much VdW overlap is tolerated before flagging as a clash, in
        Angstroms.  Default 0.6 -- i.e. atoms clash when they overlap by
        more than 0.6 Å of their combined VdW radii.  Set to 0 for strict
        contact (any overlap counts), or negative for looser definitions.
    exclude_bonded : bool, optional
        If True, 1-2 (directly bonded) and 1-3 (angle) neighbors are
        excluded from the clash search.  Default True.
    exclude_14 : bool, optional
        If True, 1-4 (dihedral) neighbors are also excluded.  Default True.
    guess_bonds : bool, optional
        If True, supplements ``mol.bonds`` with moleculekit's
        distance/covalent-radius based bond guesser.  This catches
        inter-residue peptide bonds, disulfides, etc. that are often
        absent from ``mol.bonds`` on PDB-loaded structures.  Set to
        False if ``mol.bonds`` is already complete (e.g. for systems
        built from a topology file) to skip the guessing overhead and
        avoid false positives from overlapping atoms.  Default True.

    Returns
    -------
    clashes : ndarray of shape (N, 2), dtype int
        Pairs of atom indices that clash.  Pairs are ordered so the first
        index is always < the second.  Empty array if no clashes.
    distances : ndarray of shape (N,), dtype float32
        Distance (Å) for each clash pair.
    overlaps : ndarray of shape (N,), dtype float32
        Overlap amount ``(r_vdw_1 + r_vdw_2) - distance`` for each pair.
        Pairs are returned sorted by overlap (most severe first).

    Examples
    --------
    >>> mol = Molecule("3ptb")
    >>> clashes, distances, overlaps = find_clashes(mol)
    >>> for (a, b), d, o in zip(clashes, distances, overlaps):
    ...     print(f"{mol.name[a]}({a}) <-> {mol.name[b]}({b}): "
    ...           f"d={d:.2f} overlap={o:.2f}")
    """
    import numpy as np
    from moleculekit.kdtree import cKDTree
    from moleculekit.periodictable import periodictable

    mask1 = mol.atomselect(sel1)
    mask2 = mol.atomselect(sel2) if sel2 is not None else mask1
    self_mode = sel2 is None or np.array_equal(mask1, mask2)

    idx1 = np.where(mask1)[0].astype(np.int64)
    idx2 = np.where(mask2)[0].astype(np.int64)
    if idx1.size == 0 or idx2.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    frame = mol.frame
    coords = np.ascontiguousarray(mol.coords[:, :, frame], dtype=np.float32)

    # Per-atom VdW radii (fallback to carbon's 1.7 Å for unknown elements)
    radii = np.array(
        [
            periodictable.get(e, periodictable["C"]).vdw_radius or 1.7
            for e in mol.element
        ],
        dtype=np.float32,
    )
    max_radius = float(radii.max())
    cutoff = 2.0 * max_radius - overlap

    # KDTree neighbor search (O(N log N) average).  cKDTree expects float64.
    tree1 = cKDTree(coords[idx1].astype(np.float64, copy=False))
    if self_mode:
        pair_idx = np.asarray(
            tree1.query_pairs(cutoff, output_type="ndarray"), dtype=np.int64
        )
        if pair_idx.size == 0:
            local_i = local_j = np.empty(0, dtype=np.int64)
        else:
            local_i = pair_idx[:, 0]
            local_j = pair_idx[:, 1]
    else:
        tree2 = cKDTree(coords[idx2].astype(np.float64, copy=False))
        # Returns a list (len == tree1.n) of lists of neighbor indices in tree2.
        nn = tree1.query_ball_tree(tree2, cutoff)
        if any(nn):
            ii = np.concatenate(
                [
                    np.full(len(lst), i, dtype=np.int64)
                    for i, lst in enumerate(nn)
                    if lst
                ]
            )
            jj = np.concatenate([np.asarray(lst, dtype=np.int64) for lst in nn if lst])
            local_i, local_j = ii, jj
        else:
            local_i = local_j = np.empty(0, dtype=np.int64)

    if local_i.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    # Map back to original atom indices
    a_arr = idx1[local_i]
    b_arr = idx2[local_j]

    if self_mode:
        # query_pairs guarantees i < j already, but keep the filter defensive
        keep = a_arr < b_arr
        a_arr = a_arr[keep]
        b_arr = b_arr[keep]

    if a_arr.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    # Filter bonded/1-3/1-4 neighbors
    if exclude_bonded or exclude_14:
        from collections import defaultdict

        # Combine file-provided bonds with moleculekit's bond guesser so
        # inter-residue peptide bonds, disulfides, etc. that are often
        # missing from ``mol.bonds`` on PDB-loaded structures are still
        # excluded.  ``guess_bonds=False`` skips the guesser for systems
        # whose bond table is already trusted.
        all_bonds = mol._getBonds(fileBonds=True, guessBonds=guess_bonds)
        bonded = defaultdict(set)
        for a, b in all_bonds:
            a, b = int(a), int(b)
            bonded[a].add(b)
            bonded[b].add(a)

        excluded = set()
        if exclude_bonded:
            for a, neigh in bonded.items():
                for b in neigh:
                    excluded.add((min(a, b), max(a, b)))
                    # 1-3 (angle neighbors)
                    for c in bonded[b]:
                        if c != a:
                            excluded.add((min(a, c), max(a, c)))
        if exclude_14:
            # 1-4 (dihedral neighbors)
            for a, neigh in bonded.items():
                for b in neigh:
                    for c in bonded[b]:
                        if c == a:
                            continue
                        for d in bonded[c]:
                            if d == b or d == a:
                                continue
                            excluded.add((min(a, d), max(a, d)))

        keep = np.array(
            [(int(a), int(b)) not in excluded for a, b in zip(a_arr, b_arr)]
        )
        a_arr = a_arr[keep]
        b_arr = b_arr[keep]

    if a_arr.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    # Compute distances and per-pair VdW overlap cutoff
    diffs = coords[a_arr] - coords[b_arr]
    distances = np.sqrt(np.einsum("ij,ij->i", diffs, diffs)).astype(np.float32)
    vdw_sum = (radii[a_arr] + radii[b_arr]).astype(np.float32)
    clash_mask = distances < (vdw_sum - overlap)

    a_arr = a_arr[clash_mask]
    b_arr = b_arr[clash_mask]
    distances = distances[clash_mask]
    overlaps = (radii[a_arr] + radii[b_arr] - distances).astype(np.float32)

    pairs = np.stack([a_arr, b_arr], axis=1)
    order = np.argsort(-overlaps)
    return pairs[order], distances[order], overlaps[order]


def cdist(coords1, coords2):
    from moleculekit.distance_utils import cdist
    import numpy as np

    assert coords1.ndim == 2, "cdist only supports 2D arrays"
    assert coords2.ndim == 2, "cdist only supports 2D arrays"
    assert (
        coords1.shape[1] == coords2.shape[1]
    ), "Second dimension of input arguments must match"
    if coords1.dtype != np.float32:
        coords1 = coords1.astype(np.float32)
    if coords2.dtype != np.float32:
        coords2 = coords2.astype(np.float32)

    results = np.zeros((coords1.shape[0], coords2.shape[0]), dtype=np.float32)
    cdist(coords1, coords2, results)
    return results


def pdist(coords):
    from moleculekit.distance_utils import pdist
    import numpy as np

    assert coords.ndim == 2, "pdist only supports 2D arrays"
    if coords.dtype != np.float32:
        coords = coords.astype(np.float32)

    n_points = coords.shape[0]
    results = np.zeros(int(n_points * (n_points - 1) / 2), dtype=np.float32)
    pdist(coords, results)
    return results


def squareform(distances):
    from moleculekit.distance_utils import squareform
    import numpy as np

    return np.array(squareform(distances.astype(np.float32)))


def calculate_contacts(mol, sel1, sel2, periodic, threshold=4):
    from moleculekit.distance_utils import contacts_trajectory
    import numpy as np

    assert isinstance(sel1, np.ndarray) and sel1.dtype == bool
    assert isinstance(sel2, np.ndarray) and sel2.dtype == bool

    selfdist = np.array_equal(sel1, sel2)
    sel1 = np.where(sel1)[0].astype(np.uint32)
    sel2 = np.where(sel2)[0].astype(np.uint32)

    coords = mol.coords
    box = mol.box
    if periodic is not None:
        if box is None or np.sum(box) == 0:
            raise RuntimeError(
                "No periodic box dimensions given in the molecule/trajectory. "
                "If you want to calculate distance without wrapping, set the periodic option to None"
            )
    else:
        box = np.zeros((3, coords.shape[2]), dtype=np.float32)

    if box.shape[1] != coords.shape[2]:
        raise RuntimeError(
            "Different number of frames in mol.coords and mol.box. "
            "Please ensure they both have the same number of frames"
        )

    # Digitize chains to not do PBC calculations of the same chain
    if periodic is None:  # Won't be used since box is 0
        digitized_chains = np.zeros(mol.numAtoms, dtype=np.uint32)
    elif periodic == "chains":
        digitized_chains = np.unique(mol.chain, return_inverse=True)[1].astype(
            np.uint32
        )
    elif periodic == "selections":
        digitized_chains = np.ones(mol.numAtoms, dtype=np.uint32)
        digitized_chains[sel2] = 2
    else:
        raise RuntimeError(f"Invalid periodic option {periodic}")

    results = contacts_trajectory(
        coords,
        box,
        sel1,
        sel2,
        digitized_chains,
        selfdist,
        periodic is not None,
        threshold,
    )
    for f in range(len(results)):
        results[f] = np.array(results[f], dtype=np.uint32).reshape(-1, 2)

    return results
