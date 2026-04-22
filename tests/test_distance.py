def _test_cdist():
    import numpy as np
    from moleculekit.distance import cdist

    refdists = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
    x = np.array([0, 1, 2])[:, None]
    y = np.array([3, 4, 5])[:, None]
    dists = cdist(x, y)

    assert np.allclose(dists, refdists)

    refdists = np.array(
        [[5.656854, 8.485281, 11.313708], [2.828427, 5.656854, 8.485281]]
    )
    dists = cdist(np.array([[0, 1], [2, 3]]), np.array([[4, 5], [6, 7], [8, 9]]))

    assert np.allclose(dists, refdists)


def _test_pdist():
    import numpy as np
    from moleculekit.distance import pdist

    refdists = np.array([2.828427, 5.656854, 2.828427])
    x = np.array([[4, 5], [6, 7], [8, 9]])
    dists = pdist(x)

    assert np.allclose(dists, refdists)


# ---------------------------------------------------------------------------
# find_clashes
# ---------------------------------------------------------------------------

def _make_synthetic_molecule(coords, names, elements, bonds=None):
    """Build a minimal Molecule with the given atoms/bonds (all one residue)."""
    import numpy as np
    from moleculekit.molecule import Molecule

    n = len(names)
    mol = Molecule().empty(n)
    mol.name[:] = names
    mol.element[:] = elements
    mol.record[:] = "ATOM"
    mol.resname[:] = "MOL"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.coords = np.asarray(coords, dtype=np.float32).reshape(n, 3, 1)
    if bonds is not None:
        mol.bonds = np.asarray(bonds, dtype=Molecule._dtypes["bonds"])
        mol.bondtype = np.asarray(["1"] * len(bonds), dtype=Molecule._dtypes["bondtype"])
    return mol


def _test_find_clashes_synthetic_two_atoms():
    """Two carbons closer than 2*r_vdw - overlap should clash; further apart
    they should not."""
    import numpy as np
    from moleculekit.distance import find_clashes

    # Two carbons at 2.0 Å: 2*1.7 - 2.0 = 1.4 Å overlap > 0.6 -> clash.
    # Use guess_bonds=False so the synthetic geometry isn't auto-bonded.
    mol = _make_synthetic_molecule(
        coords=[[0, 0, 0], [2.0, 0, 0]],
        names=["C1", "C2"],
        elements=["C", "C"],
    )
    clashes, distances, overlaps = find_clashes(mol, guess_bonds=False)
    assert len(clashes) == 1
    assert tuple(clashes[0]) == (0, 1)
    assert np.isclose(distances[0], 2.0, atol=1e-3)
    assert np.isclose(overlaps[0], 3.4 - 2.0, atol=1e-3)

    # Same atoms at 3.5 Å: 2*1.7 - 3.5 = -0.1 Å overlap -> not a clash
    mol.coords[1, 0, 0] = 3.5
    assert len(find_clashes(mol, guess_bonds=False)[0]) == 0


def _test_find_clashes_threshold_overlap():
    """The ``overlap`` parameter controls the VdW tolerance."""
    from moleculekit.distance import find_clashes

    # Two carbons at 3.1 Å -> VdW overlap = 3.4 - 3.1 = 0.3 Å.
    mol = _make_synthetic_molecule(
        coords=[[0, 0, 0], [3.1, 0, 0]],
        names=["C1", "C2"],
        elements=["C", "C"],
    )
    kw = dict(guess_bonds=False)
    assert len(find_clashes(mol, overlap=0.6, **kw)[0]) == 0  # 0.3 < 0.6
    assert len(find_clashes(mol, overlap=0.2, **kw)[0]) == 1  # 0.3 > 0.2
    assert len(find_clashes(mol, overlap=0.0, **kw)[0]) == 1  # any overlap


def _test_find_clashes_excludes_bonded():
    """Two carbons at 1.54 Å (C-C bond length) must not be flagged when
    they are bonded, but must be flagged when they are not."""
    import numpy as np
    from moleculekit.distance import find_clashes

    # Case 1: two bonded carbons at bond length.
    mol_bonded = _make_synthetic_molecule(
        coords=[[0, 0, 0], [1.54, 0, 0]],
        names=["C1", "C2"],
        elements=["C", "C"],
        bonds=[[0, 1]],
    )
    assert len(find_clashes(mol_bonded)[0]) == 0
    # Forcing exclude_bonded=False makes them clash again.
    assert len(find_clashes(mol_bonded, exclude_bonded=False, exclude_14=False)[0]) == 1


def _test_find_clashes_excludes_13():
    """Three carbons in a C-C-C angle geometry: the 1-3 pair is at
    ~2.5 Å and must not be flagged by default."""
    import math
    from moleculekit.distance import find_clashes

    l = 1.54
    a = math.radians(109.5)
    c1 = (0.0, 0.0, 0.0)
    c2 = (l, 0.0, 0.0)
    c3 = (l + l * math.cos(math.pi - a), l * math.sin(math.pi - a), 0.0)
    mol = _make_synthetic_molecule(
        coords=[c1, c2, c3],
        names=["C1", "C2", "C3"],
        elements=["C", "C", "C"],
        bonds=[[0, 1], [1, 2]],
    )
    # All default -> 1-2 and 1-3 filtered
    assert len(find_clashes(mol)[0]) == 0
    # Keep bonds excluded but reveal 1-3 by disabling exclude_bonded -> C1/C3 clash
    # (exclude_bonded=False drops both 1-2 and 1-3; but C1-C2 and C2-C3 are at
    #  1.54 Å which also clashes -> 3 clashes total: 1-2, 2-3, 1-3)
    clashes = find_clashes(mol, exclude_bonded=False, exclude_14=False)[0]
    assert len(clashes) == 3


def _test_find_clashes_ordered_by_overlap():
    """Results should be returned sorted by overlap severity, worst first."""
    from moleculekit.distance import find_clashes

    # Three well-separated pairs at different intra-pair distances.
    coords = [
        [0, 0, 0], [1.5, 0, 0],     # Pair A: biggest overlap
        [0, 10, 0], [2.0, 10, 0],    # Pair B
        [0, 20, 0], [2.5, 20, 0],    # Pair C: smallest overlap
    ]
    mol = _make_synthetic_molecule(
        coords=coords,
        names=[f"C{i}" for i in range(6)],
        elements=["C"] * 6,
    )
    clashes, distances, overlaps = find_clashes(mol, guess_bonds=False)
    assert len(clashes) == 3
    assert overlaps[0] > overlaps[1] > overlaps[2]
    assert distances[0] < distances[1] < distances[2]
    assert all(a < b for a, b in clashes)


def _test_find_clashes_self_vs_cross_selection():
    """Cross-selection mode should restrict clashes to the cartesian product
    of sel1 x sel2, while self-mode returns unordered pairs within sel1."""
    import numpy as np
    from moleculekit.distance import find_clashes

    # Three atoms in a line at 2.0 Å spacing; adjacent pairs clash, the
    # (0,2) pair at 4.0 Å does not.
    mol = _make_synthetic_molecule(
        coords=[[0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]],
        names=["C0", "C1", "C2"],
        elements=["C", "C", "C"],
    )
    self_clashes = find_clashes(mol, guess_bonds=False)[0]
    assert sorted(map(tuple, self_clashes)) == [(0, 1), (1, 2)]

    # Cross mode: sel1={0}, sel2={1,2} -> only (0,1) is in VdW clash range.
    sel1 = np.array([True, False, False])
    sel2 = np.array([False, True, True])
    cross_clashes = find_clashes(mol, sel1, sel2, guess_bonds=False)[0]
    assert sorted(map(tuple, cross_clashes)) == [(0, 1)]


def _test_find_clashes_selection_input_types():
    """Mask/str/indices/None must produce identical results."""
    import numpy as np
    from moleculekit.molecule import Molecule
    from moleculekit.distance import find_clashes

    mol = Molecule("3ptb")

    a, _, _ = find_clashes(mol, "protein", "not protein")
    mask_prot = mol.atomselect("protein")
    mask_other = mol.atomselect("not protein")
    b, _, _ = find_clashes(mol, mask_prot, mask_other)
    c, _, _ = find_clashes(mol, np.where(mask_prot)[0], np.where(mask_other)[0])
    assert len(a) == len(b) == len(c)
    assert sorted(map(tuple, a)) == sorted(map(tuple, b)) == sorted(map(tuple, c))


def _test_find_clashes_empty_selection():
    """Empty selections should return empty outputs without errors."""
    from moleculekit.distance import find_clashes

    mol = _make_synthetic_molecule(
        coords=[[0, 0, 0], [2.0, 0, 0]],
        names=["C1", "C2"],
        elements=["C", "C"],
    )
    clashes, distances, overlaps = find_clashes(mol, "resname NONEXISTENT")
    assert clashes.shape == (0, 2)
    assert distances.shape == (0,)
    assert overlaps.shape == (0,)


def _test_find_clashes_3ptb_calcium_coordination():
    """3ptb has a Ca2+ ion coordinated by 6 nearby oxygens.  These are the
    only 'clashes' that survive the bonded-exclusion filter because they
    are not covalent bonds."""
    from moleculekit.molecule import Molecule
    from moleculekit.distance import find_clashes

    mol = Molecule("3ptb")
    clashes, distances, overlaps = find_clashes(mol)

    # All 6 surviving clashes should involve the Ca2+ ion.
    ca_idx = int((mol.resname == "CA").nonzero()[0][0])
    involves_ca = [(ca_idx in (int(a), int(b))) for a, b in clashes]
    assert len(clashes) == 6
    assert all(involves_ca)

    # Distances are all within physical Ca-O coordination range.
    assert (distances >= 2.0).all() and (distances <= 2.6).all()


def _test_find_clashes_guess_bonds_flag():
    """With ``guess_bonds=False`` only bonds present in ``mol.bonds`` are
    used, so covalent contacts missing from the bond table (e.g. two
    overlapping atoms not connected in ``mol.bonds``) are flagged as
    clashes instead of being filtered away."""
    from moleculekit.distance import find_clashes

    # Two carbons at 1.6 Å (bond length) with NO entry in mol.bonds.
    mol = _make_synthetic_molecule(
        coords=[[0, 0, 0], [1.6, 0, 0]],
        names=["C1", "C2"],
        elements=["C", "C"],
    )
    # Default: the guesser infers a bond -> no clash.
    assert len(find_clashes(mol)[0]) == 0
    # guess_bonds=False: nothing in mol.bonds, so it IS reported as a clash.
    assert len(find_clashes(mol, guess_bonds=False)[0]) == 1


def _test_find_clashes_disulfide_not_flagged():
    """3ptb has six disulfide bridges.  Even though SG-SG is closer than the
    VdW sum (2.05 Å vs 3.6 Å), they must not be reported as clashes
    because the bond guesser picks them up."""
    from moleculekit.molecule import Molecule
    from moleculekit.distance import find_clashes

    mol = Molecule("3ptb")
    clashes, _, _ = find_clashes(mol, "name SG", "name SG")
    assert len(clashes) == 0


def _test_find_clashes_exclude_14_flag():
    """Disabling exclude_14 reveals 1-4 partners that are closer than the
    VdW-overlap cutoff, so the count should only ever go up (never down)."""
    from moleculekit.molecule import Molecule
    from moleculekit.distance import find_clashes

    mol = Molecule("1ubq")
    default = find_clashes(mol)[0]
    no_14 = find_clashes(mol, exclude_14=False)[0]
    assert len(no_14) >= len(default)
    # Every "real" clash from the default run must survive when we loosen
    # the filter -- we can only gain pairs, not lose them.
    default_set = set(map(tuple, default))
    no_14_set = set(map(tuple, no_14))
    assert default_set.issubset(no_14_set)


def _test_find_clashes_return_dtypes():
    """Return types and shapes should match the documented contract."""
    import numpy as np
    from moleculekit.distance import find_clashes

    mol = _make_synthetic_molecule(
        coords=[[0, 0, 0], [2.0, 0, 0]],
        names=["C1", "C2"],
        elements=["C", "C"],
    )
    clashes, distances, overlaps = find_clashes(mol, guess_bonds=False)
    assert clashes.ndim == 2 and clashes.shape[1] == 2
    assert clashes.dtype.kind == "i"
    assert distances.dtype == np.float32
    assert overlaps.dtype == np.float32
    assert len(clashes) == len(distances) == len(overlaps) == 1
