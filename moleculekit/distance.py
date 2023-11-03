import unittest


def cdist(coords1, coords2):
    from moleculekit.distance_utils import cdist
    import numpy as np

    assert coords1.ndim == 2, "cdist only supports 2D arrays"
    assert coords2.ndim == 2, "cdist only supports 2D arrays"
    assert coords1.shape[1] <= 3, "cdist only supports 1D, 2D and 3D coordinates"
    assert coords2.shape[1] <= 3, "cdist only supports 1D, 2D and 3D coordinates"
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
    assert coords.shape[1] <= 3, "pdist only supports 1D, 2D and 3D coordinates"
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


class _TestDistances(unittest.TestCase):
    def test_cdist(self):
        import numpy as np

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

    def test_pdist(self):
        import numpy as np

        refdists = np.array([2.828427, 5.656854, 2.828427])
        x = np.array([[4, 5], [6, 7], [8, 9]])
        dists = pdist(x)

        assert np.allclose(dists, refdists)


if __name__ == "__main__":
    unittest.main(verbosity=2)
