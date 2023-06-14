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
