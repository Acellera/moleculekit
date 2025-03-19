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
