def cdist(coords1, coords2):
    from moleculekit.distance_utils import cdist
    import numpy as np

    return np.array(cdist(coords1.astype(np.float32), coords2.astype(np.float32)))


def pdist(coords):
    from moleculekit.distance_utils import pdist
    import numpy as np

    return np.array(pdist(coords.astype(np.float32)))


def squareform(distances):
    from moleculekit.distance_utils import squareform
    import numpy as np

    return np.array(squareform(distances.astype(np.float32)))
