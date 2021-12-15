# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np


def _wrapBondedDistance(pos, box):
    # Assuming bonds can't cross multiple periodic boxes this is marginally faster
    hbox = box / 2
    under = pos < -hbox
    over = pos > hbox
    pos[under] += box[under]
    pos[over] -= box[over]
    return pos


def dihedralAngle(pos, box=None):
    """Calculates a dihedral angle.

    Parameters
    ----------
    pos: np.ndarray
        An array of 4x3 size where each row are the coordinates of an atom defining the dihedral angle
    box: np.ndarray
        The size of the periodic box

    Returns
    -------
    angle: float
        The angle in radians
    """
    if pos.shape[0] != 4 or pos.shape[1] != 3:
        raise RuntimeError(
            "dihedralAngles requires a 4x3 sized coordinate matrix as input."
        )

    r12 = pos[0] - pos[1]
    r23 = pos[1] - pos[2]
    r34 = pos[2] - pos[3]
    if box is not None and not np.all(box == 0):
        r12 = _wrapBondedDistance(r12, box)
        r23 = _wrapBondedDistance(r23, box)
        r34 = _wrapBondedDistance(r34, box)

    c1 = np.cross(r23, r34, axisa=0, axisb=0, axisc=0)
    c2 = np.cross(r12, r23, axisa=0, axisb=0, axisc=0)

    p1 = (r12 * c1).sum(axis=0)
    p1 *= (r23 * r23).sum(axis=0) ** 0.5
    p2 = (c1 * c2).sum(axis=0)

    return -np.arctan2(p1, p2)
