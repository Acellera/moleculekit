# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np


def _pp_measure_fit(P, Q):
    """
    WARNING: ASSUMES CENTERED COORDINATES!!!!

    PP_MEASURE_FIT - molecule alignment function.
    For documentation see http://en.wikipedia.org/wiki/Kabsch_algorithm
    the Kabsch algorithm is a method for calculating the optimal
    rotation matrix that minimizes the RMSD (root mean squared deviation)
    between two paired sets of points
    """
    covariance = np.dot(P.T, Q)

    (V, S, W) = np.linalg.svd(covariance)
    W = W.T

    E0 = np.sum(P * P) + np.sum(Q * Q)
    RMSD = E0 - (2 * np.sum(S.ravel()))
    RMSD = np.sqrt(np.abs(RMSD / P.shape[0]))

    d = np.sign(np.linalg.det(W) * np.linalg.det(V))
    z = np.eye(3).astype(P.dtype)
    z[2, 2] = d
    U = np.dot(np.dot(W, z), V.T)
    return U, RMSD


def _pp_align(
    coords, refcoords, sel, refsel, frames, refframe, matchingframes, inplace=False
):
    if not inplace:
        newcoords = coords.copy()
    else:
        newcoords = coords

    for f in frames:
        P = coords[sel, :, f]
        if matchingframes:
            Q = refcoords[refsel, :, f]
        else:
            Q = refcoords[refsel, :, refframe]
        all1 = coords[:, :, f]

        centroidP = P.mean(axis=0)
        centroidQ = Q.mean(axis=0)

        rot, _ = _pp_measure_fit(P - centroidP, Q - centroidQ)

        all1 = all1 - centroidP
        # Rotating mol
        all1 = np.dot(all1, rot.T)
        # Translating to centroid of refmol
        all1 = all1 + centroidQ
        newcoords[:, :, f] = all1

    if not inplace:
        return newcoords


def molTMscore(mol, ref, molsel="protein", refsel="protein"):
    return molTMalign(mol, ref, molsel, refsel, return_alignments=False)


def molTMalign(
    mol,
    ref,
    molsel="protein",
    refsel="protein",
    return_alignments=True,
    frames=None,
    matchingframes=False,
):
    """Calculates the TMscore between two protein Molecules

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A Molecule containing a single or multiple frames
    ref : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A reference Molecule containing a single frame. Will automatically keep only ref.frame.
    molsel : str
        Atomselect string for which atoms of `mol` to calculate TMScore
    refsel : str
        Atomselect string for which atoms of `ref` to calculate TMScore
    return_alignments : bool
        If True it will return the aligned structures of mol and the transformation matrices used to produce them
    frames : list
        A list of frames of mol to align to ref. If None it will align all frames.
    matchingframes : bool
        If set to True it will align the selected frames of this molecule to the corresponding frames of the refmol.
        This requires both molecules to have the same number of frames.

    Returns
    -------
    tmscore : numpy.ndarray
        TM score (if normalized by length of ref) for each frame in mol
    rmsd : numpy.ndarray
        RMSD only OF COMMON RESIDUES for all frames. This is not the same as a full protein RMSD!!!
    nali : numpy.ndarray
        Number of aligned residues for each frame in mol
    alignments : list of Molecules
        Each frame of `mol` aligned to `ref`
    transformation : list of numpy.ndarray
        Contains the transformation for each frame of mol to align to ref. The first element is the rotation
        and the second is the translation. Look at examples on how to manually produce the aligned structure.

    Examples
    --------
    >>> tmscore, rmsd, nali, alignments, transformation = molTMalign(mol, ref)

    To manually generate the aligned structure for the first frame, first rotate, then translate
    >>> mol.rotateBy(transformation[0][0])
    >>> mol.moveBy(transformation[0][1])
    """
    from moleculekit.tmalign import tmalign

    if matchingframes and mol.numFrames != ref.numFrames:
        raise RuntimeError(
            "This molecule and the reference molecule need the same number or frames to use the matchinframes option."
        )
    if frames is None:
        frames = range(mol.numFrames)

    molsel = mol.atomselect(molsel)
    refsel = ref.atomselect(refsel)

    if molsel.sum() == 0:
        raise RuntimeError("No atoms in `molsel`")
    if refsel.sum() == 0:
        raise RuntimeError("No atoms in `refsel`")

    seqx = mol.sequence(noseg=True, sel=molsel, _logger=False)["protein"].encode(
        "UTF-8"
    )
    seqy = ref.sequence(noseg=True, sel=refsel, _logger=False)["protein"].encode(
        "UTF-8"
    )

    if len(seqx) == 0:
        raise RuntimeError(
            f"No protein sequence found in `mol` for selection '{molsel}'"
        )
    if len(seqy) == 0:
        raise RuntimeError(
            f"No protein sequence found in `ref` for selection '{refsel}'"
        )

    # Transpose to have fastest axis as last
    if matchingframes:
        TM1 = []
        rmsd = []
        nali = []
        t0 = []
        u0 = []
        for f in frames:
            coords1 = np.transpose(
                mol.coords[molsel, :, f].astype(np.float64), (2, 0, 1)
            ).copy()
            coords2 = ref.coords[refsel, :, f].astype(np.float64).copy()
            res = tmalign(coords1, coords2, seqx, seqy)
            TM1.append(res[0][0])
            rmsd.append(res[1][0])
            nali.append(res[2][0])
            t0.append(res[3][0])
            u0.append(res[4][0])
    else:
        coords1 = np.transpose(
            mol.coords[molsel][:, :, frames].astype(np.float64), (2, 0, 1)
        ).copy()
        coords2 = ref.coords[refsel, :, ref.frame].astype(np.float64).copy()
        TM1, rmsd, nali, t0, u0 = tmalign(coords1, coords2, seqx, seqy)

    if return_alignments:
        transformation = []
        coords = []
        for i in range(len(u0)):
            rot = np.array(u0[i]).reshape(3, 3)
            trans = np.array(t0[i])
            transformation.append((rot, trans))
            newcoords = np.dot(mol.coords[:, :, i], np.transpose(rot)) + trans
            coords.append(newcoords[:, :, None])
        coords = np.concatenate(coords, axis=2).astype(np.float32).copy()

        return np.array(TM1), np.array(rmsd), np.array(nali), coords, transformation
    else:
        return np.array(TM1), np.array(rmsd), np.array(nali)
