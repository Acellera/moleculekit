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


def _find_optimal_chain_mapping(mol, ref, molsel, refsel, mol_frame, ref_frame):
    """Find optimal chain-to-chain mapping using greedy RMSD matching.

    Computes pairwise Kabsch RMSD between all chain pairs on CA atoms, then
    greedily assigns chains starting from the best (lowest RMSD) pair.

    Parameters
    ----------
    mol, ref : Molecule objects
    molsel, refsel : np.ndarray (boolean masks, already resolved from atomselect)
    mol_frame, ref_frame : int

    Returns
    -------
    (mapping, mol_chains) or None
        mapping: dict mapping mol_chain_id -> ref_chain_id
        mol_chains: mol chain IDs in file order
        Returns None if mapping is unnecessary (single chain or fewer).
    """
    mol_prot = mol.atomselect("protein")
    ref_prot = ref.atomselect("protein")

    mol_ca = mol.name == "CA"
    ref_ca = ref.name == "CA"

    mol_eff = molsel & mol_prot
    ref_eff = refsel & ref_prot

    # Get unique chains in file order
    def _unique_ordered(arr):
        _, first_idx = np.unique(arr, return_index=True)
        order = np.argsort(first_idx)
        return np.unique(arr)[order]

    mol_chains = _unique_ordered(mol.chain[mol_eff])
    ref_chains = _unique_ordered(ref.chain[ref_eff])

    if len(mol_chains) <= 1 or len(ref_chains) <= 1:
        return None

    # Build per-chain CA coordinates
    mol_ca_coords = {}
    for ch in mol_chains:
        mask = mol_eff & mol_ca & (mol.chain == ch)
        mol_ca_coords[ch] = mol.coords[mask, :, mol_frame].astype(np.float64)

    ref_ca_coords = {}
    for ch in ref_chains:
        mask = ref_eff & ref_ca & (ref.chain == ch)
        ref_ca_coords[ch] = ref.coords[mask, :, ref_frame].astype(np.float64)

    # Compute pairwise RMSD for all chain pairs
    pairs = []
    for mc in mol_chains:
        P_full = mol_ca_coords[mc]
        if P_full.shape[0] == 0:
            continue
        for rc in ref_chains:
            Q_full = ref_ca_coords[rc]
            if Q_full.shape[0] == 0:
                continue
            n_min = min(P_full.shape[0], Q_full.shape[0])
            n_max = max(P_full.shape[0], Q_full.shape[0])
            if n_min / n_max < 0.5:
                continue
            P = P_full[:n_min]
            Q = Q_full[:n_min]
            _, rmsd = _pp_measure_fit(P - P.mean(axis=0), Q - Q.mean(axis=0))
            pairs.append((rmsd, mc, rc))

    # Greedy assignment: best RMSD pair first, skip already-used chains
    pairs.sort()
    mapping = {}
    used_mol = set()
    used_ref = set()
    for rmsd, mc, rc in pairs:
        if mc in used_mol or rc in used_ref:
            continue
        mapping[mc] = rc
        used_mol.add(mc)
        used_ref.add(rc)

    if not mapping:
        return None

    return mapping, mol_chains


def _reorder_ref_atoms(ref, refsel, mapping, mol_chains):
    """Reorder ref's selected atoms so chains appear in mol's chain order.

    Parameters
    ----------
    ref : Molecule
    refsel : np.ndarray (boolean mask)
    mapping : dict (mol_chain -> ref_chain)
    mol_chains : array of mol chain IDs in file order

    Returns
    -------
    np.ndarray : integer index array of ref atoms in reordered chain order
    """
    reordered = []
    used_ref_chains = set()

    for mc in mol_chains:
        if mc in mapping:
            rc = mapping[mc]
            used_ref_chains.add(rc)
            indices = np.where(refsel & (ref.chain == rc))[0]
            reordered.append(indices)

    # Append unmatched ref chains
    all_ref_indices = np.where(refsel)[0]
    matched_set = set(np.concatenate(reordered)) if reordered else set()
    unmatched = np.array([i for i in all_ref_indices if i not in matched_set])
    if len(unmatched) > 0:
        reordered.append(unmatched)

    return np.concatenate(reordered)


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

    # Find optimal chain mapping and reorder ref atoms to match mol's chain order
    result = _find_optimal_chain_mapping(
        mol, ref, molsel, refsel, mol_frame=frames[0], ref_frame=ref.frame
    )
    if result is not None:
        mapping, mol_chains = result
        ref_indices = _reorder_ref_atoms(ref, refsel, mapping, mol_chains)
        # Build reordered ref sequence from per-chain sequences in mol's chain order
        ref_seq_by_chain = ref.getSequence(
            dict_key="chain", sel=refsel, _logger=False
        )
        seqy = "".join(
            ref_seq_by_chain[mapping[mc]] for mc in mol_chains if mc in mapping
        )
        # Add sequences from unmatched ref chains
        matched_ref_chains = set(mapping.values())
        ref_prot = ref.atomselect("protein")
        all_ref_chains_arr = ref.chain[refsel & ref_prot]
        if len(all_ref_chains_arr):
            _, first_idx = np.unique(all_ref_chains_arr, return_index=True)
            all_ref_chains = np.unique(all_ref_chains_arr)[np.argsort(first_idx)]
            for rc in all_ref_chains:
                if rc not in matched_ref_chains and rc in ref_seq_by_chain:
                    seqy += ref_seq_by_chain[rc]
        seqy = seqy.encode("UTF-8")
    else:
        ref_indices = None
        seqy = ref.getSequence(dict_key=None, sel=refsel, _logger=False)[
            "protein"
        ].encode("UTF-8")

    seqx = mol.getSequence(dict_key=None, sel=molsel, _logger=False)["protein"].encode(
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

    # Select ref coordinates using reordered indices or original mask
    ref_coord_sel = ref_indices if ref_indices is not None else refsel

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
            coords2 = ref.coords[ref_coord_sel, :, f].astype(np.float64).copy()
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
        coords2 = ref.coords[ref_coord_sel, :, ref.frame].astype(np.float64).copy()
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
