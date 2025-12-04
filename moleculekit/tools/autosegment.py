# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.molecule import Molecule
import string
import numpy as np
import logging


logger = logging.getLogger(__name__)


CHAIN_ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
SEGID_ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)


def autoSegment(
    mol: Molecule,
    sel="all",
    basename="P",
    spatial=True,
    spatialgap=4.0,
    fields=("segid",),
    field=None,
    _logger=True,
):
    """Detects resid gaps in a selection and assigns incrementing segid to each fragment

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The Molecule object
    sel : str
        Atom selection string on which to check for gaps.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    basename : str
        The basename for segment ids. For example if given 'P' it will name the segments 'P1', 'P2', ...
    spatial : bool
        Only considers a discontinuity in resid as a gap if no atoms of the two residues have distance less than `spatialgap` Angstrom
    spatialgap : float
        The size of a spatial gap which validates a discontinuity (A)
    fields : list
        Fields in which to set the segments. Must be a combination of "chain", "segid" or only one of them.

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A new Molecule object with modified segids

    Example
    -------
    >>> newmol = autoSegment(mol, "chain B", "P", fields=("chain", "segid"))
    """
    from moleculekit.residues import WATER_RESIDUE_NAMES
    from moleculekit.distance import cdist

    # Backwards‑compatible handling of the deprecated ``field`` argument
    # (mirrors the behaviour of :func:`autoSegment`).
    if field is not None and isinstance(field, str):
        if field == "both":
            fields = ("chain", "segid")
        else:
            fields = (field,)

    # Work on a copy so that the original molecule remains untouched
    mol = mol.copy()

    # Letters to be used for chains/segments, if free: 0123456789abcd...ABCD...,
    # minus chain/segid symbols already present outside the selection.
    sel_mask = mol.atomselect(sel)
    sel_idx = np.where(sel_mask)[0]

    if len(sel_idx) == 0:
        return mol

    # ``residue_idx`` is a list of index arrays, one per residue, in order.
    # ``uq_segments`` will hold the segment index assigned to each atom
    # (initially -1 for all atoms).
    _, residue_idx = mol.getResidues(sel=sel, return_idx=True)
    residue_idx = [sel_idx[idx] for idx in residue_idx]
    uq_segments = np.full(mol.numAtoms, -1)
    seg_idx = 0  # current segment counter
    water_resid = 0  # running resid number for water molecules
    for i, idx in enumerate(residue_idx):
        ii = idx[0]  # representative atom index for the current residue
        is_water = mol.resname[ii] in WATER_RESIDUE_NAMES
        curr_res = (
            f"{mol.resname[ii]}:{mol.resid[ii]}{mol.insertion[ii]}:{mol.chain[ii]}"
        )
        if is_water:
            # Water residues get sequential ``resid`` values (0‑9999). Once we
            # exceed that range we start a new segment and reset the counter.
            if water_resid > 9999:
                # Ran out of water resids, create new segment
                seg_idx += 1
                water_resid = 0
                mol.resid[idx] = water_resid
                uq_segments[idx] = seg_idx
                continue

            mol.resid[idx] = water_resid
            water_resid += 2

        # Determine the previous residue (if any) to decide whether we need to
        # open a new segment based on changes in water/non‑water type or gaps.
        prev_idx = None
        prev_is_water = None
        if i > 0:
            prev_idx = residue_idx[i - 1]
            prev_is_water = mol.resname[prev_idx[0]] in WATER_RESIDUE_NAMES
            prev_res = f"{mol.resname[prev_idx[0]]}:{mol.resid[prev_idx[0]]}{mol.insertion[prev_idx[0]]}:{mol.chain[prev_idx[0]]}"

        if prev_idx is not None:
            if is_water and not prev_is_water:
                logger.info(
                    f"Water appears after non-water residue {prev_res}. Creating new chain."
                )
                seg_idx += 1
            elif not is_water and prev_is_water:
                logger.info(
                    f"Non-water residue {curr_res} appears after water. Creating new chain."
                )
                seg_idx += 1
            elif not is_water and mol.resid[idx[0]] != mol.resid[prev_idx[0]] + 1:
                # Residue gap in sequence for non‑water; optionally validate it
                # by checking spatial distance between the two residues.
                if spatial:
                    # Check smallest distance between atoms in the two residues
                    dist = cdist(
                        mol.coords[idx, :, 0], mol.coords[prev_idx, :, 0]
                    ).min()
                    if dist > spatialgap:
                        logger.info(
                            f"Residue gap between {prev_res} and {curr_res} with min distance {dist:.1f}A > {spatialgap}A. Creating new chain."
                        )
                        seg_idx += 1
                else:
                    logger.info(
                        f"Residue gap between {prev_res} and {curr_res}. Creating new chain."
                    )
                    seg_idx += 1
        # Assign the current segment index to all atoms of the residue
        uq_segments[idx] = seg_idx

    # Distinct water segments
    water_mask = np.isin(mol.resname, list(WATER_RESIDUE_NAMES))
    water_segments = set(uq_segments[water_mask]) - {-1}

    # Reset chain/segment identifiers for the selected atoms so that we can
    # reassign them consistently from scratch.
    if "chain" in fields:
        mol.chain[sel_mask] = ""
    if "segid" in fields:
        mol.segid[sel_mask] = ""

    # Generator that produces unique segment IDs.
    # It first uses the provided basename, then iterates over the allowed
    # `SEGID_ALPHABET`, and for each base it appends an integer index.
    def _segid_gen(basename):
        for base in [basename] + SEGID_ALPHABET:
            for i in range(1000):
                yield f"{base}{i}"

    # Single shared generator instance from which we draw new segids.
    segid_gen = _segid_gen(basename)

    # Find which chains existed before this function was called
    preexisting_chains = set(mol.chain) - {""}

    # Decide the next `(chain, segid)` pair to use.
    # - `water=True` enforces water chains to preferably use chain "W".
    # - It avoids reusing chain IDs already present in `mol.chain`.
    # - It also guarantees that the segid is unique in `mol.segid`.
    def _get_next_segment_name(mol, water, segid_gen, seg):
        # All chains that are still available for assignment
        available_chains = [x for x in CHAIN_ALPHABET if x not in preexisting_chains]
        chain = available_chains[seg % len(available_chains)]

        # Find the next unused segment identifier
        segid = next(segid_gen)
        while segid in mol.segid:
            segid = next(segid_gen)

        if water and "W" in available_chains:
            return "W", segid
        else:
            return chain, segid

    # Loop over each unique segment index and assign consistent chain/segid
    # values to all atoms that belong to that segment.
    for seg in range(max(uq_segments) + 1):
        ch, sg = _get_next_segment_name(mol, seg in water_segments, segid_gen, seg)
        if "chain" in fields:
            mol.chain[uq_segments == seg] = ch
        if "segid" in fields:
            mol.segid[uq_segments == seg] = sg

    return mol


def autoSegment2(
    mol,
    sel="(protein or resname ACE NME)",
    basename="P",
    fields=("segid",),
    residgaps=False,
    residgaptol=1,
    chaingaps=True,
    _logger=True,
):
    """Detects bonded segments in a selection and assigns incrementing segid to each segment

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The Molecule object
    sel : str
        Atom selection string on which to check for gaps.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    basename : str
        The basename for segment ids. For example if given 'P' it will name the segments 'P1', 'P2', ...
    fields : tuple of strings
        Field to fix. Can be "segid" (default) or any other Molecule field or combinations thereof.
    residgaps : bool
        Set to True to consider gaps in resids as structural gaps. Set to False to ignore resids
    residgaptol : int
        Above what resid difference is considered a gap. I.e. with residgaptol 1, 235-233 = 2 > 1 hence is a gap. We set
        default to 2 because in many PDBs single residues are missing in the proteins without any gaps.
    chaingaps : bool
        Set to True to consider changes in chains as structural gaps. Set to False to ignore chains

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A new Molecule object with modified segids

    Example
    -------
    >>> newmol = autoSegment2(mol)
    """
    import networkx as nx

    if isinstance(fields, str):
        fields = (fields,)

    orig_sel = sel
    sel = f"({sel}) and backbone or (resname NME ACE and name N C O CH3)"  # Looking for bonds only over the backbone of the protein
    # Keep the original atom indexes to map from submol to mol
    idx = mol.atomselect(sel, indexes=True)
    # We filter out everything not on the backbone to calculate only those bonds
    submol = mol.copy()
    submol.filter(sel, _logger=False)
    bonds = submol._getBonds()  # Calculate both file and guessed bonds

    if residgaps:
        # Remove bonds between residues without continuous resids
        bondresiddiff = np.abs(submol.resid[bonds[:, 0]] - submol.resid[bonds[:, 1]])
        bonds = bonds[bondresiddiff <= residgaptol, :]
    else:
        # Warning about bonds bonding non-continuous resids
        bondresiddiff = np.abs(submol.resid[bonds[:, 0]] - submol.resid[bonds[:, 1]])
        if _logger and np.any(bondresiddiff > 1):
            for i in np.where(bondresiddiff > residgaptol)[0]:
                logger.warning(
                    "Bonds found between resid gaps: resid {} and {}".format(
                        submol.resid[bonds[i, 0]], submol.resid[bonds[i, 1]]
                    )
                )
    if chaingaps:
        # Remove bonds between residues without same chain
        bondsamechain = submol.chain[bonds[:, 0]] == submol.chain[bonds[:, 1]]
        bonds = bonds[bondsamechain, :]
    else:
        # Warning about bonds bonding different chains
        bondsamechain = submol.chain[bonds[:, 0]] == submol.chain[bonds[:, 1]]
        if _logger and np.any(bondsamechain == False):
            for i in np.where(bondsamechain == False)[0]:
                logger.warning(
                    "Bonds found between chain gaps: resid {}/{} and {}/{}".format(
                        submol.resid[bonds[i, 0]],
                        submol.chain[bonds[i, 0]],
                        submol.resid[bonds[i, 1]],
                        submol.chain[bonds[i, 1]],
                    )
                )

    # Calculate connected components using the bonds
    submol.bonds = bonds  # Restore the modified bonds to the submol
    submol.bondtype = np.array([], dtype=object)
    components = list(nx.connected_components(submol.toGraph(fields=[])))
    numcomp = len(components)

    # Letters to be used for chains, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    sel_mask = mol.atomselect(orig_sel)
    used_chains = set(mol.chain[~sel_mask])
    available_chains = [x for x in CHAIN_ALPHABET if x not in used_chains]
    used_segids = set([x[0] for x in mol.segid[~sel_mask] if x != ""])
    available_segids = [x for x in [basename] + SEGID_ALPHABET if x not in used_segids]
    basename = available_segids[0]

    mol = mol.copy()
    prevsegres = None
    for i in range(numcomp):  # For each connected component / segment
        segid = basename + str(i)
        backboneSegIdx = idx[list(components[i])]  # The backbone atoms of the segment
        segres = mol.atomselect(
            f"same residue as index {' '.join(map(str, backboneSegIdx))}"
        )  # Get whole residues

        # Warning about separating segments with continuous resids
        if (
            _logger
            and i > 0
            and (np.min(mol.resid[segres]) - np.max(mol.resid[prevsegres])) == 1
        ):
            logger.warning(
                f"Separated segments {basename + str(i - 1)} and {segid}, despite continuous resids, due to lack of bonding."
            )

        # Add the new segment ID to all fields the user specified
        for f in fields:
            if f != "chain":
                mol.__dict__[f][segres] = segid  # Assign the segid to the correct atoms
            else:
                mol.__dict__[f][segres] = available_chains[i % len(available_chains)]
        if _logger:
            logger.info(
                f"Created segment {segid} between resid {mol.resid[segres].min()} and {mol.resid[segres].max()}."
            )
        prevsegres = segres  # Store old segment atom indexes for the warning about continuous resids

    return mol
