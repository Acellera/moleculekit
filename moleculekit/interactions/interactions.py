# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import networkx as nx
import numpy as np
from moleculekit.tools.moleculechecks import isProteinProtonated, proteinHasBonds
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)


def get_ligand_props(mol, offset_idx: int = 0):
    props = {
        "donors": [],
        "acceptors": [],
        "rings": [],
        "pos_charged": [],
        "neg_charged": [],
        "aryl_halides": [],
    }
    props["donors"], props["acceptors"] = get_ligand_donors_acceptors(mol)
    props["rings"] = get_ligand_rings(mol)
    props["pos_charged"], props["neg_charged"] = get_ligand_charged(mol)
    props["aryl_halides"] = get_ligand_aryl_halides(mol)

    return offset_ligand_props(props, offset_idx)


def offset_ligand_props(props, offset_idx: int = 0, idx_mapping_fn=None):
    from moleculekit.util import ensurelist

    new_props = {
        "donors": [],
        "acceptors": [],
        "rings": [],
        "pos_charged": [],
        "neg_charged": [],
        "aryl_halides": [],
    }

    if idx_mapping_fn is None:
        idx_mapping_fn = lambda x: x

    offset_idx = ensurelist(offset_idx)

    if len(offset_idx) == 0 and offset_idx[0] == 0:
        return props

    for offidx in offset_idx:
        for k in props:
            if isinstance(props[k], np.ndarray):
                new_props[k].append(
                    idx_mapping_fn(np.array(props[k], dtype=np.uint32)) + offidx
                )
            elif isinstance(props[k], list):
                new_props[k] += [idx_mapping_fn(pp) + offidx for pp in props[k]]

    new_props = {
        "donors": np.vstack(new_props["donors"]),
        "acceptors": np.hstack(new_props["acceptors"]),
        "rings": new_props["rings"],
        "pos_charged": np.hstack(new_props["pos_charged"]),
        "neg_charged": np.hstack(new_props["neg_charged"]),
        "aryl_halides": np.hstack(new_props["aryl_halides"]),
    }
    return new_props


def filter_props(props, exclude_idx=None, include_idx=None):
    """Remove atoms from the properties based on their index"""

    def _filter_func(arr):
        if arr.ndim == 2:
            if include_idx is not None:
                arr = arr[np.all(np.isin(arr, include_idx), axis=1)]
            if exclude_idx is not None:
                arr = arr[~np.any(np.isin(arr, exclude_idx), axis=1)]
        else:
            if include_idx is not None:
                arr = arr[np.isin(arr, include_idx)]
            if exclude_idx is not None:
                arr = arr[~np.isin(arr, exclude_idx)]
        return arr

    for k in props:
        if isinstance(props[k], np.ndarray):
            props[k] = _filter_func(props[k])
        elif isinstance(props[k], list):
            props[k] += [_filter_func(pp) for pp in props[k]]


def get_receptor_props(mol: "Molecule"):
    props = {
        "donors": [],
        "acceptors": [],
        "rings": [],
        "charged": [],
        "aryl_halides": [],
    }
    props["donors"], props["acceptors"] = get_donors_acceptors(
        mol, exclude_water=True, exclude_backbone=False
    )
    props["rings"] = get_receptor_rings(mol, rec_type="both")
    pos_prot, neg_prot = get_protein_charged(mol)
    pos_nucl, neg_nucl = get_nucleic_charged(mol)
    props["pos_charged"] = np.hstack((pos_prot, pos_nucl))
    props["neg_charged"] = np.hstack((neg_prot, neg_nucl))
    props["aryl_halides"] = get_protein_aryl_halides(mol)

    return props


def get_donors_acceptors(mol: "Molecule", exclude_water: bool = True, exclude_backbone: bool = False):
    if not isProteinProtonated(mol):
        raise RuntimeError(
            "The protein seems to not be protonated. You must provide a protonated system for H-bonds to be detected."
        )
    if not proteinHasBonds(mol):
        raise RuntimeError(
            "The protein is missing bonds. Cannot calculate donors/acceptors"
        )

    donor_elements = ("N", "O")
    acceptor_elements = ("N", "O")
    potential_donors = np.isin(mol.element, donor_elements)
    potential_acceptors = np.isin(mol.element, acceptor_elements)

    if not np.any(potential_donors) and not np.any(potential_acceptors):
        logger.warning("Could not find any [N, O] elements in the molecule")
        return [], []

    backbone = mol.atomselect("protein and backbone")
    # Amide nitrogens are only donors
    potential_acceptors[(mol.element == "N") & backbone] = False

    if exclude_water:
        waters = mol.atomselect("water")
        potential_donors = ~waters & potential_donors
        potential_acceptors = ~waters & potential_acceptors

    if exclude_backbone:
        potential_donors = ~backbone & potential_donors
        potential_acceptors = ~backbone & potential_acceptors

    if not np.any(potential_donors):
        logger.warning("Did not find any potential donors")
        return [], []

    if not np.any(potential_acceptors):
        logger.warning("Did not find any potential acceptors")
        return [], []

    hydrogens = tuple(np.where(mol.element == "H")[0])
    donor_pairs = []
    potential_donors = np.where(potential_donors)[0]

    mask1 = np.isin(mol.bonds[:, 0], potential_donors) & np.isin(
        mol.bonds[:, 1], hydrogens
    )
    mask2 = np.isin(mol.bonds[:, 1], potential_donors) & np.isin(
        mol.bonds[:, 0], hydrogens
    )
    donor_pairs1 = mol.bonds[mask1, :]
    donor_pairs2 = mol.bonds[
        mask2, ::-1
    ]  # Invert the order here to have the hyd second
    donor_pairs = np.vstack((donor_pairs1, donor_pairs2))

    acceptors = np.where(potential_acceptors)[0]

    if not len(donor_pairs) or not len(acceptors):
        return [], []
    return donor_pairs.astype(np.uint32), acceptors.astype(np.uint32)


def get_ligand_donors_acceptors(smol, start_idx: int = 0):
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
    import os

    start_idx = int(start_idx)

    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    molFeats = factory.GetFeaturesForMol(smol._mol)

    donor_pairs = []
    acceptors = []
    for feat in molFeats:
        family = feat.GetFamily()
        for idx in feat.GetAtomIds():
            if family == "Acceptor":
                acceptors.append(idx)
            if family == "Donor":
                at = smol._mol.GetAtomWithIdx(idx + start_idx)
                for bond in at.GetBonds():
                    oat = bond.GetOtherAtom(at)
                    if oat.GetSymbol() == "H":
                        donor_pairs.append([idx + start_idx, oat.GetIdx() + start_idx])

    return np.array(donor_pairs, dtype=np.uint32), np.array(acceptors, dtype=np.uint32)


def view_hbonds(mol: "Molecule", hbonds):
    from moleculekit.vmdgraphics import VMDCylinder

    viewname = "mol"
    if mol.viewname is not None:
        viewname = mol.viewname

    for f in range(mol.numFrames):
        mol.viewname = f"{viewname}_frame_{f}"
        mol.view()
        for i in range(hbonds[f].shape[0]):
            end = hbonds[f][i, 2]
            start = hbonds[f][i, 1]
            if start == -1:
                start = hbonds[f][i, 0]

            VMDCylinder(
                mol.coords[start, :, f],
                mol.coords[end, :, f],
                radius=0.1,
            )
    mol.viewname = viewname


def get_protein_rings(mol: "Molecule"):
    return get_receptor_rings(mol, "protein")


def get_nucleic_rings(mol: "Molecule"):
    return get_receptor_rings(mol, "nucleic")


def get_receptor_rings(mol: "Molecule", rec_type: str):
    _prot_aromatics = ["PHE", "HIS", "HID", "HIE", "HIP", "TYR", "TRP"]
    _prot_excluded_atoms = ["N", "CA", "C", "O"]  # , "CB", "OH"]
    _prot_sel = np.isin(mol.resname, _prot_aromatics) & ~np.isin(
        mol.name, _prot_excluded_atoms
    )
    _nucl_sel = mol.atomselect("nucleic and not backbone")
    _both_sel = _prot_sel | _nucl_sel

    sel = {"protein": _prot_sel, "nucleic": _nucl_sel, "both": _both_sel}[rec_type]

    mol2 = mol.copy(sel=sel)
    bonds = mol2._guessBonds()

    graph = nx.Graph()
    graph.add_edges_from(bonds)
    cycles = nx.cycle_basis(graph)

    original_idx = np.where(sel)[0]
    cycles = [np.array(sorted(original_idx[cc]), dtype=np.uint32) for cc in cycles]
    cycles = sorted(cycles, key=lambda x: x[0])

    return cycles


def get_protein_aryl_halides(mol: "Molecule"):
    # TODO: Need to support non-standard residues
    pass


metals = [
    "Fe",
    "Cu",
    "Ni",
    "Mo",
    "Rh",
    "Re",
    "Mn",
    "Mg",
    "Ca",
    "Na",
    "K",
    "Cs",
    "Zn",
    "Se",
]


def get_metal_charged(mol: "Molecule"):
    return np.where(np.isin(mol.element, metals))[0].astype(np.uint32), []


def get_protein_charged(mol: "Molecule"):
    lys_n = (mol.resname == "LYS") & (mol.name == "NZ")
    arg_c = (mol.resname == "ARG") & (mol.name == "CZ")
    hip_c = (mol.resname == "HIP") & (mol.name == "CE1")
    pos = lys_n | arg_c | hip_c

    asp_c = (mol.resname == "ASP") & (mol.name == "CG")
    glu_c = (mol.resname == "GLU") & (mol.name == "CD")
    neg = asp_c | glu_c

    return np.where(pos)[0].astype(np.uint32), np.where(neg)[0].astype(np.uint32)


def get_nucleic_charged(mol: "Molecule"):
    nuc = mol.atomselect("nucleic and backbone and name OP2")
    return np.array([], dtype=np.uint32), np.where(nuc)[0].astype(np.uint32)


def get_ligand_rings(sm, start_idx: int = 0):
    ligandRings = sm._mol.GetRingInfo().AtomRings()
    ligandAtomAromaticRings = []
    for ring in ligandRings:
        aromatics = sum([sm._mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring])
        if aromatics != len(ring):
            continue
        ligandAtomAromaticRings.append(
            np.array(sorted([r + start_idx for r in ring]), dtype=np.uint32)
        )
    ligandAtomAromaticRings = sorted(ligandAtomAromaticRings, key=lambda x: x[0])
    return ligandAtomAromaticRings


def get_ligand_charged(sm, start_idx: int = 0):
    pos = []
    neg = []
    for i in range(sm.numAtoms):
        fc = sm._mol.GetAtomWithIdx(i).GetFormalCharge()
        if fc > 0:
            pos.append(i + start_idx)
        elif fc < 0:
            neg.append(i + start_idx)
    return np.array(pos, dtype=np.uint32), np.array(neg, dtype=np.uint32)


def get_ligand_aryl_halides(sm, start_idx: int = 0):
    import networkx as nx

    halogens = ["Cl", "Br", "I"]
    rings = get_ligand_rings(sm)
    if len(rings) == 0:
        return np.array([], dtype=np.uint32)

    ring_atoms = np.hstack(rings)
    halogens = np.where(np.isin(sm._element, halogens))[0]

    graph = nx.Graph()
    graph.add_edges_from(sm._bonds)

    halides = []
    for hi in halogens:
        neighs = list(graph.neighbors(hi))
        if len(neighs) != 1:
            continue
        if neighs[0] in ring_atoms:
            halides.append([hi + start_idx, neighs[0] + start_idx])

    return np.array(halides, dtype=np.uint32)


def hbonds_calculate(
    mol: "Molecule",
    donors: np.ndarray,
    acceptors: np.ndarray,
    sel1: str | np.ndarray = "all",
    sel2: str | np.ndarray | None = None,
    dist_threshold: float = 2.5,
    angle_threshold: float = 120,
    ignore_hs: bool = False,
):
    """Detect hydrogen bonds in each frame of a Molecule.

    A hydrogen bond is reported when a donor hydrogen lies within `dist_threshold`
    of an acceptor atom and the donor-heavy / hydrogen / acceptor angle is greater
    than `angle_threshold`.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect hydrogen bonds. Its
        ``box`` must have the same number of frames as its ``coords``.
    donors : numpy.ndarray
        An array of shape (n_donors, 2) of donor pairs, where each row holds the
        index of the donor heavy atom and the index of its bonded hydrogen, as
        returned by :func:`get_donors_acceptors` or
        :func:`get_ligand_donors_acceptors`.
    acceptors : numpy.ndarray
        A 1D array of acceptor atom indices.
    sel1 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        first group of atoms taking part in the
        hydrogen bonds. See more `here <https://software.acellera.com/moleculekit/atomselect.html>`__.
    sel2 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        second group of atoms taking part in the
        hydrogen bonds. If None, `sel1` is used for both groups and only
        intra-selection hydrogen bonds are computed.
    dist_threshold : float
        The maximum hydrogen-to-acceptor distance in angstroms for a hydrogen bond.
    angle_threshold : float
        The minimum donor-hydrogen-acceptor angle in degrees for a hydrogen bond.
    ignore_hs : bool
        If True, hydrogens are ignored and only the donor heavy atoms are used,
        with distances measured between donor heavy atoms and acceptors. The
        reported hydrogen index will be -1 in this case.

    Returns
    -------
    hbonds : list of numpy.ndarray
        A list with one entry per frame of `mol`. Each entry is an array of shape
        (n_hbonds, 3) where each row holds the indices of the donor heavy atom, the
        donor hydrogen (or -1 if `ignore_hs` is True) and the acceptor atom.
    """
    from moleculekit.interactions import hbonds

    if mol.box.shape[1] != mol.coords.shape[2]:
        raise RuntimeError("mol.box should have same number of frames as mol.coords")

    if len(donors) == 0 or len(acceptors) == 0:
        return [np.empty((0, 3), dtype=np.int64) for _ in range(mol.numFrames)]

    sel1 = mol.atomselect(sel1).astype(np.uint32).copy()
    if sel2 is None:
        sel2 = sel1.copy()
        intra = True
    else:
        sel2 = mol.atomselect(sel2).astype(np.uint32).copy()
        intra = False

    if len(sel1) != mol.numAtoms or len(sel2) != mol.numAtoms:
        raise RuntimeError(
            "Selections must be boolean of size equal to number of atoms in the molecule"
        )

    # Filter donors and acceptors list if they are not in the selections to reduce calculations
    sel_idx = np.where(sel1 | sel2)[0]
    donors = donors[np.all(np.isin(donors, sel_idx), axis=1)]
    acceptors = acceptors[np.isin(acceptors, sel_idx)]

    if ignore_hs:
        # if we are ignoring hydrogens, reduce the donors list to only heavy atoms
        donors = np.unique(donors[:, 0])[:, None]

    hb = hbonds.calculate(
        donors.astype(np.uint32),
        acceptors.astype(np.uint32),
        mol.coords.astype(np.float32),
        mol.box.astype(np.float32),
        sel1.astype(np.uint32),
        sel2.astype(np.uint32),
        dist_threshold=float(dist_threshold),
        angle_threshold=float(angle_threshold),
        intra=bool(intra),
        ignore_hs=bool(ignore_hs),
    )

    hbond_list = []
    for f in range(mol.numFrames):
        # The cython kernel emits a flat sequence of (donor_heavy, donor_H, acceptor)
        # triples. Reshape to (N, 3) so downstream callers (view_hbonds,
        # waterbridge_calculate, user analysis) can index it as a 2D array.
        hbond_list.append(np.asarray(hb[f], dtype=np.int64).reshape(-1, 3))
    return hbond_list


def waterbridge_calculate(
    mol: "Molecule",
    donors: np.ndarray,
    acceptors: np.ndarray,
    sel1: str | np.ndarray,
    sel2: str | np.ndarray,
    order: int = 1,
    dist_threshold: float = 2.5,
    angle_threshold: float = 120,
    ignore_hs: bool = False,
):
    """Detect water-mediated hydrogen-bond bridges between two selections.

    A water bridge is a chain of hydrogen bonds that connects an atom in `sel1` to
    an atom in `sel2` through one or more intervening water molecules. The chain of
    hydrogen bonds is built by iteratively detecting hydrogen bonds (see
    :func:`hbonds_calculate`) starting from `sel1`, hopping through waters, and a
    network is then searched for paths whose intermediate atoms are all water.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect water bridges. It must
        contain the waters that can mediate the bridges.
    donors : numpy.ndarray
        An array of shape (n_donors, 2) of donor pairs (donor heavy atom index,
        bonded hydrogen index), as returned by :func:`get_donors_acceptors`. The
        waters must not be excluded (i.e. call it with ``exclude_water=False``).
    acceptors : numpy.ndarray
        A 1D array of acceptor atom indices, including water acceptors.
    sel1 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        source group of atoms (one end of the bridge).
        See more `here <https://software.acellera.com/moleculekit/atomselect.html>`__.
    sel2 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        target group of atoms (other end of the bridge).
    order : int
        The maximum number of bridging water molecules allowed between `sel1` and
        `sel2`. ``order=1`` finds bridges through a single water, ``order=2`` finds
        bridges through up to two waters, and so on.
    dist_threshold : float
        The maximum hydrogen-to-acceptor distance in angstroms for each hydrogen
        bond in the bridge.
    angle_threshold : float
        The minimum donor-hydrogen-acceptor angle in degrees for each hydrogen bond
        in the bridge.
    ignore_hs : bool
        If True, hydrogens are ignored and the bridge paths are expressed in terms
        of donor/acceptor heavy atoms only.

    Returns
    -------
    water_bridges : list of list of list
        A list with one entry per frame of `mol`. Each frame entry is a list of
        bridges, where each bridge is a list of atom indices describing the path
        from a `sel1` atom to a `sel2` atom; every intermediate atom in the path
        belongs to a water molecule.
    """
    import networkx as nx

    if len(donors) == 0 or len(acceptors) == 0:
        return [[] for _ in range(mol.numFrames)]

    sel1_b = mol.atomselect(sel1)
    sel2_b = mol.atomselect(sel2)
    water_b = mol.atomselect("water")

    args = {
        "mol": mol,
        "donors": donors,
        "acceptors": acceptors,
        "dist_threshold": dist_threshold,
        "angle_threshold": angle_threshold,
        "ignore_hs": ignore_hs,
    }

    water_goal = water_b | sel2_b
    water_goal_idx = np.where(water_goal)[0]
    water_idx = np.where(water_b)[0]

    # Add one because an order 1 water bridge requires two iterations to find the target
    order += 1
    edges = []
    for f in range(mol.numFrames):
        edges.append([])

    sel1_b_curr = sel1_b.copy()
    for _ in range(order):
        curr_shell = hbonds_calculate(sel1=sel1_b_curr, sel2=water_goal, **args)
        for f in range(mol.numFrames):
            # Only keep interactions which have at least one water
            curr_shell[f] = np.array(curr_shell[f])
            if len(curr_shell[f]) == 0:
                continue
            has_water = np.any(np.isin(curr_shell[f], water_idx), axis=1)
            curr_shell[f] = curr_shell[f][has_water, :]
            # Append the valid edges for this frame
            if ignore_hs:
                edges[f].append(curr_shell[f][:, [0, 2]])
            else:
                edges[f].append(curr_shell[f][:, :2])
                edges[f].append(curr_shell[f][:, 1:])

        # Find which water atoms interacted with sel1_b_curr to use them for the next shell
        curr_shell = [cs for cs in curr_shell if len(cs) > 0]
        if len(curr_shell) == 0:
            break
        shell = np.vstack(curr_shell)[:, [0, 2]]
        sel1_b_curr = np.zeros(mol.numAtoms, dtype=bool)
        interacted = np.unique(shell[np.isin(shell, water_goal_idx)])
        if len(interacted) == 0:
            break
        sel1_b_curr[interacted] = True

    # Create networks and check for shortest paths between source and target
    sel1_idx = np.where(sel1_b)[0]
    sel2_idx = np.where(sel2_b)[0]

    water_bridges = []
    for f in range(mol.numFrames):
        water_bridges.append([])

    for f in range(mol.numFrames):
        if len(edges[f]) == 0:
            continue
        ee = np.vstack(edges[f])
        starts = np.unique(ee[np.isin(ee, sel1_idx)])
        ends = np.unique(ee[np.isin(ee, sel2_idx)])

        # If they never connected skip
        if not (np.any(starts) and np.any(ends)):
            continue

        network = nx.Graph()
        network.add_edges_from(ee)
        # For all start and end indexes (sel1/sel2 can have multiple donors/acceptors)
        for st in starts:
            for en in ends:
                for pp in nx.all_simple_paths(network, source=st, target=en):
                    # Exclude direct source-target hbonds. Probably not needed due to the previous water check
                    if len(pp) < 3:
                        continue
                    # Exclude paths which are not pure water bridges
                    if not np.all(np.isin(pp[1:-1], water_idx)):
                        continue
                    water_bridges[f].append(pp)

    return water_bridges


def pipi_calculate(
    mol: "Molecule",
    rings1,
    rings2,
    dist_threshold1: float = 4.4,
    angle_threshold1_max: float = 30,
    dist_threshold2: float = 5.5,
    angle_threshold2_min: float = 60,
    return_rings: bool = False,
):
    """Detect pi-pi (aromatic ring stacking) interactions in each frame of a Molecule.

    A pi-pi interaction is reported when the distance between two ring centroids and
    the angle between the two ring planes satisfy one of two criteria, which together
    capture both parallel (sandwich/offset-stacked) and perpendicular (T-shaped)
    arrangements:

    - centroid distance below `dist_threshold1` and inter-plane angle below
      `angle_threshold1_max` (near-parallel stacking), or
    - centroid distance below `dist_threshold2` and inter-plane angle above
      `angle_threshold2_min` (near-perpendicular / T-shaped).

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect pi-pi interactions.
    rings1 : list of numpy.ndarray
        The first group of aromatic rings, where each ring is an array of its atom
        indices, as returned by e.g. :func:`get_protein_rings` or
        :func:`get_ligand_rings`.
    rings2 : list of numpy.ndarray
        The second group of aromatic rings, in the same format as `rings1`.
    dist_threshold1 : float
        The maximum centroid-to-centroid distance in angstroms for the
        near-parallel criterion.
    angle_threshold1_max : float
        The maximum inter-plane angle in degrees ([0, 90]) for the near-parallel
        criterion.
    dist_threshold2 : float
        The maximum centroid-to-centroid distance in angstroms for the
        near-perpendicular criterion.
    angle_threshold2_min : float
        The minimum inter-plane angle in degrees ([0, 90]) for the near-perpendicular
        criterion.
    return_rings : bool
        If True, the first returned value contains the ring atom-index arrays for
        each interacting pair instead of the ring indices into `rings1`/`rings2`.

    Returns
    -------
    pp_list : list
        A list with one entry per frame. By default each entry is a list of
        ``[i, j]`` pairs, where ``i`` indexes into `rings1` and ``j`` indexes into
        `rings2`. If `return_rings` is True, each entry is instead a list of
        ``[ring1_atoms, ring2_atoms]`` pairs holding the atom-index arrays of the
        two interacting rings.
    dist_ang_list : list
        A list with one entry per frame, each a list of ``[distance, angle]`` pairs
        (centroid distance in angstroms and inter-plane angle in degrees) aligned
        with `pp_list`.
    """
    from moleculekit.interactions import pipi

    if (
        angle_threshold1_max < 0
        or angle_threshold1_max > 90
        or angle_threshold2_min < 0
        or angle_threshold2_min > 90
    ):
        raise RuntimeError("Values for angles should be [0, 90] degrees")

    if len(rings1) == 0 or len(rings2) == 0:
        return [[] for _ in range(mol.numFrames)], [[] for _ in range(mol.numFrames)]

    ring_atoms = np.hstack((np.hstack(rings1), np.hstack(rings2)))
    ring_starts1 = np.insert(np.cumsum([len(rr) for rr in rings1]), 0, 0)
    ring_starts2 = np.insert(np.cumsum([len(rr) for rr in rings2]), 0, 0)
    ring_starts2 += ring_starts1.max()

    pp, da = pipi.calculate(
        ring_atoms.astype(np.uint32),
        ring_starts1.astype(np.uint32),
        ring_starts2.astype(np.uint32),
        mol.coords,
        mol.box,
        dist_threshold1=dist_threshold1,
        angle_threshold1_max=angle_threshold1_max,
        dist_threshold2=dist_threshold2,
        angle_threshold2_min=angle_threshold2_min,
    )

    pp_list = []
    dist_ang_list = []
    for f in range(mol.numFrames):
        reshaped = [pp[f][i : i + 2] for i in range(0, len(pp[f]), 2)]
        if return_rings:
            pp_list.append([[rings1[pp[0]], rings2[pp[1]]] for pp in reshaped])
        else:
            pp_list.append(reshaped)
        dist_ang_list.append([da[f][i : i + 2] for i in range(0, len(da[f]), 2)])
    return pp_list, dist_ang_list


def saltbridge_calculate(
    mol: "Molecule",
    pos: np.ndarray,
    neg: np.ndarray,
    sel1: str | np.ndarray = "all",
    sel2: str | np.ndarray | None = None,
    threshold: float = 4,
):
    """Detect salt bridges (charged contacts) in each frame of a Molecule.

    A salt bridge is reported when a positively charged atom and a negatively
    charged atom lie within `threshold` of each other. Each reported pair contains
    exactly one positive and one negative atom.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect salt bridges.
    pos : numpy.ndarray
        A 1D array of positively charged atom indices, as returned by e.g.
        :func:`get_protein_charged` or :func:`get_ligand_charged`.
    neg : numpy.ndarray
        A 1D array of negatively charged atom indices.
    sel1 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        first group of atoms taking part in the salt
        bridges. See more `here <https://software.acellera.com/moleculekit/atomselect.html>`__.
    sel2 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        second group of atoms. If None, `sel1` is used
        for both groups.
    threshold : float
        The maximum distance in angstroms between the two charged atoms.

    Returns
    -------
    salt_bridges : list of numpy.ndarray
        A list with one entry per frame of `mol`. Each entry is an array of shape
        (n_bridges, 2) of atom-index pairs forming a salt bridge.
    """
    from moleculekit.distance import calculate_contacts

    if len(pos) == 0 or len(neg) == 0:
        return [[] for _ in range(mol.numFrames)]

    sel1 = mol.atomselect(sel1)
    if sel2 is None:
        sel2 = sel1.copy()
    else:
        sel2 = mol.atomselect(sel2)

    charged = np.zeros(sel1.shape, dtype=bool)
    charged[pos] = True
    charged[neg] = True

    sel1 = sel1 & charged
    sel2 = sel2 & charged

    periodic = "selections" if not np.all(mol.box == 0) else None
    _inter = calculate_contacts(mol, sel1, sel2, periodic, threshold)

    inter = []
    for f in range(mol.numFrames):
        inter.append(_inter[f][np.sum(np.isin(_inter[f], pos), axis=1) == 1])
    return inter


def cationpi_calculate(
    mol: "Molecule",
    rings,
    cations,
    dist_threshold: float = 5,
    angle_threshold_min: float = 60,
    return_rings: bool = False,
):
    """Detect cation-pi interactions in each frame of a Molecule.

    A cation-pi interaction is reported when a cation lies within `dist_threshold`
    of an aromatic ring centroid and the angle between the ring plane and the
    centroid-to-cation vector is above `angle_threshold_min` (i.e. the cation sits
    roughly above the ring face).

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect cation-pi interactions.
    rings : list of numpy.ndarray
        The aromatic rings, where each ring is an array of its atom indices, as
        returned by e.g. :func:`get_protein_rings` or :func:`get_ligand_rings`.
    cations : list or numpy.ndarray
        A 1D collection of cation atom indices, as returned by e.g.
        :func:`get_protein_charged` (positive indices), :func:`get_ligand_charged`
        or :func:`get_metal_charged`.
    dist_threshold : float
        The maximum centroid-to-cation distance in angstroms.
    angle_threshold_min : float
        The minimum angle in degrees ([0, 90]) between the ring plane and the
        centroid-to-cation vector.
    return_rings : bool
        If True, the first returned value contains the ring atom-index array for
        each interacting pair instead of the ring index into `rings`.

    Returns
    -------
    index_list : list
        A list with one entry per frame. By default each entry is a list of
        ``[i, cation_idx]`` pairs, where ``i`` indexes into `rings` and
        ``cation_idx`` is the cation atom index. If `return_rings` is True, each
        entry is instead a list of ``[ring_atoms, cation_idx]`` pairs.
    dist_ang_list : list
        A list with one entry per frame, each a list of ``[distance, angle]`` pairs
        (centroid-to-cation distance in angstroms and angle in degrees) aligned with
        `index_list`.
    """
    from moleculekit.interactions import cationpi

    if angle_threshold_min < 0 or angle_threshold_min > 90:
        raise RuntimeError("Values for angles should be [0, 90] degrees")

    if len(rings) == 0 or len(cations) == 0:
        return [[] for _ in range(mol.numFrames)], [[] for _ in range(mol.numFrames)]

    ring_atoms = np.hstack(rings)
    ring_starts = np.insert(np.cumsum([len(rr) for rr in rings]), 0, 0)

    pp, da = cationpi.calculate(
        ring_atoms.astype(np.uint32),
        ring_starts.astype(np.uint32),
        np.array(cations, dtype=np.uint32),
        mol.coords,
        mol.box,
        dist_threshold=dist_threshold,
        angle_threshold_min=angle_threshold_min,
    )

    index_list = []
    dist_ang_list = []
    for f in range(mol.numFrames):
        reshaped = [pp[f][i : i + 2] for i in range(0, len(pp[f]), 2)]
        if return_rings:
            index_list.append([[rings[pp[0]], pp[1]] for pp in reshaped])
        else:
            index_list.append(reshaped)
        dist_ang_list.append([da[f][i : i + 2] for i in range(0, len(da[f]), 2)])
    return index_list, dist_ang_list


def sigmahole_calculate(
    mol: "Molecule",
    rings,
    halides,
    dist_threshold: float = 4.5,
    angle_threshold_min: float = 60,
    return_rings: bool = False,
):
    """Detect sigma-hole (halogen-pi) interactions in each frame of a Molecule.

    A sigma-hole interaction is reported when an aryl-halide halogen lies within
    `dist_threshold` of an aromatic ring centroid and the angle between the ring
    plane and the centroid-to-halogen vector is above `angle_threshold_min`.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect sigma-hole interactions.
    rings : list of numpy.ndarray
        The aromatic rings, where each ring is an array of its atom indices, as
        returned by e.g. :func:`get_protein_rings` or :func:`get_ligand_rings`.
    halides : numpy.ndarray
        An array of shape (n_halides, 2) of aryl-halide bonds (halogen atom index,
        bonded ring carbon index), as returned by :func:`get_ligand_aryl_halides`.
    dist_threshold : float
        The maximum centroid-to-halogen distance in angstroms.
    angle_threshold_min : float
        The minimum angle in degrees ([0, 90]) between the ring plane and the
        centroid-to-halogen vector.
    return_rings : bool
        If True, the first returned value contains the ring atom-index array for
        each interacting pair instead of the ring index into `rings`.

    Returns
    -------
    index_list : list
        A list with one entry per frame. By default each entry is a list of
        ``[i, halogen_idx]`` pairs, where ``i`` indexes into `rings` and
        ``halogen_idx`` is the halogen atom index. If `return_rings` is True, each
        entry is instead a list of ``[ring_atoms, halogen_idx]`` pairs.
    dist_ang_list : list
        A list with one entry per frame, each a list of ``[distance, angle]`` pairs
        (centroid-to-halogen distance in angstroms and angle in degrees) aligned
        with `index_list`.
    """
    from moleculekit.interactions import sigmahole

    if angle_threshold_min < 0 or angle_threshold_min > 90:
        raise RuntimeError("Values for angles should be [0, 90] degrees")

    if len(rings) == 0 or len(halides) == 0:
        return [[] for _ in range(mol.numFrames)], [[] for _ in range(mol.numFrames)]

    ring_atoms = np.hstack(rings)
    ring_starts = np.insert(np.cumsum([len(rr) for rr in rings]), 0, 0)

    indexes, da = sigmahole.calculate(
        ring_atoms.astype(np.uint32),
        ring_starts.astype(np.uint32),
        np.array(halides, dtype=np.uint32),
        mol.coords,
        mol.box,
        dist_threshold=dist_threshold,
        angle_threshold_min=angle_threshold_min,
    )

    index_list = []
    dist_ang_list = []
    for f in range(mol.numFrames):
        reshaped = [indexes[f][i : i + 2] for i in range(0, len(indexes[f]), 2)]
        if return_rings:
            index_list.append([[rings[pp[0]], pp[1]] for pp in reshaped])
        else:
            index_list.append(reshaped)
        dist_ang_list.append([da[f][i : i + 2] for i in range(0, len(da[f]), 2)])
    return index_list, dist_ang_list


def hydrophobic_calculate(
    mol: "Molecule",
    sel1: str | np.ndarray,
    sel2: str | np.ndarray,
    dist_threshold: float = 4.0,
):
    """Detect hydrophobic contacts in each frame of a Molecule.

    A hydrophobic contact is reported when a carbon atom in `sel1` lies within
    `dist_threshold` of a carbon atom in `sel2`. Both selections are intersected
    with the carbon atoms of the molecule before computing contacts.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect hydrophobic contacts.
    sel1 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        first group of atoms. See more
        `here <https://software.acellera.com/moleculekit/atomselect.html>`__.
    sel2 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        second group of atoms.
    dist_threshold : float
        The maximum carbon-to-carbon distance in angstroms.

    Returns
    -------
    contacts : list of numpy.ndarray
        A list with one entry per frame of `mol`. Each entry is an array of shape
        (n_contacts, 2) of carbon atom-index pairs in contact.
    """
    from moleculekit.distance import calculate_contacts

    # TODO: Maybe check for ligand hydrophobic atoms since we can with rdkit?

    carbons = mol.element == "C"
    sel1 = mol.atomselect(sel1)
    sel1 &= carbons
    sel2 = mol.atomselect(sel2)
    sel2 &= carbons

    periodic = "selections" if not np.all(mol.box == 0) else None
    return calculate_contacts(mol, sel1, sel2, periodic, dist_threshold)


def metal_coordination_calculate(
    mol: "Molecule",
    sel1: str | np.ndarray,
    sel2: str | np.ndarray,
    dist_threshold: float = 3.5,
):
    """Detect metal coordination contacts in each frame of a Molecule.

    A metal coordination contact is reported when a metal atom lies within
    `dist_threshold` of a coordinating atom (one of N, O, S or a halogen). Contacts
    are detected in both directions: metals in `sel1` against coordinating atoms in
    `sel2`, and coordinating atoms in `sel1` against metals in `sel2`.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The Molecule (one or more frames) in which to detect metal coordination.
    sel1 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        first group of atoms. See more
        `here <https://software.acellera.com/moleculekit/atomselect.html>`__.
    sel2 : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array for the
        second group of atoms.
    dist_threshold : float
        The maximum metal-to-coordinating-atom distance in angstroms.

    Returns
    -------
    contacts : list of numpy.ndarray
        A list with one entry per frame of `mol`. Each entry is an array of shape
        (n_contacts, 2) of atom-index pairs in coordination contact.
    """
    from moleculekit.distance import calculate_contacts
    from moleculekit.periodictable import METAL_ELEMENTS

    # See also:
    # https://chem.libretexts.org/Bookshelves/General_Chemistry/Chemistry_(OpenSTAX)/19%3A_Transition_Metals_and_Coordination_Chemistry/19.2%3A_Coordination_Chemistry_of_Transition_Metals
    metals = sorted(METAL_ELEMENTS)
    coord_lig_elems = ["N", "O", "Cl", "F", "Br", "I", "CL", "BR", "S"]

    sel1 = mol.atomselect(sel1)
    sel2 = mol.atomselect(sel2)

    sel1_c = sel1 & np.isin(mol.element, metals)
    sel2_c = sel2 & np.isin(mol.element, coord_lig_elems)
    periodic = None if np.all(mol.box == 0) else "selections"
    inter1 = calculate_contacts(mol, sel1_c, sel2_c, periodic, dist_threshold)

    sel1_c = sel1 & np.isin(mol.element, coord_lig_elems)
    sel2_c = sel2 & np.isin(mol.element, metals)
    periodic = None if np.all(mol.box == 0) else "selections"
    inter2 = calculate_contacts(mol, sel1_c, sel2_c, periodic, dist_threshold)

    inter = [np.vstack((inter1[f], inter2[f])) for f in range(mol.numFrames)]

    return inter
