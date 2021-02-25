import networkx as nx
import numpy as np
from moleculekit.tools.moleculechecks import isProteinProtonated


def get_donors_acceptors(mol, exclude_water=True, exclude_backbone=False):
    if not isProteinProtonated(mol):
        raise RuntimeError(
            "The protein seems to not be protonated. You must provide a protonated system for H-bonds to be detected."
        )

    mol_g = nx.Graph()
    mol_g.add_edges_from(mol.bonds)

    donor_elements = ("N", "O")
    acceptor_elements = ("N", "O")
    potential_donors = np.in1d(mol.element, donor_elements)
    potential_acceptors = np.in1d(mol.element, acceptor_elements)

    backbone = mol.atomselect("backbone")
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
        return

    hydrogens = tuple(np.where(mol.element == "H")[0])
    donor_pairs = []
    for pot_do in np.where(potential_donors)[0]:
        neighbours = np.array(list(mol_g.neighbors(pot_do)))
        h_neighbours = neighbours[np.in1d(neighbours, hydrogens)]
        for hn in h_neighbours:
            donor_pairs.append([pot_do, hn])

    acceptors = np.where(potential_acceptors)[0]

    if not len(donor_pairs) or not len(acceptors):
        return
    return np.vstack(donor_pairs).astype(np.uint32), acceptors.astype(np.uint32)
