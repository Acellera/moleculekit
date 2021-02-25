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


def hbonds_calculate(
    mol,
    donors,
    acceptors,
    sel1="all",
    sel2=None,
    dist_threshold=2.5,
    angle_threshold=120,
):
    from moleculekit.interactions import hbonds

    sel1 = mol.atomselect(sel1).astype(np.uint32).copy()
    if sel2 is None:
        sel2 = sel1.copy()
    else:
        sel2 = mol.atomselect(sel2).astype(np.uint32).copy()

    hb = hbonds.calculate(donors, acceptors, mol.coords, mol.box, sel1, sel2, 2.5, 120)

    hbond_list = []
    for f in range(mol.numFrames):
        hbond_list.append(np.array(hb[f]).reshape(-1, 3))
    return hbond_list


def view_hbonds(mol, hbonds):
    from moleculekit.vmdgraphics import VMDCylinder

    viewname = "mol"
    if mol.viewname is not None:
        viewname = mol.viewname.copy()

    for f in range(mol.numFrames):
        mol.viewname = f"{viewname}_frame_{f}"
        mol.view()
        for i in range(hbonds[f].shape[0]):
            VMDCylinder(
                mol.coords[hbonds[f][i, 0], :, f],
                mol.coords[hbonds[f][i, 2], :, f],
                radius=0.1,
            )
    mol.viewname = viewname


import unittest


class _TestInteractions(unittest.TestCase):
    def test_hbonds(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.tools.preparation import proteinPrepare
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        prot = os.path.join(home(dataDir="test-interactions"), "3PTB_prepared.pdb")
        lig = os.path.join(home(dataDir="test-interactions"), "BEN.sdf")

        mol = Molecule(prot)
        lig = SmallMol(lig).toMolecule()
        mol.append(lig)

        # TODO: Add check if bonds exist in molecule! Add it to moleculechecks
        mol.bonds = mol._guessBonds()
        mol.coords = np.tile(mol.coords, (1, 1, 2)).copy()  # Fake second frame
        donors, acceptors = get_donors_acceptors(
            mol, exclude_water=True, exclude_backbone=False
        )

        hb = hbonds_calculate(mol, donors, acceptors, "protein", "resname BEN")
        assert len(hb) == 2
        ref = np.array([[3229, 3236, 2472], [3230, 3237, 2473], [3230, 3238, 2483]])
        assert np.array_equal(hb[0], ref) and np.array_equal(hb[1], ref)

        hb = hbonds_calculate(mol, donors, acceptors, "all")
        assert len(hb) == 2
        assert hb[0].shape == (179, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)