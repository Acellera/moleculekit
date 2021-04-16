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
                mol.coords[hbonds[f][i, 1], :, f],
                mol.coords[hbonds[f][i, 2], :, f],
                radius=0.1,
            )
    mol.viewname = viewname


def get_ligand_rings(sm):
    ligandRings = sm._mol.GetRingInfo().AtomRings()
    ligandAtomAromaticRings = []
    for r in ligandRings:
        aromatics = sum([sm._mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in r])
        if aromatics != len(r):
            continue
        ligandAtomAromaticRings.append(r)
    return ligandAtomAromaticRings


def get_protein_rings(mol):
    from moleculekit.util import sequenceID
    import networkx as nx

    _aromatics = ["PHE", "HIS", "HID", "HIE", "HIP", "TYR", "TRP"]
    _excluded_atoms = ["N", "CA", "C", "O"]  # , "CB", "OH"]

    arom_res_atoms = np.isin(mol.resname, _aromatics) & ~np.isin(
        mol.name, _excluded_atoms
    )
    mol2 = mol.copy()
    mol2.filter(arom_res_atoms, _logger=False)
    bonds = mol2._guessBonds()

    graph = nx.Graph()
    graph.add_edges_from(bonds)
    cycles = nx.cycle_basis(graph)

    original_idx = np.where(arom_res_atoms)[0]
    cycles = [tuple(original_idx[cc]) for cc in cycles]

    return cycles


def pipi_calculate(
    mol,
    rings1,
    rings2,
    dist_threshold1=4.4,
    angle_threshold1_min=0,
    angle_threshold1_max=30,
    dist_threshold2=5.5,
    angle_threshold2_min=60,
    angle_threshold2_max=120,
):
    from moleculekit.interactions import pipi

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
        angle_threshold1_min=angle_threshold1_min,
        angle_threshold1_max=angle_threshold1_max,
        dist_threshold2=dist_threshold2,
        angle_threshold2_min=angle_threshold2_min,
        angle_threshold2_max=angle_threshold2_max,
    )

    pp_list = []
    dist_ang_list = []
    for f in range(mol.numFrames):
        pp_list.append(np.array(pp[f]).reshape(-1, 2))
        dist_ang_list.append(np.array(da[f]).reshape(-1, 2))
    return pp_list, dist_ang_list


def get_charged(mol):
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
    metal = np.isin(mol.element, metals)
    lys_n = (mol.resname == "LYS") & (mol.name == "NZ")
    arg_c = (mol.resname == "ARG") & (mol.name == "CZ")
    hip_c = (mol.resname == "HIP") & (mol.name == "CE1")
    pos = metal | lys_n | arg_c | hip_c

    asp_c = (mol.resname == "ASP") & (mol.name == "CG")
    glu_c = (mol.resname == "GLU") & (mol.name == "CD")
    neg = asp_c | glu_c
    return list(np.where(pos)[0]), list(np.where(neg)[0])


def get_ligand_charged(sm):
    pos = []
    neg = []
    for i in range(sm.numAtoms):
        fc = sm._mol.GetAtomWithIdx(i).GetFormalCharge()
        if fc > 0:
            pos.append(i)
        elif fc < 0:
            neg.append(i)
    return pos, neg


def saltbridge_calculate(mol, pos, neg, sel1="all", sel2=None):
    from moleculekit.projections.util import pp_calcDistances

    sel1 = mol.atomselect(sel1).astype(np.uint32).copy()
    if sel2 is None:
        sel2 = sel1.copy()
    else:
        sel2 = mol.atomselect(sel2).astype(np.uint32).copy()

    charged = np.zeros(sel1.shape, dtype=bool)
    charged[pos] = True
    charged[neg] = True

    periodic = "selections" if not np.all(mol.box == 0) else None
    dists = pp_calcDistances(
        mol, sel1 & charged, sel2 & charged, periodic, metric="distances"
    )

    from IPython.core.debugger import set_trace

    set_trace()

    salt_bridge_list = []

    return salt_bridge_list


import unittest


class _TestInteractions(unittest.TestCase):
    def test_hbonds(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        prot = os.path.join(home(dataDir="test-interactions"), "3PTB_prepared.pdb")
        lig = os.path.join(home(dataDir="test-interactions"), "3PTB_BEN.sdf")

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
        ref = np.array(
            [
                [3414, 3421, 2471],
                [3414, 3422, 2789],
                [3415, 3423, 2472],
                [3415, 3424, 2482],
            ]
        )
        assert np.array_equal(hb[0], ref) and np.array_equal(hb[1], ref), print(hb, ref)

        hb = hbonds_calculate(mol, donors, acceptors, "all")
        assert len(hb) == 2
        assert hb[0].shape == (178, 3), hb[0].shape

    def test_pipi(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        mol = Molecule(os.path.join(home(dataDir="test-interactions"), "5L87.pdb"))
        lig = SmallMol(os.path.join(home(dataDir="test-interactions"), "5L87_6RD.sdf"))

        lig_idx = np.where(mol.resname == "6RD")[0][0]

        prot_rings = get_protein_rings(mol)
        lig_rings = get_ligand_rings(lig)

        lig_rings = [tuple(lll + lig_idx for lll in ll) for ll in lig_rings]
        pipis, distang = pipi_calculate(mol, prot_rings, lig_rings)

        assert len(pipis) == 1

        ref_rings = np.array([[0, 2], [0, 4], [2, 4]])
        assert np.array_equal(pipis[0], ref_rings)

        ref_distang = np.array(
            [
                [5.33927107, 97.67315674],
                [5.23078251, 85.32985687],
                [5.16490269, 81.33213806],
            ]
        )
        assert np.allclose(distang[0], ref_distang)

    def test_salt_bridge(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        # 3ptb 5tvn 5l87

        prot = os.path.join(home(dataDir="test-interactions"), "3PTB_prepared.pdb")
        lig = os.path.join(home(dataDir="test-interactions"), "3PTB_BEN.sdf")

        mol = Molecule(prot)
        lig = SmallMol(lig)
        mol.append(lig.toMolecule())
        mol.coords = np.tile(mol.coords, (1, 1, 2)).copy()  # Fake second frame

        prot_pos, prot_neg = get_charged(mol)
        lig_pos, lig_neg = get_ligand_charged(lig)

        bridges = saltbridge_calculate(
            mol, prot_pos + lig_pos, prot_neg + lig_neg, "protein", "resname BEN"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)