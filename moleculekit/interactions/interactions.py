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

    # TODO: Add halogens to the pyx code

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


def saltbridge_calculate(mol, pos, neg, sel1="all", sel2=None, threshold=4):
    from moleculekit.projections.util import pp_calcDistances

    sel1 = mol.atomselect(sel1).astype(np.uint32).copy()
    if sel2 is None:
        sel2 = sel1.copy()
    else:
        sel2 = mol.atomselect(sel2).astype(np.uint32).copy()

    charged = np.zeros(sel1.shape, dtype=bool)
    charged[pos] = True
    charged[neg] = True

    sel1 = sel1 & charged
    sel2 = sel2 & charged

    if np.sum(sel1) == 0 or np.sum(sel2) == 0:
        return [[] for _ in range(mol.numFrames)]

    periodic = "selections" if not np.all(mol.box == 0) else None
    dists = pp_calcDistances(mol, sel1, sel2, periodic, metric="distances")

    sel1 = np.where(sel1)[0]
    sel2 = np.where(sel2)[0]

    salt_bridge_list = []
    for f in range(mol.numFrames):
        idx = np.where(dists[f] < threshold)[0]
        sel1_sub = sel1[(idx % len(sel1)).astype(int)]
        sel2_sub = sel2[(idx / len(sel1)).astype(int)]
        inter = np.vstack((sel1_sub, sel2_sub)).T
        # Should only have one positive per row (the other is necessarily negative)
        opposite_charges = np.sum(np.isin(inter, pos), axis=1) == 1
        salt_bridge_list.append(inter[opposite_charges])

    return salt_bridge_list


def get_aryl_halides_protein(mol):
    # TODO: Need to support non-standard residues
    pass


def get_ligand_aryl_halides(sm):
    import networkx as nx

    halogens = ["Cl", "Br", "I"]
    rings = get_ligand_rings(sm)
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
            halides.append([hi, neighs[0]])

    return halides


def cationpi_calculate(
    mol,
    rings,
    cations,
    dist_threshold=5,
    angle_threshold_min=60,
    angle_threshold_max=120,
):
    from moleculekit.interactions import cationpi

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
        angle_threshold_max=angle_threshold_max,
    )

    index_list = []
    dist_ang_list = []
    for f in range(mol.numFrames):
        index_list.append(np.array(pp[f]).reshape(-1, 2))
        dist_ang_list.append(np.array(da[f]).reshape(-1, 2))
    return index_list, dist_ang_list


def sigmahole_calculate(
    mol,
    rings,
    halides,
    dist_threshold=4.5,
    angle_threshold_min=60,
    angle_threshold_max=120,
):
    from moleculekit.interactions import sigmahole

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
        angle_threshold_max=angle_threshold_max,
    )

    index_list = []
    dist_ang_list = []
    for f in range(mol.numFrames):
        index_list.append(np.array(indexes[f]).reshape(-1, 2))
        dist_ang_list.append(np.array(da[f]).reshape(-1, 2))
    return index_list, dist_ang_list


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

        lig_idx = np.where(mol.resname == "BEN")[0][0]
        lig_pos = [ll + lig_idx for ll in lig_pos]
        lig_neg = [ll + lig_idx for ll in lig_neg]
        bridges = saltbridge_calculate(
            mol, prot_pos + lig_pos, prot_neg + lig_neg, "protein", "resname BEN"
        )

        assert len(bridges) == 2

        ref_bridge = np.array([[2470, 3414]])
        assert np.array_equal(bridges[0], ref_bridge)

    def test_cationpi_protein(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        mol = Molecule(os.path.join(home(dataDir="test-interactions"), "1LPI.pdb"))
        prot_rings = get_protein_rings(mol)
        prot_pos, _ = get_charged(mol)

        catpi, distang = cationpi_calculate(mol, prot_rings, prot_pos)

        ref_atms = np.array([[0, 8], [17, 1001], [18, 1001]])
        assert np.array_equal(ref_atms, catpi[0]), print(ref_atms, catpi[0])
        ref_distang = np.array(
            [
                [4.10110903, 63.69768524],
                [4.70270395, 60.51351929],
                [4.12248421, 82.81176758],
            ]
        )
        assert np.allclose(ref_distang, distang)

    def test_cationpi_protein_ligand(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        lig = SmallMol(os.path.join(home(dataDir="test-interactions"), "2BOK_784.sdf"))
        mol = Molecule(
            os.path.join(home(dataDir="test-interactions"), "2BOK_prepared.pdb")
        )
        mol.append(lig.toMolecule())

        lig_idx = np.where(mol.resname == "784")[0][0]

        prot_rings = get_protein_rings(mol)
        lig_rings = get_ligand_rings(lig)
        lig_rings = [tuple(lll + lig_idx for lll in ll) for ll in lig_rings]

        prot_pos, _ = get_charged(mol)
        lig_pos, _ = get_ligand_charged(lig)
        lig_pos = [ll + lig_idx for ll in lig_pos]

        catpi, distang = cationpi_calculate(
            mol, prot_rings + lig_rings, prot_pos + lig_pos
        )

        ref_atms = np.array([[11, 3494]])
        assert np.array_equal(ref_atms, catpi[0]), print(ref_atms, catpi[0])
        ref_distang = np.array([[4.74848127, 74.07044983]])
        assert np.allclose(ref_distang, distang)

    def test_sigma_holes(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        from moleculekit.smallmol.smallmol import SmallMol
        import os

        lig = SmallMol(os.path.join(home(dataDir="test-interactions"), "2P95_ME5.sdf"))
        mol = Molecule(
            os.path.join(home(dataDir="test-interactions"), "2P95_prepared.pdb")
        )
        mol.append(lig.toMolecule())

        lig_idx = np.where(mol.resname == "ME5")[0][0]
        lig_halides = get_ligand_aryl_halides(lig)
        lig_halides = [[ll[0] + lig_idx, ll[1] + lig_idx] for ll in lig_halides]

        prot_rings = get_protein_rings(mol)

        sh, distang = sigmahole_calculate(mol, prot_rings, lig_halides)

        ref_atms = np.array([[29, 3702]])
        assert np.array_equal(ref_atms, sh[0]), print(ref_atms, sh[0])
        ref_distang = np.array([[4.26179695, 66.55052185]])
        assert np.allclose(ref_distang, distang)


if __name__ == "__main__":
    unittest.main(verbosity=2)