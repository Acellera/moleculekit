import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_hbonds():
    from moleculekit.molecule import Molecule
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.interactions.interactions import (
        get_donors_acceptors,
        get_ligand_donors_acceptors,
        hbonds_calculate,
    )
    import os

    prot = os.path.join(curr_dir, "test_interactions", "3PTB_prepared.pdb")
    lig = os.path.join(curr_dir, "test_interactions", "3PTB_BEN.sdf")

    mol = Molecule(prot)
    mol.guessBonds()
    donors, acceptors = get_donors_acceptors(
        mol, exclude_water=True, exclude_backbone=False
    )

    lig_sm = SmallMol(lig)
    lig = lig_sm.toMolecule()
    mol.append(lig)
    lig_idx = np.where(mol.resname == "BEN")[0][0]

    lig_don, lig_acc = get_ligand_donors_acceptors(lig_sm)

    # TODO: Add check if bonds exist in molecule! Add it to moleculechecks
    mol.bonds = mol._guessBonds()
    mol.coords = np.tile(mol.coords, (1, 1, 2)).copy()  # Fake second frame
    mol.box = np.tile(mol.box, (1, 2)).copy()

    donors = np.vstack((donors, lig_don + lig_idx))
    acceptors = np.hstack((acceptors, lig_acc + lig_idx))

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
    assert np.array_equal(hb[0], ref) and np.array_equal(hb[1], ref), f"{hb}, {ref}"

    hb = hbonds_calculate(mol, donors, acceptors, "all")
    assert len(hb) == 2
    assert np.array(hb[0]).shape == (178, 3), np.array(hb[0]).shape


def _test_pipi():
    from moleculekit.molecule import Molecule
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.interactions.interactions import (
        get_protein_rings,
        get_ligand_rings,
        pipi_calculate,
        get_nucleic_rings,
    )
    import os

    mol = Molecule(os.path.join(curr_dir, "test_interactions", "5L87.pdb"))
    lig = SmallMol(os.path.join(curr_dir, "test_interactions", "5L87_6RD.sdf"))

    lig_idx = np.where(mol.resname == "6RD")[0][0]

    prot_rings = get_protein_rings(mol)
    lig_rings = get_ligand_rings(lig, start_idx=lig_idx)

    pipis, distang = pipi_calculate(mol, prot_rings, lig_rings)

    assert len(pipis) == 1

    ref_rings = np.array([[0, 2], [0, 3], [2, 3]])
    assert np.array_equal(pipis[0], ref_rings)

    ref_distang = np.array(
        [
            [5.339271068572998, 81.60645294189453],
            [5.230782508850098, 84.51882934570312],
            [5.164902687072754, 80.55224609375],
        ]
    )
    assert np.allclose(distang[0], ref_distang), f"\n{distang[0]}\n{ref_distang}"

    mol = Molecule(os.path.join(curr_dir, "test_interactions", "6dn1.pdb"))
    lig = SmallMol(os.path.join(curr_dir, "test_interactions", "6dn1_ligand-RDK.sdf"))

    lig_idx = np.where(mol.resname == "MOL")[0][0]

    prot_rings = get_nucleic_rings(mol)
    lig_rings = get_ligand_rings(lig, start_idx=lig_idx)

    pipis, distang = pipi_calculate(mol, prot_rings, lig_rings)

    assert len(pipis) == 1

    ref_rings = np.array([[72, 2], [73, 1], [74, 1]])
    assert np.array_equal(pipis[0], ref_rings)
    ref_distang = np.array(
        [
            [4.146290302276611, 15.108987808227539],
            [4.059301853179932, 18.3873348236084],
            [3.521082639694214, 5.454310894012451],
        ]
    )
    assert np.allclose(distang[0], ref_distang)


def _test_salt_bridge():
    from moleculekit.molecule import Molecule
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.interactions.interactions import (
        get_protein_charged,
        get_ligand_charged,
        saltbridge_calculate,
    )
    import os

    prot = os.path.join(curr_dir, "test_interactions", "3PTB_prepared.pdb")
    lig = os.path.join(curr_dir, "test_interactions", "3PTB_BEN.sdf")

    mol = Molecule(prot)
    lig = SmallMol(lig)
    mol.append(lig.toMolecule())
    mol.coords = np.tile(mol.coords, (1, 1, 2)).copy()  # Fake second frame

    lig_idx = np.where(mol.resname == "BEN")[0][0]

    prot_pos, prot_neg = get_protein_charged(mol)
    lig_pos, lig_neg = get_ligand_charged(lig, start_idx=lig_idx)

    bridges = saltbridge_calculate(
        mol,
        np.hstack((prot_pos, lig_pos)),
        np.hstack((prot_neg, lig_neg)),
        "protein",
        "resname BEN",
    )

    assert len(bridges) == 2

    ref_bridge = np.array([[2470, 3414]])
    assert np.array_equal(bridges[0], ref_bridge)

    prot = os.path.join(curr_dir, "test_interactions", "5ME6_prepared.pdb")
    mol = Molecule(prot)
    prot_pos, prot_neg = get_protein_charged(mol)

    bridges = saltbridge_calculate(mol, prot_pos, prot_neg, "protein", "protein")
    expected = np.array([[694, 725], [2146, 2183], [2158, 2346]])
    assert np.array_equal(
        expected, bridges[0]
    ), f"Failed test_salt_bridge test due to bridges: {expected} {bridges[0]}"


def _test_cationpi_protein():
    from moleculekit.molecule import Molecule
    from moleculekit.interactions.interactions import (
        get_protein_rings,
        get_protein_charged,
        cationpi_calculate,
        get_metal_charged,
    )
    import os

    mol = Molecule(os.path.join(curr_dir, "test_interactions", "1LPI.pdb"))

    prot_rings = get_protein_rings(mol)
    prot_pos, _ = get_protein_charged(mol)
    metal_pos, _ = get_metal_charged(mol)

    catpi, distang = cationpi_calculate(
        mol, prot_rings, np.hstack((prot_pos, metal_pos))
    )

    ref_atms = np.array([[0, 8], [17, 1001], [18, 1001]])
    assert np.array_equal(ref_atms, catpi[0]), f"{ref_atms}, {catpi[0]}"
    ref_distang = np.array(
        [
            [4.101108551025391, 63.75471115112305],
            [4.702703952789307, 60.362159729003906],
            [4.122482776641846, 82.87332153320312],
        ]
    )
    assert np.allclose(ref_distang, distang), distang


def _test_cationpi_protein_ligand():
    from moleculekit.molecule import Molecule
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.interactions.interactions import (
        get_protein_rings,
        get_ligand_rings,
        get_protein_charged,
        get_ligand_charged,
        cationpi_calculate,
    )
    import os

    lig = SmallMol(os.path.join(curr_dir, "test_interactions", "2BOK_784.sdf"))
    mol = Molecule(os.path.join(curr_dir, "test_interactions", "2BOK_prepared.pdb"))
    mol.append(lig.toMolecule())

    lig_idx = np.where(mol.resname == "784")[0][0]

    prot_rings = get_protein_rings(mol)
    lig_rings = get_ligand_rings(lig, start_idx=lig_idx)

    prot_pos, _ = get_protein_charged(mol)
    lig_pos, _ = get_ligand_charged(lig, start_idx=lig_idx)

    catpi, distang = cationpi_calculate(
        mol, prot_rings + lig_rings, np.hstack((prot_pos, lig_pos))
    )

    ref_atms = np.array([[11, 3494]])
    assert np.array_equal(ref_atms, catpi[0]), f"{ref_atms}, {catpi[0]}"
    ref_distang = np.array([[4.74848127, 74.56939697265625]])
    assert np.allclose(ref_distang, distang), distang


def _test_sigma_holes():
    from moleculekit.molecule import Molecule
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.interactions.interactions import (
        get_protein_rings,
        get_ligand_aryl_halides,
        sigmahole_calculate,
    )
    import os

    lig = SmallMol(os.path.join(curr_dir, "test_interactions", "2P95_ME5.sdf"))
    mol = Molecule(os.path.join(curr_dir, "test_interactions", "2P95_prepared.pdb"))
    mol.append(lig.toMolecule())

    lig_idx = np.where(mol.resname == "ME5")[0][0]
    lig_halides = get_ligand_aryl_halides(lig, start_idx=lig_idx)

    prot_rings = get_protein_rings(mol)

    sh, distang = sigmahole_calculate(mol, prot_rings, lig_halides)

    ref_atms = np.array([[29, 3702]])
    assert np.array_equal(ref_atms, sh[0]), f"{ref_atms}, {sh[0]}"

    ref_distang = np.array([[4.26179695, 66.73172760009766]])
    assert np.allclose(ref_distang, distang)


def _test_water_bridge():
    from moleculekit.molecule import Molecule
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.interactions.interactions import (
        get_donors_acceptors,
        waterbridge_calculate,
    )
    import os

    mol = Molecule(
        os.path.join(curr_dir, "test_interactions", "5gw6_receptor_H_wet.pdb")
    )
    mol.bonds = mol._guessBonds()

    lig = SmallMol(os.path.join(curr_dir, "test_interactions", "5gw6_ligand-RDK.sdf"))
    mol.append(lig.toMolecule())

    donors, acceptors = get_donors_acceptors(
        mol, exclude_water=False, exclude_backbone=False
    )

    wb = waterbridge_calculate(
        mol,
        donors,
        acceptors,
        "resname GOL",
        "protein and resname ASN and resid 155",
        order=1,
        dist_threshold=3.8,
        ignore_hs=True,
    )
    assert np.array_equal(wb, [[[3140, 2899, 2024]]]), wb

    wb = waterbridge_calculate(
        mol,
        donors,
        acceptors,
        "resname GOL",
        "protein and resname ASN and resid 155",
        order=2,
        dist_threshold=3.8,
        ignore_hs=True,
    )
    ref = [[[3140, 2899, 2944, 2023], [3140, 2899, 2024]]]
    for i, x in enumerate(ref):
        assert np.array_equal(wb[0][i], ref[0][i]), wb

    wb = waterbridge_calculate(
        mol,
        donors,
        acceptors,
        "resname GOL",
        "protein",
        order=1,
        dist_threshold=3.8,
        ignore_hs=True,
    )
    assert np.array_equal(
        wb,
        [
            [
                [3140, 2899, 2024],
                [3142, 2857, 1317],
                [3142, 2857, 2720],
                [3142, 2857, 2737],
                [3142, 2857, 2789],
            ]
        ],
    ), wb


def _test_metal_coordination():
    from moleculekit.molecule import Molecule
    from moleculekit.interactions.interactions import (
        metal_coordination_calculate,
    )

    mol = Molecule("5vl5")
    res = metal_coordination_calculate(mol, "all", "resname S31 and not element Cu")

    ref = np.array(
        [[933, 922], [933, 932], [933, 934], [933, 935], [933, 937], [933, 944]],
        dtype=np.uint32,
    )
    assert np.array_equal(res[0], ref)

    mol = Molecule("3ptb")
    res = metal_coordination_calculate(mol, "not protein", "protein")

    ref = np.array(
        [[1629, 383], [1629, 396], [1629, 420], [1629, 460]], dtype=np.uint32
    )
    assert np.array_equal(res[0], ref)
