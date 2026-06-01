from moleculekit.molecule import Molecule


def test_optimized_ligands():
    from moleculekit.tools.moleculechecks import isLigandOptimized

    mol = Molecule("3ptb")
    mol.filter("resname BEN")

    assert isLigandOptimized(mol)
    mol.coords[:, 0, :] = -1.7  # Flatten on x axis
    assert not isLigandOptimized(mol)


def test_docked_ligands():
    from moleculekit.tools.moleculechecks import isLigandDocked

    prot = Molecule("3ptb")
    prot.filter("protein")
    lig = Molecule("3ptb")
    lig.filter("resname BEN")

    assert isLigandDocked(prot, lig)
    lig.moveBy([0, 20, 0])
    assert not isLigandDocked(prot, lig)


def test_protonated_protein():
    from moleculekit.tools.moleculechecks import isProteinProtonated
    from moleculekit.tools.preparation import systemPrepare

    prot = Molecule("3ptb")
    prot.filter("protein")
    prot_protonated = prot.copy()
    prot_protonated, _ = systemPrepare(prot_protonated)

    assert not isProteinProtonated(prot)
    assert isProteinProtonated(prot_protonated)
