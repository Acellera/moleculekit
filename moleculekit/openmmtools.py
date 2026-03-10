def openmm_to_molecule(topology, positions):
    from moleculekit.molecule import Molecule
    from openmm import unit
    import numpy as np

    positions = np.array(positions.value_in_unit(unit.angstrom))

    mol = Molecule()
    mol.empty(topology.getNumAtoms(), numFrames=1)

    for i, atom in enumerate(topology.atoms()):
        mol.name[i] = atom.name
        mol.element[i] = atom.element.symbol
        mol.resname[i] = atom.residue.name
        mol.resid[i] = int(atom.residue.id)
        mol.chain[i] = atom.residue.chain.id
        mol.insertion[i] = atom.residue.insertionCode.strip()
        mol.formalcharge[i] = atom.formalCharge if atom.formalCharge is not None else 0
        mol.coords[i, :, 0] = positions[i]

    bonds = []
    bondtype = []
    for bond in topology.bonds():
        bonds.append([bond.atom1.index, bond.atom2.index])
        bondtype.append(bond.order if bond.order is not None else "un")
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(bondtype, dtype=np.object_)

    return mol