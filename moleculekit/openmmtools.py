import logging

import numpy as np

logger = logging.getLogger(__name__)


def _add_bonded_forces(system, mol, mobile_atom_indices, positions_nm):
    """Add harmonic bond and angle restraints for mobile atoms.

    Equilibrium bond lengths and angles are taken from the current
    coordinates (assumed to be ideal geometry from a CIF template or
    RDKit embedding).  Only terms involving at least one mobile atom
    are added.

    Parameters
    ----------
    system : openmm.System
    mol : Molecule
    mobile_atom_indices : set of int
    positions_nm : np.ndarray, shape (N, 3)
        Atom positions in **nanometers**.
    """
    from openmm import HarmonicBondForce, HarmonicAngleForce
    from openmm import unit as u

    if mol.bonds is None or len(mol.bonds) == 0:
        return

    from collections import defaultdict

    neighbors = defaultdict(set)
    for b in mol.bonds:
        a1, a2 = int(b[0]), int(b[1])
        neighbors[a1].add(a2)
        neighbors[a2].add(a1)

    # ── Harmonic bonds ────────────────────────────────────────────────
    bond_force = HarmonicBondForce()
    bond_k = 100000.0  # kJ/mol/nm²  (~239 kcal/mol/Å²)

    for b in mol.bonds:
        a1, a2 = int(b[0]), int(b[1])
        if a1 not in mobile_atom_indices and a2 not in mobile_atom_indices:
            continue
        r0 = float(np.linalg.norm(positions_nm[a1] - positions_nm[a2]))
        if r0 < 1e-6:
            continue
        bond_force.addBond(a1, a2, r0 * u.nanometer, bond_k)

    if bond_force.getNumBonds() > 0:
        system.addForce(bond_force)
        logger.debug(f"Added {bond_force.getNumBonds()} harmonic bond restraints")

    # ── Harmonic angles ───────────────────────────────────────────────
    angle_force = HarmonicAngleForce()
    angle_k = 500.0  # kJ/mol/rad²  (~120 kcal/mol/rad²)

    seen = set()
    for center in neighbors:
        bonded = sorted(neighbors[center])
        for i in range(len(bonded)):
            for j in range(i + 1, len(bonded)):
                a1, a3 = bonded[i], bonded[j]
                if (
                    a1 not in mobile_atom_indices
                    and center not in mobile_atom_indices
                    and a3 not in mobile_atom_indices
                ):
                    continue
                key = (min(a1, a3), center, max(a1, a3))
                if key in seen:
                    continue
                seen.add(key)

                v1 = positions_nm[a1] - positions_nm[center]
                v2 = positions_nm[a3] - positions_nm[center]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                theta0 = float(np.arccos(cos_theta))
                angle_force.addAngle(a1, center, a3, theta0 * u.radian, angle_k)

    if angle_force.getNumAngles() > 0:
        system.addForce(angle_force)
        logger.debug(f"Added {angle_force.getNumAngles()} harmonic angle restraints")


def minimize_soft_potential(mol, mobile_atom_indices, max_iterations=200):
    """Run a soft-potential energy minimization on selected atoms.

    All other atoms are frozen (mass = 0).  Uses a soft repulsive
    ``CustomNonbondedForce`` so overlapping atoms are gently pushed apart
    rather than exploding.  Only interactions involving at least one
    mobile atom are computed.

    Harmonic bond and angle restraints are added for bonds in
    ``mol.bonds`` involving at least one mobile atom; equilibrium values
    are taken from the current coordinates.  This keeps bond lengths and
    angles close to their starting geometry so the minimization resolves
    clashes by rotating dihedrals rather than distorting covalent
    structure.

    Parameters
    ----------
    mol : Molecule
        The molecule to minimize (modified **in place**).
    mobile_atom_indices : set or list of int
        Indices of atoms that are free to move.
    max_iterations : int, optional
        Maximum number of minimization iterations.  Default 200.

    Returns
    -------
    bool
        True if minimization was performed, False if OpenMM is unavailable.
    """
    try:
        from openmm import CustomNonbondedForce, System, LangevinMiddleIntegrator
        from openmm import LocalEnergyMinimizer, Platform, Context
        from openmm import unit as u
    except ImportError:
        logger.info("OpenMM not available -- skipping soft-potential minimization.")
        return False

    mobile_atom_indices = set(mobile_atom_indices)
    n_atoms = mol.numAtoms
    frame = mol.frame

    if mol.bonds is None or len(mol.bonds) == 0:
        raise ValueError(
            "minimize_soft_potential requires mol.bonds to be populated for "
            "bond/angle restraints. Call mol._guessBonds() or load a topology "
            "with bond information."
        )
    bonds_arr = np.asarray(mol.bonds)
    mobile_mask = np.zeros(n_atoms, dtype=bool)
    mobile_mask[list(mobile_atom_indices)] = True
    if not (mobile_mask[bonds_arr[:, 0]] | mobile_mask[bonds_arr[:, 1]]).any():
        raise ValueError(
            "No bonds in mol.bonds involve any of the mobile atoms; cannot "
            "build bond/angle restraints. Check that mol.bonds connects the "
            "mobile selection."
        )

    system = System()
    for i in range(n_atoms):
        mass = 12.0 if i in mobile_atom_indices else 0.0
        system.addParticle(mass * u.dalton)

    positions_nm = mol.coords[:, :, frame].astype(np.float64) * 0.1  # Å -> nm

    # Bonded forces keep bond lengths and angles close to ideal geometry
    _add_bonded_forces(system, mol, mobile_atom_indices, positions_nm)

    # Soft repulsive potential: C / ((r/0.2)^4 + 1)
    # C = 10 kJ/mol; 0.2 nm = 2 Angstrom core
    nb_force = CustomNonbondedForce("C / ((r/0.2)^4 + 1); C=10")
    nb_force.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
    for _ in range(n_atoms):
        nb_force.addParticle([])
    nb_force.addInteractionGroup(list(mobile_atom_indices), list(range(n_atoms)))
    system.addForce(nb_force)

    integrator = LangevinMiddleIntegrator(
        300 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picoseconds
    )

    platform = Platform.getPlatformByName("Reference")
    context = Context(system, integrator, platform)
    context.setPositions(positions_nm * u.nanometer)

    e_before = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(u.kilocalories_per_mole)
    )

    LocalEnergyMinimizer.minimize(context, maxIterations=max_iterations)

    state = context.getState(getPositions=True, getEnergy=True)
    e_after = state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
    logger.info(
        f"Soft-potential minimization: {e_before:.1f} -> {e_after:.1f} kcal/mol "
        f"(delta = {e_after - e_before:.1f} kcal/mol)"
    )

    new_positions = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    new_positions = np.array(new_positions) * 10.0  # nm -> Angstrom

    for idx in mobile_atom_indices:
        mol.coords[idx, :, frame] = new_positions[idx].astype(np.float32)

    return True


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