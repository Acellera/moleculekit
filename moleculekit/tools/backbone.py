from moleculekit.molecule import Molecule
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MissingBackboneError(Exception):
    pass


def _extend_c_terminus_c_atom(
    mol: Molecule, prev_idx: np.ndarray, curr_idx: np.ndarray
):
    """
    Reconstructs the missing C-terminal Carbon (C) atom.

    Strategy Priority:
    1. Rigid Geometry: Uses N, CA, and HA (if available) for exact analytical placement.
    2. Heuristic: Uses the previous residue's Psi angle to guess Helix vs. Sheet.
    """

    # --- Constants ---
    BOND_LENGTH_CA_C = 1.51
    ANGLE_N_CA_C_RAD = np.radians(111.2)

    # --- 1. Helper: Safe Coordinate Extraction ---
    def get_vec(indices, atom_name):
        hits = indices[mol.name[indices] == atom_name]
        if len(hits) == 0:
            return None
        return mol.coords[hits[0], :, 0]

    # Extract Essential Atoms
    vec_n = get_vec(curr_idx, "N")
    vec_ca = get_vec(curr_idx, "CA")

    if vec_n is None or vec_ca is None:
        raise ValueError("Current residue must have N and CA atoms.")

    # --- Strategy A: Rigid Frame (Requires HA) ---
    # We look for HA, HA1 (Gly), HA2 (Gly-Pro-S), or HA3 (Gly-Pro-R)
    vec_ha = None
    for candidate_name in ("HA", "HA1", "HA2"):
        candidate_vec = get_vec(curr_idx, candidate_name)
        if candidate_vec is not None:
            vec_ha = candidate_vec
            break
    chirality = 1.0

    # Check for Glycine 'HA3' which is on the opposite side (Pro-R)
    if vec_ha is None:
        vec_ha = get_vec(curr_idx, "HA3")
        if vec_ha is not None:
            chirality = -1.0

    final_coords = None

    if vec_ha is not None:
        # We have a Hydrogen! Solve analytically.
        # This treats CA as the origin for calculation
        u_n = (vec_n - vec_ca) / np.linalg.norm(vec_n - vec_ca)
        u_h = (vec_ha - vec_ca) / np.linalg.norm(vec_ha - vec_ca)

        # Constraints: Dot products based on ideal bond angles
        target_dot_n = np.cos(ANGLE_N_CA_C_RAD)  # Angle N-CA-C
        target_dot_h = np.cos(np.radians(109.5))  # Angle H-CA-C
        dot_n_h = np.dot(u_n, u_h)

        # Solve Linear System: u_c = alpha*u_n + beta*u_h + gamma*(cross_prod)
        try:
            M = np.array([[1.0, dot_n_h], [dot_n_h, 1.0]])
            coeffs = np.linalg.solve(M, [target_dot_n, target_dot_h])

            # Calculate out-of-plane component (gamma)
            vec_in_plane = coeffs[0] * u_n + coeffs[1] * u_h
            remainder = 1.0 - np.dot(vec_in_plane, vec_in_plane)

            if remainder > 0:
                u_cross = np.cross(u_n, u_h)
                u_cross /= np.linalg.norm(u_cross)
                gamma_vec = np.sqrt(remainder) * u_cross * chirality

                # Result
                final_coords = vec_ca + (vec_in_plane + gamma_vec) * BOND_LENGTH_CA_C
        except np.linalg.LinAlgError:
            pass  # Fallback to Strategy B if geometry is degenerate

    # --- Strategy B: Heuristic NeRF (Fallback) ---
    if final_coords is None:
        vec_prev_c = get_vec(prev_idx, "C")
        if vec_prev_c is None:
            # Absolute fallback (Start of chain?): Assume Extended
            psi_deg = 180.0
        else:
            # Calculate Previous Psi: N_prev -> CA_prev -> C_prev -> N_curr
            # We use a simplified vector math here to avoid external func dependency
            p0 = get_vec(prev_idx, "N")
            p1 = get_vec(prev_idx, "CA")
            p2 = vec_prev_c
            p3 = vec_n

            # Calculate Psi (Dihedral)
            b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
            b2_u = b2 / np.linalg.norm(b2)
            v = b1 - np.dot(b1, b2_u) * b2_u
            w = b3 - np.dot(b3, b2_u) * b2_u
            psi_deg = np.degrees(np.arctan2(np.dot(np.cross(b2_u, v), w), np.dot(v, w)))

        # Guess Phi based on Psi
        # Helix (-60) if Psi is roughly -45 (-70 to -10)
        # Sheet (-120) otherwise
        phi_guess = -60.0 if (-70 <= psi_deg <= -10) else -120.0

        # NeRF Placement
        # Plane defined by: C_prev (A) -> N (B) -> CA (C)
        # We construct bond CA -> C_new (D)
        # Note: If C_prev missing, we assume a generic vector 'ab' along x-axis
        bc = vec_ca - vec_n
        bc_u = bc / np.linalg.norm(bc)

        if vec_prev_c is not None:
            ab = vec_n - vec_prev_c
            n_vec = np.cross(ab, bc_u)
        else:
            n_vec = np.cross(np.array([1, 0, 0]), bc_u)  # Arbitrary reference

        n_u = n_vec / np.linalg.norm(n_vec)
        cross_u = np.cross(n_u, bc_u)

        theta = np.radians(180 - 111.2)
        torsion = np.radians(phi_guess)

        d_vec = (
            (np.cos(theta) * bc_u)
            + (np.sin(theta) * np.cos(torsion) * cross_u)
            + (np.sin(theta) * np.sin(torsion) * n_u)
        )

        final_coords = vec_ca + (d_vec * BOND_LENGTH_CA_C)

    # --- 3. Create Atom ---
    # Copy CA atom to inherit properties
    ca_idx = curr_idx[mol.name[curr_idx] == "CA"][0]
    new_atom = mol.copy(sel=ca_idx)
    new_atom.name[:] = "C"
    new_atom.element[:] = "C"
    new_atom.coords[:, :, 0] = final_coords
    return new_atom, curr_idx[0] + 2


def _reconstruct_backbone_planar_atom(
    mol: Molecule,
    prev_idx: np.ndarray,
    curr_idx: np.ndarray,
    next_idx: np.ndarray,
    missing_atom: str,
):
    # Given a backbone with 3 atoms this function will add the missing 4th atom to the backbone
    # Based on the CA-C-N plane on which the O also lies
    C_BOND_LENGTHS = {"O": 1.23, "CA": 1.51, "N": 1.33}

    new_atom = mol.copy(sel=curr_idx[0])
    new_atom.name[:] = missing_atom
    new_atom.element[:] = missing_atom[0]

    if missing_atom == "C":
        # Calculate the coordinates of the C atom as the weighted average of the O, CA, and N atom coordinates
        # With weights inversely proportional to the bond lengths
        o_idx = curr_idx[mol.name[curr_idx] == "O"][0]
        ca_idx = curr_idx[mol.name[curr_idx] == "CA"][0]
        n_idx = next_idx[mol.name[next_idx] == "N"][0]
        w_ca = 1 / C_BOND_LENGTHS["CA"]
        w_o = 1 / C_BOND_LENGTHS["O"]
        w_n = 1 / C_BOND_LENGTHS["N"]
        w_total = w_ca + w_o + w_n
        c_coords = (
            w_ca * mol.coords[ca_idx, :, 0]
            + w_o * mol.coords[o_idx, :, 0]
            + w_n * mol.coords[n_idx, :, 0]
        ) / w_total
        new_atom.coords[:, :, 0] = c_coords
        return new_atom, curr_idx[0] + 2

    if missing_atom == "N":
        center = prev_idx[mol.name[prev_idx] == "C"][0]
        neighbor_1 = prev_idx[mol.name[prev_idx] == "CA"][0]
        neighbor_2 = prev_idx[mol.name[prev_idx] == "O"][0]
        insert_at = curr_idx[0]
    elif missing_atom == "CA":
        center = curr_idx[mol.name[curr_idx] == "C"][0]
        neighbor_1 = next_idx[mol.name[next_idx] == "N"][0]
        neighbor_2 = curr_idx[mol.name[curr_idx] == "O"][0]
        insert_at = curr_idx[0] + 1
    elif missing_atom == "O":
        center = curr_idx[mol.name[curr_idx] == "C"][0]
        neighbor_1 = curr_idx[mol.name[curr_idx] == "CA"][0]
        neighbor_2 = next_idx[mol.name[next_idx] == "N"][0]
        insert_at = curr_idx[0] + 3
    else:
        raise ValueError(f"Invalid missing atom: {missing_atom}")

    # Vector from Center to Neighbor 1
    v1 = mol.coords[neighbor_1, :, 0] - mol.coords[center, :, 0]
    # Vector from Center to Neighbor 2
    v2 = mol.coords[neighbor_2, :, 0] - mol.coords[center, :, 0]

    # Normalize to create unit vectors
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)

    # The missing atom is roughly opposite to the sum of the two known vectors
    direction = -(u1 + u2)

    u_dir = direction / np.linalg.norm(direction)

    new_atom.coords[:, :, 0] = (
        mol.coords[center, :, 0] + u_dir * C_BOND_LENGTHS[missing_atom]
    )
    return (new_atom, insert_at)


def _iterate_residues(mol: Molecule):
    BB_ATOM_NAMES = {"N", "CA", "C", "O"}

    _, res_idx = mol.getResidues(return_idx=True)
    for i, curr_idx in enumerate(res_idx):
        ii = curr_idx[0]
        curr_chain = mol.chain[ii]
        prev_idx = None
        if ii > 0:
            prev_idx = res_idx[i - 1]
            prev_chain = mol.chain[prev_idx[0]]
            prev_has_bb = np.sum(np.isin(mol.name[prev_idx], list(BB_ATOM_NAMES))) > 2
        next_idx = None
        if i < len(res_idx) - 1:
            next_idx = res_idx[i + 1]
            next_chain = mol.chain[next_idx[0]]
            next_has_bb = np.sum(np.isin(mol.name[next_idx], list(BB_ATOM_NAMES))) > 2

        n_terminal = prev_idx is None
        c_terminal = next_idx is None
        if prev_idx is not None:
            n_terminal |= (prev_chain != curr_chain) or not prev_has_bb
        if next_chain is not None:
            c_terminal |= (curr_chain != next_chain) or not next_has_bb

        is_terminal = n_terminal or c_terminal
        yield prev_idx, curr_idx, next_idx, is_terminal, n_terminal, c_terminal


def check_backbone(
    mol: Molecule,
    remove_broken_terminals: bool = True,
    terminal_min_heavy_atoms: int = 4,
):
    """Checks the backbone of all canonical aminoacids in a Molecule object and adds missing atoms if needed.

    If single atoms are missing in the backbone of a residue, they will be reconstructed.
    If multiple atoms are missing in the backbone of a residue and the residue is at a terminal,
    the residue will be removed.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to check the backbone of
    remove_broken_terminals : bool, optional
        Whether to remove residues that are at a terminal and have less than `terminal_min_heavy_atoms` heavy atoms.
        Default is True.
    terminal_min_heavy_atoms : int, optional
        The minimum number of heavy atoms required at a terminal to not be removed.

    Returns
    -------
    mol : moleculekit.molecule.Molecule
        The molecule with the missing backbone atoms added

    Raises
    ------
    MissingBackboneError : If the molecule has missing backbone atoms that cannot be reconstructed
    """
    from moleculekit.residues import PROTEIN_RESIDUE_NAMES

    BB_ATOM_NAMES = {"N", "CA", "C", "O"}

    report = []
    to_add = []
    to_remove = []
    for prev_idx, curr_idx, next_idx, _, _, c_terminal in _iterate_residues(mol):
        ii = curr_idx[0]
        if mol.resname[ii] in PROTEIN_RESIDUE_NAMES:
            # Check which backbone atoms are missing
            missing_atoms = BB_ATOM_NAMES - set(mol.name[curr_idx])
            # Special handling for OXT atoms at the terminals
            if "OXT" in mol.name[curr_idx] and "O" in missing_atoms:
                missing_atoms.remove("O")
            # If only one atom is missing, we might be able to reconstruct the backbone atom
            # With the help of the previous or next residue
            if len(missing_atoms) == 1:
                missing_atom = list(missing_atoms)[0]
                if (
                    missing_atom in ("CA", "C", "O")
                    and next_idx is not None
                    and "N" in mol.name[next_idx]
                ):
                    to_add.append(
                        _reconstruct_backbone_planar_atom(
                            mol, prev_idx, curr_idx, next_idx, missing_atom
                        )
                    )
                if (
                    missing_atom == "N"
                    and prev_idx is not None
                    and "C" in mol.name[prev_idx]
                    and "CA" in mol.name[prev_idx]
                    and "O" in mol.name[prev_idx]
                ):
                    to_add.append(
                        _reconstruct_backbone_planar_atom(
                            mol, prev_idx, curr_idx, next_idx, missing_atom
                        )
                    )
            if (
                c_terminal
                and "C" in missing_atoms
                and not "N" in missing_atoms
                and not "CA" in missing_atoms
            ):  # We are at the C-terminal and missing a C atom, we can extend the C-terminal
                to_add.append(_extend_c_terminus_c_atom(mol, prev_idx, curr_idx))

    # Add all the new atoms to the molecule
    if len(to_add) > 0:
        logger.info(f"Adding {len(to_add)} missing backbone atoms")
        for i, (new_atom, new_idx) in enumerate(to_add):
            # Add the +i to the idx to account for the previous insertions
            mol.insert(new_atom, new_idx + i)

    # Now check again for residues with missing backbone atoms
    # This time we will remove the residues if they are at a terminal and have less than 4 total atoms
    # Otherwise we will throw an error
    for _, curr_idx, _, is_terminal, n_terminal, c_terminal in _iterate_residues(mol):
        ii = curr_idx[0]
        if mol.resname[ii] in PROTEIN_RESIDUE_NAMES:
            missing_atoms = BB_ATOM_NAMES - set(mol.name[curr_idx])
            if "OXT" in mol.name[curr_idx] and "O" in missing_atoms:
                missing_atoms.remove("O")
            if c_terminal and "O" in missing_atoms:
                # O missing at the C-terminal is OK, it will be capped correctly by pdb2pqr
                missing_atoms.remove("O")
            if n_terminal and "N" in missing_atoms:
                # N missing at the N-terminal is OK, it will be capped correctly by pdb2pqr
                missing_atoms.remove("N")
            if len(missing_atoms) != 0:
                if (
                    remove_broken_terminals
                    and is_terminal
                    and np.sum(mol.element[curr_idx] != "H") < terminal_min_heavy_atoms
                ):
                    # Remove this residue from the molecule if it has less than terminal_min_heavy_atoms heavy atoms
                    to_remove.append(curr_idx)
                    logger.warning(
                        f"Removing terminal residue {mol.resname[ii]}:{mol.resid[ii]}{mol.insertion[ii]}:{mol.chain[ii]} "
                        f"because it is missing backbone atoms: {missing_atoms}"
                    )
                    continue
                msg = (
                    f"Residue {mol.resname[ii]}:{mol.resid[ii]}{mol.insertion[ii]}:{mol.chain[ii]} "
                    f"is missing backbone atoms: {missing_atoms}"
                )
                report.append(msg)

    # Remove all the residues that we flagged for removal
    if len(to_remove) > 0:
        mol.remove(np.hstack(to_remove), _logger=False)

    # If we still have any residues with missing backbone atoms, we will throw an error
    if len(report) > 0:
        raise MissingBackboneError(
            "The following residues have invalid backbones:\n"
            + "\n".join(report)
            + "\nStructure preparation cannot continue without a complete backbone. "
            "Please fix the backbones of these residues or remove them from the structure and run the function again."
        )
