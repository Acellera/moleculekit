from moleculekit.molecule import Molecule
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MissingBackboneError(Exception):
    """Raised when a residue is missing backbone atoms that cannot be reconstructed."""

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


# Per polymer: the atomselect keyword, and the link kind that continues a chain of
# it. A side-chain isopeptide is deliberately not here -- it crosslinks.
_POLYMER_LINKS = {"protein": "peptide", "nucleic": "phosphodiester"}


def _polymer_masks(mol: Molecule):
    """``{"protein": mask, "nucleic": mask}`` -- the shared polymer test.

    What ``atomselect("protein")`` is usually asked for, answered so that a residue
    modelled short of its backbone still counts. A residue is protein to that
    selection only once four connected backbone atoms are present, and the protein
    list is exactly ``N, CA, C, O``, so a residue missing only its carbonyl O is
    not protein: it drops out of any sequence derived from the selection, and an
    alignment against the reference then reports a residue missing that is sitting
    in the file, bonded to both its neighbours. Gaps, termini and the caps a build
    is asked for all inherit that.

    The selection is added to, never reduced. Whatever it accepts stays accepted,
    so a non-canonical residue known to no name list and recognised only by its
    shape is unaffected, as is a structure whose coordinates carry no usable
    geometry. A residue it rejects joins the chain when its resname is a polymer
    one *and*
    :func:`~moleculekit.tools.nonstandard_residues.geometric_interresidue_links`
    puts a backbone link to a neighbour -- a peptide bond for protein, a
    phosphodiester for nucleic. The link is what separates a residue of the chain
    from a free amino acid or nucleotide in the solvent, which belongs to neither.
    A side-chain isopeptide does not count: it crosslinks, it does not continue a
    chain. Requiring the link is also what keeps a name from deciding alone: a
    phosphoserine carries a P and no nucleic backbone at all.

    The same four-atom threshold applies to ``atomselect("nucleic")``, but its
    backbone list holds about ten names per nucleotide (``P``, ``OP1``, ``OP2``,
    ``C5'``, ``O5'``, ``C4'``, ``C3'``, ``O3'`` and the ``*`` spellings), so four
    of them is a far looser bar than four of four and a nucleotide has to lose most
    of its backbone before it drops out.
    """
    from moleculekit.residues import (
        MODIFIED_NUCLEIC_RESIDUE_NAMES,
        MODIFIED_PROTEIN_RESIDUE_NAMES,
        NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS,
        PROTEIN_RESIDUE_NAMES_WITH_VARIANTS,
    )
    from moleculekit.tools.nonstandard_residues import geometric_interresidue_links

    tables = {
        "protein": PROTEIN_RESIDUE_NAMES_WITH_VARIANTS | MODIFIED_PROTEIN_RESIDUE_NAMES,
        "nucleic": NUCLEIC_RESIDUE_NAMES_WITH_VARIANTS | MODIFIED_NUCLEIC_RESIDUE_NAMES,
    }
    masks = {w: mol.atomselect(w) for w in _POLYMER_LINKS}
    names = {w: {n.upper() for n in tables[w]} for w in _POLYMER_LINKS}

    # No `sel` here on purpose: over "all" the returned indices are absolute.
    # Selecting narrows them to positions within the selection.
    _, res_idx = mol.getResidues(return_idx=True)

    for i, atoms in enumerate(res_idx):
        atoms = np.asarray(atoms, dtype=np.int64)
        rname = str(mol.resname[atoms[0]]).upper()
        # Which polymers would take this residue on its name, and do not already
        # hold it. Anything the selections accepted needs no second opinion.
        wanted = {
            w: _POLYMER_LINKS[w]
            for w in _POLYMER_LINKS
            if rname in names[w] and not masks[w][atoms].any()
        }
        if not wanted:
            continue
        for other in (
            res_idx[i - 1] if i else None,
            res_idx[i + 1] if i + 1 < len(res_idx) else None,
        ):
            if other is None:
                continue
            found = {
                kind for _ia, _ib, kind in
                geometric_interresidue_links(mol, atoms, np.asarray(other))
            }
            for w, kind in wanted.items():
                if kind in found:
                    masks[w][atoms] = True
    return masks


def chainResidueMask(mol: Molecule, polymer: str = "protein") -> np.ndarray:
    """Boolean atom mask over the residues of a polymer chain.

    The drop-in for ``mol.atomselect("protein")`` where a residue modelled short of
    its backbone has to count; see :func:`_polymer_masks` for the rule. Capping
    groups are excluded, as ``atomselect`` excludes them: a cap belongs to the
    chain it caps but is not a residue of its sequence. Combine polymers the way
    atomselect masks combine -- ``chainResidueMask(mol, "protein") |
    chainResidueMask(mol, "nucleic")``, which is what ``polymer="both"`` returns.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The molecule. Not modified.
    polymer : str
        ``"protein"``, ``"nucleic"``, or ``"both"``.
    """
    which = ("protein", "nucleic") if polymer == "both" else (polymer,)
    if any(w not in _POLYMER_LINKS for w in which):
        raise ValueError(
            f"polymer accepts one of 'protein', 'nucleic', 'both', not {polymer!r}"
        )
    masks = _polymer_masks(mol)
    out = np.zeros(mol.numAtoms, dtype=bool)
    for w in which:
        out |= masks[w]
    return out


def residuePolymerStatus(mol: Molecule, sel="all"):
    """Yield ``(status, (segid, chain, resid, insertion), atom_indices)`` per
    residue, in file order, with absolute atom indices.

    ``status`` is one of ``"protein"``, ``"nucleic"``, ``"cap"``, ``"water"``,
    ``"ion"``, ``"lipid"`` or ``"other"``. One classification for every caller to
    read as its own purpose requires, rather than each deciding the chemistry
    again: segmentation wants a cap walked with the chain it caps, while a sequence,
    its gaps and its termini must not count a cap as a residue of the sequence.
    Folding ``"cap"`` into ``"protein"`` is what stopped one answer serving both.

    Polymer status comes from :func:`_polymer_masks`, so a residue modelled short of
    its backbone keeps it. The indices index ``mol`` itself, unlike
    ``Molecule.getResidues(sel=...)``, whose indices count within the selection.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The molecule. Not modified.
    sel : str or np.ndarray
        Atom selection to classify. A residue with no selected atom is skipped,
        and one partly selected yields only its selected atoms.
    """
    from moleculekit.residues import (
        CAP_RESIDUE_NAMES,
        ION_RESIDUE_NAMES,
        LIPID_RESIDUE_NAMES,
        METAL_ION_RESIDUE_NAMES,
        WATER_RESIDUE_NAMES,
    )

    masks = _polymer_masks(mol)
    selected = mol.atomselect(sel)
    _, res_idx = mol.getResidues(return_idx=True)

    for atoms in res_idx:
        atoms = np.asarray(atoms, dtype=np.int64)
        atoms = atoms[selected[atoms]]
        if not len(atoms):
            continue
        first = int(atoms[0])
        key = (
            str(mol.segid[first]),
            str(mol.chain[first]),
            int(mol.resid[first]),
            str(mol.insertion[first]),
        )
        resname = str(mol.resname[first])
        if resname in WATER_RESIDUE_NAMES:
            status = "water"
        # The single-atom guard keeps a polyatomic molecule whose code collides
        # with an element symbol (e.g. CO, carbon monoxide) out of this branch.
        elif resname in ION_RESIDUE_NAMES or (
            resname in METAL_ION_RESIDUE_NAMES and len(atoms) == 1
        ):
            status = "ion"
        elif resname in LIPID_RESIDUE_NAMES:
            status = "lipid"
        elif resname in CAP_RESIDUE_NAMES:
            status = "cap"
        elif masks["protein"][atoms].any():
            status = "protein"
        elif masks["nucleic"][atoms].any():
            status = "nucleic"
        else:
            status = "other"
        yield status, key, atoms


def _observed_sequence(mol):
    """``({chain: letters}, {chain: [atom_indices]})`` over the protein chains.

    The shape ``Molecule.getSequence(dict_key="chain", return_idx=True)`` returns,
    over the residues :func:`chainResidueMask` admits, and lettered by the one
    table: a canonical residue by its own code, a known modified one by its
    parent's, anything else ``X``.

    Every consumer of an observed protein sequence reads it from here -- gap
    detection, terminus classification and reference resolution -- so a chain is
    not dropped by one and analysed by the next.
    """
    from moleculekit.molecule import _atoms_to_sequence
    from moleculekit.util import sequenceID

    mask = chainResidueMask(mol)
    seqs, idxs = {}, {}
    for chain in np.unique(mol.chain[mask]):
        sel = mask & (mol.chain == chain)
        if not sel.any():
            continue
        increm = sequenceID((mol.resid[sel], mol.insertion[sel], mol.chain[sel]))
        letters, atoms = _atoms_to_sequence(
            mol, sel, oneletter=True, incremseg=increm, _logger=False
        )
        seqs[str(chain)] = "".join(letters)
        idxs[str(chain)] = atoms
    return seqs, idxs


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
        if next_idx is not None:
            c_terminal |= (curr_chain != next_chain) or not next_has_bb

        is_terminal = n_terminal or c_terminal
        yield prev_idx, curr_idx, next_idx, is_terminal, n_terminal, c_terminal


def _place_carboxylate_oxygen(
    mol: Molecule, c_idx: int, ca_idx: int, o_idx: int, bond_length: float = 1.25
) -> Molecule:
    """Build the second carboxyl oxygen (``OXT``) for a free C-terminal carbonyl.

    The atom is placed in the CA-C-O plane, opposite the average of the C->CA
    and C->O directions (idealised sp2 geometry).

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule containing the carbonyl carbon.
    c_idx : int
        Index of the backbone carbonyl carbon.
    ca_idx : int
        Index of the CA atom bonded to the carbonyl carbon.
    o_idx : int
        Index of the existing carbonyl oxygen.
    bond_length : float
        C-OXT bond length in Angstrom.

    Returns
    -------
    oxt : moleculekit.molecule.Molecule
        A single-atom molecule for the new ``OXT``, inheriting the residue
        identity and carrying a ``-1`` formal charge (deprotonated carboxylate).
    """
    center = mol.coords[c_idx, :, 0]
    u_ca = mol.coords[ca_idx, :, 0] - center
    u_ca = u_ca / np.linalg.norm(u_ca)
    u_o = mol.coords[o_idx, :, 0] - center
    u_o = u_o / np.linalg.norm(u_o)
    direction = -(u_ca + u_o)
    direction = direction / np.linalg.norm(direction)

    oxt = mol.copy(sel=o_idx)
    oxt.name[:] = "OXT"
    oxt.element[:] = "O"
    oxt.coords[:, :, 0] = center + direction * bond_length
    if oxt.formalcharge is not None:
        oxt.formalcharge[:] = -1
    return oxt


def _complete_free_cterm_carboxyls(mol: Molecule) -> int:
    """Add the missing second carboxyl oxygen (``OXT``) to free C-terminal
    carboxyls that a generic backbone template left under-coordinated.

    Targets *non-canonical* residues only: a generic SMILES backbone template
    can represent the backbone carbonyl as a single-oxygen aldehyde, which
    leaves a free alpha-carboxyl as a 3-coordinate carbon (e.g. microcystin's
    D-glutamate / D-methyl-aspartate, whose alpha-carboxyls are free branches).
    Canonical protein residues are left to PDB2PQR, which handles their termini
    and whose chain-gap awareness this bond-based check does not replicate.

    A residue is completed when its backbone carbonyl ``C`` is bonded to a
    single oxygen, no nitrogen (i.e. no peptide bond out) and no second oxygen
    (no ``OXT``). Only a *neutral* gap is completed: a non-zero formal charge on
    the carbonyl carbon or its oxygen signals an intentional state (a templated
    carbanion, or a pre-formed ``[O-]`` carboxylate) and is left untouched. Any
    aldehyde hydrogen the template placed on the carbon is removed. The molecule
    must carry bonds; a bond-less molecule is skipped.

    Parameters
    ----------
    mol : moleculekit.molecule.Molecule
        The molecule to complete in place.

    Returns
    -------
    n_completed : int
        The number of carboxyl groups completed.
    """
    from moleculekit.residues import PROTEIN_RESIDUE_NAMES

    if mol.bonds is None or len(mol.bonds) == 0:
        return 0

    _, res_idx = mol.getResidues(return_idx=True)
    to_add = []  # (residue-key, OXT atom)
    h_remove = []
    for curr_idx in res_idx:
        if str(mol.resname[curr_idx[0]]) in PROTEIN_RESIDUE_NAMES:
            continue  # canonical residues (incl. their termini) are PDB2PQR's job
        names = set(mol.name[curr_idx])
        if not {"N", "CA", "C", "O"} <= names:
            continue
        c_idx = int(curr_idx[mol.name[curr_idx] == "C"][0])
        ca_idx = int(curr_idx[mol.name[curr_idx] == "CA"][0])
        neigh = mol.getNeighbors(c_idx)
        if any(str(mol.element[j]) == "N" for j in neigh):
            continue  # peptide-bonded carbonyl -> already complete
        oxygens = [j for j in neigh if str(mol.element[j]) == "O"]
        if len(oxygens) != 1:
            continue  # not a lone carbonyl (0 = none, >=2 = already a carboxyl)
        o_idx = oxygens[0]
        # Intent guard: only complete a genuinely neutral (accidental) gap. A
        # formal charge means the user templated the state deliberately.
        fc = mol.formalcharge
        if fc is not None and (
            int(round(float(fc[c_idx]))) != 0 or int(round(float(fc[o_idx]))) != 0
        ):
            continue
        key = (
            int(mol.resid[c_idx]),
            str(mol.insertion[c_idx]),
            str(mol.chain[c_idx]),
            str(mol.segid[c_idx]),
        )
        to_add.append((key, _place_carboxylate_oxygen(mol, c_idx, ca_idx, o_idx)))
        h_remove.extend(j for j in neigh if str(mol.element[j]) == "H")

    if not to_add:
        return 0

    # Drop the template's aldehyde H (if any) before inserting, so the carbonyl
    # carbon ends up with the three carboxyl substituents (CA, =O, -OXT).
    if h_remove:
        mask = np.zeros(mol.numAtoms, dtype=bool)
        mask[h_remove] = True
        mol.remove(np.where(mask)[0], _logger=False)

    for (resid, ins, ch, seg), oxt in to_add:
        res_mask = (
            (mol.resid == resid)
            & (mol.insertion == ins)
            & (mol.chain == ch)
            & (mol.segid == seg)
        )
        o_sel = np.where(res_mask & (mol.name == "O"))[0]
        insert_at = int(o_sel[0]) + 1 if len(o_sel) else mol.numAtoms
        mol.insert(oxt, insert_at)
        # Bond the new OXT to the residue's carbonyl C (indices are valid in the
        # post-insert molecule). Downstream builders won't regenerate this bond.
        res_mask = (
            (mol.resid == resid)
            & (mol.insertion == ins)
            & (mol.chain == ch)
            & (mol.segid == seg)
        )
        c_sel = np.where(res_mask & (mol.name == "C"))[0]
        oxt_sel = np.where(res_mask & (mol.name == "OXT"))[0]
        if len(c_sel) and len(oxt_sel):
            mol.addBond(int(c_sel[0]), int(oxt_sel[0]), "1")

    logger.info(
        f"Completed {len(to_add)} free C-terminal carboxyl group(s) by adding OXT"
    )
    return len(to_add)


def check_backbone(
    mol: Molecule,
    remove_broken_terminals: bool = True,
    terminal_min_heavy_atoms: int = 4,
) -> Molecule:
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
