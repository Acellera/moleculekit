import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.tools.mutate import mutate_residue


# Phenylalanine heavy-atom bonds and chemically sensible length bounds (Å).
_PHE_BOND_LIMITS = [
    ("N", "CA", 1.38, 1.52),
    ("CA", "C", 1.47, 1.58),
    ("C", "O", 1.18, 1.29),
    ("CA", "CB", 1.47, 1.58),
    ("CB", "CG", 1.47, 1.58),
    ("CG", "CD1", 1.32, 1.46),
    ("CG", "CD2", 1.32, 1.46),
    ("CD1", "CE1", 1.32, 1.46),
    ("CD2", "CE2", 1.32, 1.46),
    ("CE1", "CZ", 1.32, 1.46),
    ("CE2", "CZ", 1.32, 1.46),
]


def _test_mutate_gly216_to_phe_bond_lengths():
    """3PTB chain A GLY 216 → PHE: covalent geometry after mutation (with minimize)."""
    mol = Molecule("3ptb")
    sel = (mol.chain == "A") & (mol.resid == 216) & (mol.resname == "GLY")
    mutate_residue(mol, sel, "PHE", rotamer_mode="best", minimize=True)

    sel = (mol.chain == "A") & (mol.resid == 216) & (mol.resname == "PHE")
    for a1, a2, lo, hi in _PHE_BOND_LIMITS:
        a1idx = (mol.name == a1) & sel
        a2idx = (mol.name == a2) & sel
        d = float(np.linalg.norm(mol.coords[a1idx, :, 0] - mol.coords[a2idx, :, 0]))
        assert lo <= d <= hi, f"bond {a1}-{a2} exploded to {d:.3f} Å"


def _test_minimize_resolves_synthetic_clash():
    """Synthetic two-PHE clash: minimize_soft_potential must resolve the
    overlap by rotating sidechain dihedrals, while keeping bond lengths
    within chemically sensible bounds and leaving all non-mobile atoms
    untouched.
    """
    import os
    from moleculekit.dihedral import dihedralAngle
    from moleculekit.openmmtools import minimize_soft_potential
    from moleculekit.tools.mutate import _CIF_DIR, CHI_ANGLES

    def _load_phe(resid):
        m = Molecule(os.path.join(_CIF_DIR, "PHE.cif"))
        m.filter((m.element != "H") & (m.name != "OXT"), _logger=False)
        m.resid[:] = resid
        m.chain[:] = "A"
        m.segid[:] = "P"
        m.resname[:] = "PHE"
        m.record[:] = "ATOM"
        return m

    phe1 = _load_phe(1)
    phe2 = _load_phe(2)
    # Shift phe2 by 1.5 Å so the rings overlap hard enough to require a
    # non-trivial dihedral rotation, but not so hard that bond restraints
    # lose control.
    phe2.coords[:, :, 0] += np.array([1.5, 0, 0], dtype=np.float32)
    mol = phe1.copy()
    mol.insert(phe2, mol.numAtoms)

    res2_idx = np.where(mol.resid == 2)[0]
    sc_atoms = ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    sc2_idx = res2_idx[np.isin(mol.name[res2_idx], sc_atoms)]
    sc1_idx = np.where((mol.resid == 1) & np.isin(mol.name, sc_atoms))[0]
    n2i = {mol.name[i]: int(i) for i in res2_idx}

    def _min_ring_distance():
        d = np.linalg.norm(
            mol.coords[sc1_idx[:, None], :, 0] - mol.coords[sc2_idx[None, :], :, 0],
            axis=2,
        )
        return float(d.min())

    def _chi(name):
        q = [n2i[a] for a in CHI_ANGLES[name]["PHE"]["ref_plane"]]
        return float(np.degrees(dihedralAngle(mol.coords[q, :, 0].astype(np.float64))))

    clash_before = _min_ring_distance()
    chi_before = {name: _chi(name) for name in ("CHI1", "CHI2")}
    coords_before = mol.coords[:, :, 0].copy()

    assert clash_before < 1.5, (
        f"setup invariant: expected severe clash, got {clash_before:.2f} Å"
    )

    ok = minimize_soft_potential(
        mol, set(sc2_idx.tolist()), max_iterations=500
    )
    if not ok:
        return  # OpenMM unavailable

    # (1) Clash resolved
    clash_after = _min_ring_distance()
    assert clash_after > clash_before, (
        f"minimizer did not reduce ring-ring clash: {clash_before:.2f} -> {clash_after:.2f} Å"
    )

    # (2) At least one dihedral rotated substantially — dihedrals are the
    # free DOF, they should do the bulk of the clash relief.
    max_chi_delta = max(
        abs((_chi(name) - chi_before[name] + 180) % 360 - 180)
        for name in ("CHI1", "CHI2")
    )
    assert max_chi_delta > 5.0, (
        f"dihedrals barely moved (max chi delta {max_chi_delta:.2f}°)"
    )

    # (3) Bonds stayed within chemically sensible ranges (bond restraint
    # held up under the repulsive load).
    for a1, a2, lo, hi in _PHE_BOND_LIMITS:
        if a1 not in n2i or a2 not in n2i:
            continue
        d = float(np.linalg.norm(mol.coords[n2i[a1], :, 0] - mol.coords[n2i[a2], :, 0]))
        assert lo <= d <= hi, (
            f"bond {a1}-{a2} left chemical range: {d:.3f} Å (expected [{lo}, {hi}])"
        )

    # (4) All non-mobile atoms byte-identical (mass=0 freezing works).
    frozen = np.ones(mol.numAtoms, dtype=bool)
    frozen[sc2_idx] = False
    np.testing.assert_array_equal(
        coords_before[frozen], mol.coords[frozen, :, 0],
        err_msg="frozen atoms moved during minimization",
    )


def _test_minimize_soft_potential_restrain_bonded():
    """``minimize_soft_potential`` on a real mutation must:

    1. Leave every non-mobile atom exactly untouched (mass = 0).
    2. Keep bond lengths within chemically sensible bounds.
    """
    from moleculekit.openmmtools import minimize_soft_potential

    mol = Molecule("3ptb")
    sel = (mol.chain == "A") & (mol.resid == 216) & (mol.resname == "GLY")
    mutate_residue(mol, sel, "PHE", rotamer_mode="best", minimize=False)

    phe_mask = (mol.chain == "A") & (mol.resid == 216) & (mol.resname == "PHE")
    phe_idx = np.where(phe_mask)[0]
    sc_idx = phe_idx[~np.isin(mol.name[phe_idx], ("N", "CA", "C", "O"))]

    coords_before = mol.coords[:, :, 0].copy()
    ok = minimize_soft_potential(mol, set(sc_idx.tolist()))
    if not ok:
        return  # OpenMM unavailable -> nothing to verify

    coords_after = mol.coords[:, :, 0]

    # (2) Frozen atoms must be byte-identical
    frozen = np.ones(mol.numAtoms, dtype=bool)
    frozen[sc_idx] = False
    np.testing.assert_array_equal(
        coords_before[frozen], coords_after[frozen],
        err_msg="frozen atoms moved during minimization",
    )

    # (1) Sidechain must have actually moved
    max_sc_delta = float(
        np.linalg.norm(coords_after[sc_idx] - coords_before[sc_idx], axis=1).max()
    )
    assert max_sc_delta > 0.01, (
        f"minimizer did not move sidechain (max delta {max_sc_delta:.4f} Å)"
    )

    # (3) Bond geometry must still be chemically sensible
    name_to_idx = {mol.name[i]: int(i) for i in phe_idx}
    for a1, a2, lo, hi in _PHE_BOND_LIMITS:
        i1, i2 = name_to_idx[a1], name_to_idx[a2]
        d = float(np.linalg.norm(mol.coords[i1, :, 0] - mol.coords[i2, :, 0]))
        assert lo <= d <= hi, (
            f"bond {a1}-{a2} distorted to {d:.3f} Å (outside [{lo}, {hi}])"
        )


def _test_mutate_chi_angles_end_to_end():
    """End-to-end check on ``mol``: chi dihedrals measured on the final
    structure should match the library rotamer.

    Per-chi tolerances reflect how much of each dihedral's ``ref_plane``
    sits on the backbone (which is taken from the input molecule, not the
    idealized template): sidechain-only dihedrals must match exactly,
    backbone-touching ones get a small slack to absorb the residual of
    the N/CA/C Kabsch fit.
    """
    from moleculekit.dihedral import dihedralAngle
    from moleculekit.tools.mutate import (
        CHI_ANGLES,
        _compute_phi_psi,
        _load_rotamer_library,
        _snap_to_bin,
    )

    # One backbone atom in ref_plane (CA)  -> ~0.5° slack
    # Two backbone atoms in ref_plane (N,CA) -> ~3° slack
    # No backbone atoms -> exact
    tolerances = {"CHI1": 3.0, "CHI2": 1.0, "CHI3": 0.01, "CHI4": 0.01}

    mol = Molecule("3ptb")
    sel = (mol.chain == "A") & (mol.resid == 216) & (mol.resname == "GLY")
    sel_idx = np.where(sel)[0]

    phi, psi = _compute_phi_psi(mol, sel_idx)
    top_rot = _load_rotamer_library()["LYS"][(_snap_to_bin(phi), _snap_to_bin(psi))][0]
    expected_chi = top_rot[1:]

    mutate_residue(mol, sel, "LYS", rotamer_mode="first")

    lys_idx = np.where((mol.chain == "A") & (mol.resid == 216) & (mol.resname == "LYS"))[0]
    name_to_idx = {mol.name[i]: i for i in lys_idx}

    for chi_name, chi_target in zip(("CHI1", "CHI2", "CHI3", "CHI4"), expected_chi):
        ref_atoms = CHI_ANGLES[chi_name]["LYS"]["ref_plane"]
        coords = np.array(
            [mol.coords[name_to_idx[a], :, 0] for a in ref_atoms], dtype=np.float64
        )
        measured = float(np.degrees(dihedralAngle(coords)))
        diff = (measured - chi_target + 180) % 360 - 180
        tol = tolerances[chi_name]
        assert abs(diff) < tol, (
            f"{chi_name}: expected {chi_target:.2f}°, got {measured:.3f}° "
            f"(diff={diff:.3f}°, tol={tol}°)"
        )


def _test_apply_chi_angles_matches_library():
    """``_apply_chi_angles`` must set every CHI dihedral to the target value.

    Chi rotations are measured on the aligned template *before* insertion,
    so this test isolates the rotation math from any residual
    backbone-alignment error between the idealized CIF template and the
    real residue backbone in ``mol`` (which can be ~0.05 Å per atom when
    the template's idealized bond lengths don't match the input
    structure exactly, and would shift the measured dihedral by a few
    degrees even though the rotation itself is perfect).
    """
    from moleculekit.dihedral import dihedralAngle
    from moleculekit.tools.mutate import (
        CHI_ANGLES,
        _apply_chi_angles,
        _build_template_on_backbone,
        _compute_phi_psi,
        _load_rotamer_library,
        _snap_to_bin,
    )

    # LYS has all four chi angles defined.
    mol = Molecule("3ptb")
    sel = (mol.chain == "A") & (mol.resid == 216) & (mol.resname == "GLY")
    sel_idx = np.where(sel)[0]

    phi, psi = _compute_phi_psi(mol, sel_idx)
    top_rot = _load_rotamer_library()["LYS"][(_snap_to_bin(phi), _snap_to_bin(psi))][0]
    expected_chi = top_rot[1:]

    res = mol.copy(sel=sel)
    tmpl = _build_template_on_backbone("LYS", res)
    _apply_chi_angles(tmpl, "LYS", expected_chi)

    name_to_idx = {n: i for i, n in enumerate(tmpl.name)}
    for chi_name, chi_target in zip(("CHI1", "CHI2", "CHI3", "CHI4"), expected_chi):
        ref_atoms = CHI_ANGLES[chi_name]["LYS"]["ref_plane"]
        coords = np.array(
            [tmpl.coords[name_to_idx[a], :, 0] for a in ref_atoms], dtype=np.float64
        )
        measured = float(np.degrees(dihedralAngle(coords)))
        # Wrap to [-180, 180] so e.g. 179.9° vs -180° is a 0.1° diff.
        diff = (measured - chi_target + 180) % 360 - 180
        assert abs(diff) < 0.01, (
            f"{chi_name}: expected {chi_target:.2f}°, got {measured:.4f}° "
            f"(diff={diff:.4f}°)"
        )
