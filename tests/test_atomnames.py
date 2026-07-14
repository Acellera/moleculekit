import numpy as np


def _mol(names, elements, resids, resnames=None, chains=None, segids=None):
    from moleculekit.molecule import Molecule

    n = len(names)
    mol = Molecule().empty(n)
    mol.name[:] = names
    mol.element[:] = elements
    mol.resid[:] = resids
    mol.resname[:] = resnames if resnames is not None else ["LIG"] * n
    mol.chain[:] = chains if chains is not None else ["A"] * n
    mol.segid[:] = segids if segids is not None else ["0"] * n
    mol.record[:] = ["HETATM"] * n
    mol.coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3, 1)
    return mol


def test_renames_nonunique_residue():
    """A residue whose atom names are bare element symbols (non-unique, as an
    SDF produces) gets unique <Element><index> names."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    mol = _mol(["C", "C", "C", "O"], ["C", "C", "C", "O"], [1, 1, 1, 1])
    out = canonicalizeAtomNames(mol)
    assert list(out.name) == ["C1", "C2", "C3", "O1"]


def test_noop_on_unique_names():
    """A residue that already has unique (force-field-meaningful) names is left
    untouched, so the function is safe to run on canonical residues."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    names = ["N", "CA", "C", "O", "CB"]
    mol = _mol(names, ["N", "C", "C", "O", "C"], [1] * 5, resnames=["ALA"] * 5)
    out = canonicalizeAtomNames(mol)
    assert list(out.name) == names


def test_per_residue_independent():
    """Naming restarts per residue: two copies of the same residue each get the
    same fresh C1/C2/O1 scheme rather than a molecule-wide running counter."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    mol = _mol(
        ["C", "C", "O", "C", "C", "O"],
        ["C", "C", "O", "C", "C", "O"],
        [1, 1, 1, 2, 2, 2],
    )
    out = canonicalizeAtomNames(mol)
    assert list(out.name) == ["C1", "C2", "O1", "C1", "C2", "O1"]


def test_two_char_element_is_title_cased():
    """Two-letter elements are title-cased (CL -> Cl) so the names stay valid
    element-prefixed identifiers."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    mol = _mol(["CL", "CL", "C"], ["CL", "CL", "C"], [1, 1, 1])
    out = canonicalizeAtomNames(mol)
    assert list(out.name) == ["Cl1", "Cl2", "C1"]


def test_mixed_system_leaves_protein_untouched():
    """In a system with a canonical residue (unique names) and a ligand (bare
    element names), only the ligand is renamed."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    mol = _mol(
        ["N", "CA", "C", "O", "C", "C", "O"],
        ["N", "C", "C", "O", "C", "C", "O"],
        [1, 1, 1, 1, 2, 2, 2],
        resnames=["ALA"] * 4 + ["LIG"] * 3,
    )
    out = canonicalizeAtomNames(mol)
    assert list(out.name) == ["N", "CA", "C", "O", "C1", "C2", "O1"]


def test_sel_restricts_scope():
    """With a selection, only residues in scope are canonicalized; other
    non-unique residues are left as-is."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    mol = _mol(
        ["C", "C", "O", "C", "C", "O"],
        ["C", "C", "O", "C", "C", "O"],
        [1, 1, 1, 2, 2, 2],
    )
    out = canonicalizeAtomNames(mol, sel="resid 2")
    assert list(out.name) == ["C", "C", "O", "C1", "C2", "O1"]


def test_returns_copy_without_mutating_input():
    """The input Molecule is not mutated; a canonicalized copy is returned."""
    from moleculekit.tools.atomnames import canonicalizeAtomNames

    mol = _mol(["C", "C", "O"], ["C", "C", "O"], [1, 1, 1])
    out = canonicalizeAtomNames(mol)
    assert list(mol.name) == ["C", "C", "O"]
    assert list(out.name) == ["C1", "C2", "O1"]
