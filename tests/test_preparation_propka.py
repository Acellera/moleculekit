"""Tests for the PROPKA reconciliation helpers.

These exercise the helpers against hand-built PDB2PQR and PROPKA stand-ins
rather than real structures: the behaviour under test is which residues a
helper judges and which it declines to touch, and a stand-in states that
directly instead of hoping a structure happens to contain the case.
"""

def _fake_biomolecule(residues):
    """A stand-in exposing just the ``residues`` list the helpers walk."""

    class _Biomolecule:
        def __init__(self, residues):
            self.residues = residues

    return _Biomolecule(residues)


def _fake_amino(resid, chain, insertion="", n_term=False, c_term=False):
    """A PDB2PQR ``aa.Amino`` stand-in carrying only what the helpers read.

    Subclassing is what matters: ``_pdb2pqr_terminus_decisions`` uses
    ``isinstance(residue, aa.Amino)`` to tell "``set_termini`` judged this and
    declined" apart from "``set_termini`` never looks at this kind of residue".
    """
    from pdb2pqr import aa

    class _Amino(aa.Amino):
        def __init__(self):
            self.res_seq = resid
            self.chain_id = chain
            self.ins_code = insertion
            self.is_n_term = 1 if n_term else 0
            self.is_c_term = 1 if c_term else 0

    return _Amino()


def _fake_het(resid, chain, insertion=""):
    """A non-``aa.Amino`` residue: one ``set_termini`` never judges."""

    class _Het:
        def __init__(self):
            self.res_seq = resid
            self.chain_id = chain
            self.ins_code = insertion

    return _Het()


def _fake_propka_molecule(atoms):
    """A PROPKA container stand-in with a single conformation."""

    class _Conformation:
        def __init__(self, atoms):
            self.atoms = atoms

    class _Molecule:
        def __init__(self, atoms):
            self.conformations = {"1A": _Conformation(atoms)}

    return _Molecule(atoms)


def _fake_propka_atom(terminal, resid, chain, insertion="", resname="ALA"):
    """A PROPKA ``Atom`` stand-in; only ``terminal`` is ever written back."""

    class _Atom:
        def __init__(self):
            self.terminal = terminal
            self.res_num = resid
            self.chain_id = chain
            self.icode = insertion
            self.res_name = resname

    return _Atom()


def test_clear_phantom_termini_drops_cyclic_ring_closure():
    """PROPKA reads termini out of the PDB text it is handed - the first ATOM
    residue after a TER becomes N+ - so a head-to-tail cyclic peptide gets a
    spurious +1 ammonium on the amide nitrogen that closes the ring. PDB2PQR's
    ``set_termini`` already declined that terminus (its cyclic distance guard),
    so the flag must be cleared."""
    from moleculekit.tools.preparation_propka import _clear_phantom_termini

    atom = _fake_propka_atom("N+", 1, "A", resname="DAL")
    molecule = _fake_propka_molecule([atom])
    biomolecule = _fake_biomolecule([_fake_amino(1, "A")])

    assert _clear_phantom_termini(molecule, biomolecule) == 1
    assert atom.terminal is None


def test_clear_phantom_termini_keeps_a_real_terminus():
    """A genuine chain N-terminus that PDB2PQR itself flagged is left alone -
    the reconciliation only ever removes termini, never adds them."""
    from moleculekit.tools.preparation_propka import _clear_phantom_termini

    atom = _fake_propka_atom("N+", 1, "A", resname="GLY")
    molecule = _fake_propka_molecule([atom])
    biomolecule = _fake_biomolecule([_fake_amino(1, "A", n_term=True)])

    assert _clear_phantom_termini(molecule, biomolecule) == 0
    assert atom.terminal == "N+"


def test_clear_phantom_termini_drops_insertion_code_sibling():
    """PROPKA matches the N-terminal residue on the residue NUMBER alone and
    ignores the insertion code, so in chymotrypsin-numbered structures every
    residue sharing that number is flagged. 1A4W's thrombin light chain has
    both ASP1A (the real N-terminus) and a mid-chain CYS1; only the latter's
    flag may be cleared."""
    from moleculekit.tools.preparation_propka import _clear_phantom_termini

    real = _fake_propka_atom("N+", 1, "L", insertion="A", resname="ASP")
    phantom = _fake_propka_atom("N+", 1, "L", resname="CYS")
    molecule = _fake_propka_molecule([real, phantom])
    biomolecule = _fake_biomolecule(
        [
            _fake_amino(1, "L", insertion="A", n_term=True),
            _fake_amino(1, "L"),
        ]
    )

    assert _clear_phantom_termini(molecule, biomolecule) == 1
    assert real.terminal == "N+"
    assert phantom.terminal is None


def test_clear_phantom_termini_leaves_unjudged_residues_alone():
    """``set_termini`` only assigns termini to ``aa.Amino`` residues, so its
    silence about anything else is not a decision. PROPKA's own call stands."""
    from moleculekit.tools.preparation_propka import _clear_phantom_termini

    atom = _fake_propka_atom("C-", 400, "B", resname="LIG")
    molecule = _fake_propka_molecule([atom])
    biomolecule = _fake_biomolecule([_fake_het(400, "B")])

    assert _clear_phantom_termini(molecule, biomolecule) == 0
    assert atom.terminal == "C-"


def test_pdb2pqr_terminus_decisions_reports_only_judged_residues():
    """The ``considered`` set is what separates a declined terminus from one
    PDB2PQR never ruled on."""
    from moleculekit.tools.preparation_propka import _pdb2pqr_terminus_decisions

    biomolecule = _fake_biomolecule(
        [
            _fake_amino(1, "A", n_term=True),
            _fake_amino(9, "A", c_term=True),
            _fake_amino(5, "A"),
            _fake_het(400, "B"),
        ]
    )
    n_term, c_term, considered = _pdb2pqr_terminus_decisions(biomolecule)

    assert n_term == {(1, "A", "")}
    assert c_term == {(9, "A", "")}
    assert considered == {(1, "A", ""), (9, "A", ""), (5, "A", "")}
