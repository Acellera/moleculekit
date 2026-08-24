from moleculekit.molecule import Molecule
import numpy as np
import pytest
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))

_SELECTIONS = [
    "not protein",
    "index 1 3 5",
    "index 1 to 5",
    "serial % 2 == 0",
    "resid -27",
    'resid "-27"',
    "name 'A 1'",
    "chain X",
    "chain 'y'",
    "chain 0",
    'resname "GL"',
    'name "C.*"',
    'resname "GL.*"',
    "resname ACE NME",
    "same fragment as lipid",
    "protein and within 8.3 of resname ALA",
    "within 8.3 of resname ALA or exwithin 4 of index 2",
    "protein and (within 8.3 of resname ALA or exwithin 4 of index 2)",
    "mass < 5",
    "mass = 4",
    "-sqr(mass) < 0",
    "abs(beta) > 1",
    "abs(beta) <= sqr(4)",
    "x < 6",
    "x > y",
    "(x < 6) and (x > 3)",
    "x < 6 and x > 3",
    "x > sqr(5)",
    "(x + y) > sqr(5)",
    "sqr(abs(x-5))+sqr(abs(y+4))+sqr(abs(z)) > sqr(5)",
    "sqrt(abs(x-5))+sqrt(abs(y+4))+sqrt(abs(z)) > sqrt(5)",
    "same fragment as resid 5",
    "same residue as within 8 of resid 100",
    "same residue as exwithin 8 of resid 100",
    "same fragment as within 8 of resid 100",
    "serial 1",
    "index 1",
    "index 1 2 3",
    "index 1 to 5",
    "resname ILE and (index 2)",
    "resname ALA ILE",
    "chain A",
    "beta >= 0",
    "abs(beta) >= 0",
    "lipid",
    "lipids",
    "ion",
    "ions",
    "water",
    "waters",
    "noh",
    "hydrogen",
    "backbone",
    "sidechain",
    "protein",
    "nucleic",
    "residue 0",
    "beta + 5 >= 2+3",
    "within 5 of nucleic",
    "exwithin 5 of nucleic",
    "same fragment as resid 17",
    "same resid as resid 17 18",
    "same residue as within 8 of resid 100",
    "same residue as exwithin 8 of resid 100",
    "same fragment as within 8 of resid 100",
    "nucleic and name C3'",
    'resname C8E GR4 "200" 1PE',
    "occupancy 0",
    "occupancy = 0",
    "occupancy == 0",
    "(occupancy 1) and same beta as exwithin 3 of (occupancy 0)",
    "backbonetype proteinback",
    "backbonetype nucleicback",
    "backbonetype normal",
    "backbonetype proteinback and residue 15 to 20",
    "resid < 20",
]

_PDBIDS = [
    "3ptb",
    "3wbm",
    "4k98",
    "3hyd",
    "6a5j",
    "5vbl",
    "7q5b",
    "1unc",
    "3zhi",
    "1a25",
    "1u5u",
    "1gzm",
    "6va1",
    "1bna",
    "1awf",
    "5vav",
    "2p09",
]


@pytest.fixture(scope="module")
def _pdbmols():
    return {pdbid: Molecule(pdbid) for pdbid in _PDBIDS}


@pytest.mark.parametrize("pdbid", _PDBIDS)
@pytest.mark.parametrize("sel", _SELECTIONS)
def test_atomselect(pdbid, sel, _pdbmols):
    from moleculekit.atomselect.analyze import analyze
    from moleculekit.atomselect.atomselect import atomselect
    import pickle
    import time
    import sys

    reffile = os.path.join(curr_dir, "test_atomselect", "selections.pickle")
    write_reffile = False
    time_comp = (
        sys.platform.startswith("linux")
        and os.environ.get("SKIP_SPEED_TESTS", None) is None
    )
    if not write_reffile:
        with open(reffile, "rb") as f:
            ref = pickle.load(f)

    analysis_time_threshold = 0.4  # second
    atomsel_time_threshold = 0.2
    atomsel_time_threshold_within = 0.7

    results = {}

    mol = _pdbmols[pdbid]
    mol.serial[10] = -88
    mol.beta[:] = 0
    mol.beta[1000:] = -1
    bonds = mol._getBonds(fileBonds=False, guessBonds=True)

    t = time.time()
    analysis = analyze(mol, bonds)
    t = time.time() - t
    if time_comp and t > analysis_time_threshold:
        raise RuntimeError(
            f"Analysis took longer than expected {t:.2f} > {analysis_time_threshold:.2f}"
        )

    t = time.time()
    mask, ast = atomselect(
        mol,
        sel,
        bonds,
        _analysis=analysis,
        _debug=False,
        _return_ast=True,
    )
    indices = np.where(mask)[0].tolist()
    t = time.time() - t
    if time_comp:
        if "within" in sel and t > atomsel_time_threshold_within:
            raise RuntimeError(
                f"Atom selection took longer than expected {t:.2f} > {atomsel_time_threshold_within:.2f} for sel {sel}"
            )
        elif "within" not in sel and t > atomsel_time_threshold:
            raise RuntimeError(
                f"Atom selection took longer than expected {t:.2f} > {atomsel_time_threshold:.2f} for sel {sel}"
            )

    if write_reffile:
        results[(pdbid, sel)] = indices
    else:
        assert np.array_equal(
            indices, ref[(pdbid, sel)]
        ), f"test: {len(indices)} vs ref: {len(ref[(pdbid, sel)])} atoms. AST:\n{ast}"

    if write_reffile:
        with open(reffile, "wb") as f:
            pickle.dump(results, f)


def test_empty_molecule():
    mol = Molecule()
    selections = [
        "all",
        "protein",
        "nucleic",
        "water",
        "lipid",
        "ion",
        "backbone",
        "sidechain",
        "hydrogen",
        "noh",
        "name CA",
        "resname ALA",
        "resid 1",
        "chain A",
        "index 0",
        "serial 1",
        "element C",
        "mass < 5",
        "x < 6",
        "beta >= 0",
        "not protein",
    ]
    for sel in selections:
        res = mol.atomselect(sel)
        assert res.shape == (0,), f"Expected empty result for '{sel}', got shape {res.shape}"
        assert res.dtype == bool, f"Expected bool dtype for '{sel}', got {res.dtype}"


def test_single_atom_molecule():
    mol = Molecule()
    mol.empty(1)
    mol.record[:] = "ATOM"
    mol.name[:] = "CA"
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.element[:] = "C"
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)

    expected_true = [
        "all",
        "name CA",
        "resname ALA",
        "resid 1",
        "chain A",
        "element C",
        "index 0",
        "serial 1",
        "noh",
        "not nucleic",
        "not water",
        "x < 6",
        "beta >= 0",
    ]
    for sel in expected_true:
        res = mol.atomselect(sel)
        assert res.shape == (1,), f"Expected shape (1,) for '{sel}', got {res.shape}"
        assert res.dtype == bool
        assert res[0], f"Expected True for '{sel}'"

    expected_false = [
        "nucleic",
        "water",
        "lipid",
        "ion",
        "hydrogen",
        "name CB",
        "resname GLY",
        "resid 2",
        "chain B",
        "element N",
        "index 1",
    ]
    for sel in expected_false:
        res = mol.atomselect(sel)
        assert res.shape == (1,), f"Expected shape (1,) for '{sel}', got {res.shape}"
        assert res.dtype == bool
        assert not res[0], f"Expected False for '{sel}'"


def test_numprop_list_equality():
    pdb = os.path.join(curr_dir, "test_atomselect", "test.pdb")
    mol = Molecule(pdb)
    selections = ["beta 1 2", "beta 2 3"]
    expected = [
        [False, True, False, False, True, False, False, False],
        [True, True, True, True, False, False, True, True],
    ]
    for sel, exp in zip(selections, expected):
        res = mol.atomselect(sel)
        assert np.array_equal(res, exp), f"{sel}\n{res}\n{exp}"


_PARSER_SELECTIONS = [
    "not protein",
    "index -15",
    "index 1 3 5",
    "index 1 to 5",
    "name 'A 1'",
    "chain X",
    "chain 'y'",
    "chain 0",
    'resname "GL"',
    r'resname "GL\*"',
    "resname 1PE",
    "resname PE1",
    'resid "-27"',
    'resname C8E GR4 "200" 1PE',
    "resname ACE NME",
    "same fragment as lipid",
    "protein and within 8.3 of resname ACE",
    "protein and (within -8.3 of resname ACE or exwithin 4 of index 2)",
    "mass < 5",
    "mass = 4",
    "abs(-3)",
    "abs(charge)",
    "-sqr(charge)",
    "abs(charge) > 1",
    "abs(charge) <= sqr(4)",
    "x < 6",
    "x > y",
    "x < 6 and x > 3",
    "sqr(x-5)+sqr(y+4)+sqr(z) > sqr(5)",
    "same fragment as resid 5",
    "same residue as within 8 of resid 100",
    "same residue as exwithin 8 of resid 100",
    "same fragment as within 8 of resid 100",
    "nucleic and name C3'",
    "serial % 2 == 0",
    "resname WAT and serial % 2 == 0",
    "resname WAT and index % 2 == 0",
    "resid 1 5 7 to 20 25",
    "occupancy 1",
    "occupancy = 1",
    "occupancy == 1",
    "(occupancy 1) and same beta as exwithin 3 of (occupancy 0)",
    "backbonetype proteinback or backbonetype nucleicback or backbonetype normal",
    "beta 2 3",
    "resid < 20",
    "formalcharge 1",
    "formalcharge -1",
    "formalcharge \"-1\" 1",
    "formalcharge > 0",
    "formalcharge < 0",
    "not formalcharge 0",
]


@pytest.mark.parametrize("sel", _PARSER_SELECTIONS)
def test_parser(sel):
    from moleculekit.atomselect._languageparser import parser

    # Parse an expression
    try:
        parser.parse(sel, debug=False)
    except Exception as e:
        try:
            parser.parse(sel, debug=True)
        except Exception:
            pass
        raise RuntimeError(f"Failed to parse selection '{sel}' with error {e}")


def test_formalcharge_selection():
    """formalcharge is a per-atom field but was absent from the selection
    grammar, so callers hand-wrote substitutes. It belongs in the integer
    family with resid and serial."""
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    assert np.all(mol.formalcharge == 0), "a freshly read structure has no charges"

    idx = mol.atomselect("resname BEN", indexes=True)
    mol.formalcharge[idx[0]] = 1
    mol.formalcharge[idx[1]] = -1
    both = sorted([int(idx[0]), int(idx[1])])

    assert np.array_equal(mol.atomselect("formalcharge 1", indexes=True), [idx[0]])
    assert np.array_equal(mol.atomselect("formalcharge -1", indexes=True), [idx[1]])
    assert np.array_equal(mol.atomselect('formalcharge "-1"', indexes=True), [idx[1]])
    assert np.array_equal(
        mol.atomselect('formalcharge "-1" 1', indexes=True), both
    )
    assert np.array_equal(mol.atomselect("formalcharge > 0", indexes=True), [idx[0]])
    assert np.array_equal(mol.atomselect("formalcharge < 0", indexes=True), [idx[1]])
    assert np.array_equal(mol.atomselect("not formalcharge 0", indexes=True), both)

    # Composable with the rest of the language
    assert mol.atomselect("resname BEN and not formalcharge 0").sum() == 2
    assert mol.atomselect("protein and not formalcharge 0").sum() == 0
    assert mol.atomselect("same residue as formalcharge 1").sum() == len(idx)


@pytest.mark.parametrize(
    "form",
    [
        "{} 1",
        "{} -1",
        '{} "-1"',
        '{} "-1" 1',
        "{} > 0",
        "{} < 0",
        "{} >= 0",
        "{} <= 0",
        "{} > -1",
        "not {} 0",
        "{} != 0",
        "{} -1 1",
        "{} -1 to 1",
    ],
)
def test_formalcharge_parses_exactly_like_resid(form):
    """The design claim for this change is that formalcharge joined the INTEGER
    property family, alongside resid and serial. So the test is parity: every
    form must succeed or fail for formalcharge exactly as it does for resid.

    This also pins the three forms the language does not support (`!=`, an
    unquoted negative in a list, a negative in a range). They are pre-existing
    limits shared with resid, not quirks of this field. If someone later teaches
    the grammar one of them, this test fails for both fields at once and says so.
    """
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))

    def parses(sel):
        try:
            mol.atomselect(sel)
            return True
        except Exception:
            return False

    assert parses(form.format("formalcharge")) == parses(form.format("resid")), (
        f"{form.format('formalcharge')!r} and {form.format('resid')!r} disagree, "
        f"so formalcharge is not behaving as an integer property"
    )
