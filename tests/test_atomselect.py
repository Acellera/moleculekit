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
def _test_atomselect(pdbid, sel, _pdbmols):
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


def _test_numprop_list_equality():
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
]


@pytest.mark.parametrize("sel", _PARSER_SELECTIONS)
def _test_parser(sel):
    from moleculekit.atomselect.languageparser import parser

    # Parse an expression
    try:
        parser.parse(sel, debug=False)
    except Exception as e:
        try:
            parser.parse(sel, debug=True)
        except Exception:
            pass
        raise RuntimeError(f"Failed to parse selection '{sel}' with error {e}")
