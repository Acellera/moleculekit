from moleculekit.atomselect.languageparser import parser
from moleculekit.atomselect.analyze import analyze
from moleculekit.atomselect_utils import within_distance
import numpy as np
import unittest
import re

molpropmap = {
    "serial": "serial",
    "name": "name",
    "element": "element",
    "resname": "resname",
    "resid": "resid",
    "insertion": "insertion",
    "chain": "chain",
    "segid": "segid",
    "segname": "segid",
    "altloc": "altloc",
    "mass": "masses",
    "occupancy": "occupancy",
    "beta": "beta",
    "charge": "charge",
}


def get_molprop(mol, molprop, analysis):
    if molprop == "fragment":
        return analysis["fragments"]
    if molprop in molpropmap:
        return getattr(mol, molpropmap[molprop])
    if molprop == "index":
        return np.arange(0, mol.numAtoms)
    if molprop == "residue":
        # Unique sequential residue numbering
        return analysis["residues"]
    raise RuntimeError(f"Invalid molecule property {molprop} requested")


def _is_float(val):
    if isinstance(val, (float, np.floating)) or (
        isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
    ):
        return True


def traverse_ast(mol, analysis, node):
    node = list(node)
    operation = node[0]

    # Recurse tree to resolve leaf nodes first
    for i in range(1, len(node)):
        if isinstance(node[i], tuple):
            node[i] = traverse_ast(mol, analysis, node[i])

    if operation == "molecule":
        molec = node[1]
        if molec in ("lipid", "lipids"):
            return analysis["lipids"]
        if molec in ("ion", "ions"):
            return analysis["ions"]
        if molec in ("water", "waters"):
            return analysis["waters"]
        if molec == "hydrogen":
            return mol.element == "H"
        if molec == "noh":
            return mol.element != "H"
        if molec == "backbone":
            return analysis["protein_bb"] | analysis["nucleic_bb"]
        if molec == "sidechain":
            return analysis["sidechain"]
        if molec == "protein":
            return analysis["protein"]
        if molec == "nucleic":
            return analysis["nucleic"]
        raise RuntimeError(f"Invalid molecule selection {molec}")

    if operation in ("molprop_int_eq", "molprop_str_eq"):
        molprop = node[1]
        value = node[2]
        if operation == "molprop_int_eq" and isinstance(value, list):
            value = list(map(int, value))
        if operation == "molprop_str_eq" and isinstance(value, list):
            value = list(map(str, value))

        # TODO: Improve this with Cython
        def fn(x, y):
            if not isinstance(y, list):
                if not isinstance(y, str) or ".*" not in y:
                    return x == y
                else:
                    return np.array([re.match(y, xx) for xx in x], dtype=bool)
            else:
                if not isinstance(y, str) or all([".*" not in yy for yy in y]):
                    return np.isin(x, y)
                else:
                    res = []
                    for xx in x:
                        for yy in y:
                            if ".*" in yy:
                                res.append(re.match(yy, xx))
                            else:
                                res.append(xx == yy)
                    return np.array(res, dtype=bool)

        propvals = get_molprop(mol, molprop, analysis)
        return fn(propvals, value)

    if operation == "molprop_int_modulo":
        # TODO: This can probably be simplified by upgrading it to a comp_op on a numerical property
        molprop = node[1]
        val1 = node[2]
        val2 = node[3]
        oper = node[4]

        propvals = get_molprop(mol, molprop, analysis)
        if oper == "==":
            return (propvals % val1) == val2
        if oper == "!=":
            return (propvals % val1) != val2
        raise RuntimeError(f"Unknown modulo operand {oper}")

    if operation == "logop":
        op = node[1]
        if op == "and":
            return node[2] & node[3]
        if op == "or":
            return node[2] | node[3]
        if op == "not":
            return ~node[2]
        raise RuntimeError(f"Invalid logop {op}")

    if operation == "uminus":
        return -node[1]

    if operation == "grouped":
        return node[1]

    if operation == "numprop":
        val1 = node[1]
        if val1 == "x":
            return mol.coords[:, 0, mol.frame]
        if val1 == "y":
            return mol.coords[:, 1, mol.frame]
        if val1 == "z":
            return mol.coords[:, 2, mol.frame]
        return getattr(mol, molpropmap[val1])

    if operation == "comp":
        op = node[1]
        val1, val2 = node[2], node[3]
        if op in ("=", "=="):
            if _is_float(val1) or _is_float(val2):
                return abs(val1 - val2) < 1e-6
            return val1 == val2
        if op == "<":
            return val1 < val2
        if op == ">":
            return val1 > val2
        if op == "<=":
            return val1 <= val2
        if op == ">=":
            return val1 >= val2
        raise RuntimeError(f"Invalid comparison op {op}")

    if operation == "func":
        fn = node[1]
        if fn == "abs":
            return np.abs(node[2])
        if fn == "sqr":
            return node[2] * node[2]
        if fn == "sqrt":
            if np.any(node[2] < 0):
                raise RuntimeError(f"Negative values in sqrt() call: {node[2]}")
            return np.sqrt(node[2])
        raise RuntimeError(f"Invalid function {fn}")

    if operation == "mathop":
        op = node[1]
        if op == "+":
            fn = lambda x, y: x + y
        if op == "-":
            fn = lambda x, y: x - y
        if op == "*":
            fn = lambda x, y: x * y
        if op == "/":
            fn = lambda x, y: x / y
        val1 = node[2]
        val2 = node[3]
        return fn(val1, val2)

    if operation == "sameas":
        prop = node[1]
        sel = node[2]
        if prop == "fragment":
            selvals = np.unique(analysis["fragments"][sel])
            return np.isin(analysis["fragments"], selvals)
        if prop in molpropmap:
            propvalues = getattr(mol, molpropmap[prop])
            selvals = np.unique(propvalues[sel])
            return np.isin(propvalues, selvals)
        if prop == "residue":
            selvals = np.unique(analysis["residues"][sel])
            return np.isin(analysis["residues"], selvals)
        raise RuntimeError(f"Invalid property {prop} in 'same {prop} as'")

    if operation in ("within", "exwithin"):
        mask = np.zeros(mol.numAtoms, dtype=bool)
        cutoff = node[1]
        source = node[2]
        if not np.any(source):
            return mask

        source_coor = mol.coords[source, :, mol.frame]
        min_source = source_coor.min(axis=0)
        max_source = source_coor.max(axis=0)

        within_distance(
            mol.coords[:, :, mol.frame],
            cutoff,
            np.where(source)[0].astype(np.uint32),
            min_source,
            max_source,
            mask,
        )
        if operation == "exwithin":
            mask[source] = False
        return mask

    if operation == "backbonetype":
        bbtype = node[1]
        if bbtype == "proteinback":
            return analysis["protein_bb"]
        elif bbtype == "nucleicback":
            return analysis["nucleic_bb"]
        elif bbtype == "normal":
            return ~(
                analysis["protein_bb"] | analysis["nucleic_bb"] | (mol.element == "H")
            )
        else:
            raise RuntimeError(
                "backbonetype accepts only one of the following values: (proteinback, nucleicback, normal)"
            )

    raise RuntimeError(f"Invalid operation {operation}")


def atomselect(mol, selection, bonds, _debug=False, _analysis=None, _return_ast=False):
    if _analysis is None:
        _analysis = analyze(mol, bonds)

    try:
        ast = parser.parse(selection, debug=_debug)
    except Exception as e:
        raise RuntimeError(f"Failed to parse selection {selection} with error {e}")

    try:
        mask = traverse_ast(mol, _analysis, ast)
    except Exception as e:
        raise RuntimeError(
            f"Atomselect '{selection}' failed with error '{e}'. AST trace:\n{ast}"
        )
    if _return_ast:
        return mask, ast
    return mask


class _TestAtomSelect(unittest.TestCase):
    def test_atomselect(self):
        from moleculekit.molecule import Molecule
        from moleculekit.atomselect.analyze import analyze
        from moleculekit.home import home
        import pickle
        import time
        import sys
        import os

        selections = [
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
        ]

        pdbids = [
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

        reffile = os.path.join(home(dataDir="test-atomselect"), "selections.pickle")
        write_reffile = False
        time_comp = sys.platform.startswith("linux")
        if not write_reffile:
            with open(reffile, "rb") as f:
                ref = pickle.load(f)

        analysis_time_threshold = 0.4  # second
        atomsel_time_threshold = 0.2
        atomsel_time_threshold_within = 0.7

        results = {}
        for pdbid in pdbids:
            with self.subTest(pdbid=pdbid):
                mol = Molecule(pdbid)
                mol.serial[10] = -88
                mol.beta[:] = 0
                mol.beta[1000:] = -1
                bonds = mol._getBonds(fileBonds=True, guessBonds=True)

                t = time.time()
                analysis = analyze(mol, bonds)
                t = time.time() - t
                if time_comp and t > analysis_time_threshold:
                    raise RuntimeError(
                        f"Analysis took longer than expected {t:.2f} > {analysis_time_threshold:.2f}"
                    )

                for sel in selections:
                    with self.subTest(sel=sel):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
