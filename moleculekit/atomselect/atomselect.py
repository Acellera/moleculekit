from moleculekit.atomselect.languageparser import parser
from moleculekit.atomselect.analyze import analyze
from scipy.spatial.distance import cdist
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
    "occupancy": "beta",
    "beta": "beta",
    "charge": "charge",
}


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
            bbs = analysis["protein_bb"] | analysis["nucleic_bb"]
            protnuc = analysis["protein"] | analysis["nucleic"]
            return protnuc & ~bbs
        if molec == "protein":
            return analysis["protein"]
        if molec == "nucleic":
            return analysis["nucleic"]
        raise RuntimeError(f"Invalid molecule selection {molec}")

    if operation in ("molprop_int_eq", "molprop_str_eq"):
        molprop = node[1]
        value = node[2]

        # TODO: Improve this with Cython
        def fn(x, y):
            if not isinstance(y, list):
                if not isinstance(y, str) or ".*" not in y:
                    return x == y
                else:
                    return np.array([re.match(y[1:-1], xx) for xx in x], dtype=bool)
            else:
                if not isinstance(y, str) or all([".*" not in yy for yy in y]):
                    return np.isin(x, y)
                else:
                    res = []
                    for xx in x:
                        for yy in y:
                            if ".*" in yy:
                                res.append(re.match(yy[1:-1], xx))
                            else:
                                res.append(xx == yy)
                    return np.array(res, dtype=bool)

        if molprop in molpropmap:
            return fn(getattr(mol, molpropmap[molprop]), value)
        if molprop == "index":
            return fn(np.arange(0, mol.numAtoms), value)
        if molprop == "residue":
            # Unique sequential residue numbering
            return fn(analysis["residues"], value)
        raise RuntimeError(f"Invalid molprop {molprop}")

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
        if op == "=":
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
            if np.any(node[2] < 0):
                raise RuntimeError(f"Negative values in sqr() call: {node[2]}")
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
        cutoff = node[1]
        source = node[2]
        dists = cdist(mol.coords[:, :, mol.frame], mol.coords[source, :, mol.frame])
        idx = np.unique(np.where(dists <= cutoff)[0])
        mask = np.zeros(mol.numAtoms, dtype=bool)
        mask[idx] = True
        if operation == "exwithin":
            mask[source] = False
        return mask

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

        selections = [
            "not protein",
            # "index -15",
            "index 1 3 5",
            "index 1 to 5",
            "name 'A 1'",
            "chain X",
            "chain 'y'",
            "chain 0",
            'resname "GL"',
            r'name "C.*"',
            r'resname "GL.*"',
            "resname ACE NME",
            "same fragment as lipid",
            "protein and within 8.3 of resname ACE",
            "within 8.3 of resname ACE or exwithin 4 of index 2",
            "protein and (within 8.3 of resname ACE or exwithin 4 of index 2)",
            "mass < 5",
            "mass = 4",
            "-sqr(mass) < 0",
            "abs(beta) > 1",
            "abs(beta) <= sqr(4)",
            "x < 6",
            "x > y",
            "(x < 6) and (x > 3)",
            "x < 6 and x > 3",
            "sqr(abs(x-5))+sqr(abs(y+4))+sqr(abs(z)) > sqr(5)",
            "same fragment as resid 5",
            "same residue as within 8 of resid 100",
            "same residue as exwithin 8 of resid 100",
            "same fragment as within 8 of resid 100",
            "serial 1",
            # "serial -88",
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
            "same fragment as resid 17",
            "same resid as resid 17 18",
            "same residue as within 8 of resid 100",
            "same residue as exwithin 8 of resid 100",
            "same fragment as within 8 of resid 100",
        ]

        mol = Molecule("3ptb")
        mol.serial[10] = -88
        mol.beta[:] = 0
        mol.beta[1000:] = -1
        bonds = mol._getBonds(fileBonds=True, guessBonds=True)

        analysis = analyze(mol, bonds)

        for sel in selections:
            with self.subTest(sel=sel):
                mask1, ast = atomselect(
                    mol, sel, bonds, _analysis=analysis, _debug=False, _return_ast=True
                )
                mask2 = mol.atomselect(sel)
                assert np.array_equal(
                    mask1, mask2
                ), f"test: {mask1.sum()} vs ref: {mask2.sum()} atoms. AST:\n{ast}"


if __name__ == "__main__":
    unittest.main(verbosity=2)
