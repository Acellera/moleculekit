from moleculekit.atomselect.languageparser import parser
from moleculekit.atomselect.analyze import analyze
from moleculekit.atomselect_utils import within_distance
import numpy as np
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

    if operation in ("molprop_int_eq", "molprop_str_eq", "numprop_list_eq"):
        molprop = node[1]
        value = node[2]
        if operation == "molprop_int_eq" and isinstance(value, list):
            value = list(map(int, value))
        if operation == "molprop_str_eq" and isinstance(value, list):
            value = list(map(str, value))
        if operation == "numprop_list_eq" and isinstance(value, list):
            value = list(map(float, value))

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

    if operation == "molprop_int_comp":
        # TODO: This can probably be condensed to "comp" rule by evaluating the molprop itself first
        molprop = node[2]
        val2 = node[3]
        op = node[1]

        propvals = get_molprop(mol, molprop, analysis)
        if op in ("=", "=="):
            if _is_float(val2) or _is_float(propvals):
                return abs(val2 - propvals) < 1e-6
            return propvals == val2
        if op == "<":
            return propvals < val2
        if op == ">":
            return propvals > val2
        if op == "<=":
            return propvals <= val2
        if op == ">=":
            return propvals >= val2
        raise RuntimeError(f"Invalid comparison op {op}")

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
            np.arange(0, mol.numAtoms).astype(np.uint32),
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
