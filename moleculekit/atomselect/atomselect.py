import moleculekit.ply.lex as lex
import moleculekit.ply.yacc as yacc
import numpy as np
from moleculekit.util import sequenceID
from moleculekit.molecule import Molecule

# molecule types
reserved = [
    "protein",
    "nucleic",
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
]
# molecule properties
reserved += [
    "index",
    "serial",
    "name",
    "element",
    "resname",
    "resid",
    "residue",
    "altloc",
    "insertion",
    "chain",
    "segname",
    "segid",
    "mass",
    "charge",
    "occupancy",
    "fragment",
]
# operands
reserved += [
    "and",
    "or",
    "within",
    "exwithin",
    "of",
    "same",
    "as",
    "not",
]
# functions
reserved += ["abs", "sqr"]
# List of token names.   This is always required
tokens = [
    "PLUS",
    "MINUS",
    "TIMES",
    "DIVIDE",
    "LPAREN",
    "RPAREN",
    "ID",
    "INTEGER",
    "FLOAT",
    "NORMSTRING",
    "NORMSTRINGSINGLE",
    "EQUAL",
    "LESSER",
    "GREATER",
    "LESSEREQUAL",
    "GREATEREQUAL",
    "XCOOR",
    "YCOOR",
    "ZCOOR",
] + [rr.upper() for rr in reserved]

# Regular expression rules for simple tokens
t_PLUS = r"\+"
t_MINUS = r"-"
t_TIMES = r"\*"
t_DIVIDE = r"/"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_EQUAL = r"\="
t_LESSER = r"\<"
t_GREATER = r"\>"
t_LESSEREQUAL = r"\<\="
t_GREATEREQUAL = r"\>\="


def t_XCOOR(t):
    r"x"
    return t


def t_YCOOR(t):
    r"y"
    return t


def t_ZCOOR(t):
    r"z"
    return t


def t_ID(t):
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    t.type = "ID"
    if t.value.lower() in reserved:
        t.type = t.value.upper()
    return t


def t_NORMSTRING(t):
    r'"(?:[^"\\]|\\.)*"'
    return t


def t_NORMSTRINGSINGLE(t):
    r"'(?:[^'\\]|\\.)*'"
    return t


def t_FLOAT(t):
    r"(\d*\.\d+)|(\d+\.\d*)"
    t.value = float(t.value)
    return t


def t_INTEGER(t):
    r"\d+"
    t.value = int(t.value)
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


# A string containing ignored characters (spaces and tabs)
t_ignore = " \t"


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()

precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
    ("right", "UMINUS"),
    ("right", "UNOT"),
)


# --------- Grammar rules ---------


def p_expression_logop(p):
    """
    expression : expression AND expression
               | expression OR expression
    """
    p[0] = ("logop", p[2], p[1], p[3])


def p_expression_unary_not(p):
    """
    expression : NOT expression %prec UNOT
    """
    p[0] = ("logop", "not", p[2])


def p_expression_within(p):
    """
    expression : WITHIN number OF expression
    """
    p[0] = ("within", p[2], p[4])


def p_expression_exwithin(p):
    """
    expression : EXWITHIN number OF expression
    """
    p[0] = ("exwithin", p[2], p[4])


def p_expression_sameas(p):
    """
    expression : SAME molecule AS expression
               | SAME molprop AS expression
    """
    p[0] = ("sameas", p[2], p[4])


def p_expression_grouped(p):
    """
    expression : LPAREN expression RPAREN
    """
    p[0] = ("grouped", p[2])


def p_expression_molecule(p):
    """
    expression : molecule
    """
    p[0] = p[1]


def p_molecule(p):
    """
    molecule : PROTEIN
             | NUCLEIC
             | ION
             | IONS
             | LIPID
             | LIPIDS
             | WATER
             | WATERS
             | FRAGMENT
             | BACKBONE
             | HYDROGEN
             | NOH
             | SIDECHAIN
    """
    p[0] = ("molecule", p[1])


def p_number_expression(p):
    """
    expression : number
    """
    p[0] = p[1]


def p_func_number(p):
    """
    number : func
    """
    p[0] = p[1]


def p_num_funcs(p):
    """
    func : ABS LPAREN number RPAREN
         | SQR LPAREN number RPAREN
    """
    p[0] = ("func", p[1], p[3])


def p_prop_funcs(p):
    """
    func : ABS LPAREN numprop RPAREN
         | SQR LPAREN numprop RPAREN
    """
    p[0] = ("func", p[1], p[3])


def p_expression_comp(p):
    """
    expression : comp
    """
    p[0] = p[1]


def p_prop_comp(p):
    """
    comp : numprop EQUAL number
         | numprop LESSER number
         | numprop GREATER number
         | numprop LESSEREQUAL number
         | numprop GREATEREQUAL number
    """
    p[0] = ("comp", p[2], p[1], p[3])


def p_func_comp(p):
    """
    comp : func EQUAL number
         | func LESSER number
         | func GREATER number
         | func LESSEREQUAL number
         | func GREATEREQUAL number
    """
    p[0] = ("comp", p[2], p[1], p[3])


def p_number_comp(p):
    """
    comp : number EQUAL number
         | number LESSER number
         | number GREATER number
         | number LESSEREQUAL number
         | number GREATEREQUAL number
    """
    p[0] = ("comp", p[2], p[1], p[3])


def p_number_mathop(p):
    """
    number : number PLUS number
           | number MINUS number
           | number TIMES number
           | number DIVIDE number
    """
    p[0] = ("mathop", p[2], p[1], p[3])


def p_numprop_mathop(p):
    """
    numprop : numprop PLUS number
            | numprop MINUS number
            | numprop TIMES number
            | numprop DIVIDE number
    """
    p[0] = ("mathop", p[2], p[1], p[3])


def p_expression_molprop(p):
    """
    expression : molprop
    """
    p[0] = p[1]


def p_molprop_string(p):
    """
    molprop : NAME string
            | ELEMENT string
            | RESNAME string
            | ALTLOC string
            | SEGNAME string
            | RESNAME number
            | SEGID string
            | SEGID number
            | INSERTION string
            | INSERTION number
            | CHAIN string
            | CHAIN number
    """
    p[0] = ("molprop", p[1], str(p[2]))


def p_expression_numprop(p):
    """
    expression : numprop
    """
    p[0] = p[1]


def p_molprop_number(p):
    """
    numprop : CHARGE
            | MASS
            | OCCUPANCY
            | XCOOR
            | YCOOR
            | ZCOOR
    """
    p[0] = ("numprop", p[1])


def p_molprop_int(p):
    """
    molprop : INDEX integer
            | SERIAL integer
            | RESID integer
            | RESIDUE integer
    """
    p[0] = ("molprop", p[1], p[2])


def p_string(p):
    """
    string : ID
           | NORMSTRING
           | NORMSTRINGSINGLE
    """
    p[0] = p[1]


def p_number(p):
    """
    number : integer
           | float
    """
    p[0] = p[1]


def p_number_unary_minus(p):
    """
    number : MINUS number %prec UMINUS
    """
    p[0] = ("uminus", p[2])


def p_integer_unary_minus(p):
    """
    integer : MINUS integer %prec UMINUS
    """
    p[0] = ("uminus", p[2])


def p_float_unary_minus(p):
    """
    float : MINUS float %prec UMINUS
    """
    p[0] = ("uminus", p[2])


def p_integer(p):
    """
    integer : INTEGER
    """
    p[0] = p[1]


def p_float(p):
    """
    float : FLOAT
    """
    p[0] = p[1]


def p_error(p):
    print(f"Syntax error at {p.value!r}")


# Build the parser
parser = yacc.yacc()

# Parse an expression
selections = [
    r"not protein",
    r"index -15",
    # r"index 1 3 5", # TODO: Implement this!
    r"name 'A 1'",
    r"chain X",
    r"chain 'y'",
    r"chain 0",
    r'resname "GL"',
    r'resname "GL\*"',
    r"same fragment as lipid",
    r"protein and within 8.3 of resname ACE",
    r"protein and (within -8.3 of resname ACE or exwithin 4 of index 2)",
    r"mass < 5",
    r"mass = 4",
    r"abs(-3)",
    r"abs(charge)",
    r"-sqr(charge)",
    r"abs(charge) > 1",
    r"abs(charge) <= sqr(4)",
    r"x < 6",
    r"x < 6 and x > 3",
    r"sqr(x-5)+sqr(y+4)+sqr(z) > sqr(5)",
]


def print_first_elems(ast):
    print(ast[0])
    for elem in ast:
        if isinstance(elem, tuple):
            print_first_elems(elem)


for sel in selections:
    ast = parser.parse(sel)
    print(f"{sel}:\n   ", ast)
    # print_first_elems(ast)

from moleculekit.atomselect.analyze import analyze


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

    if operation == "molprop":
        matchingprops = (
            "serial",
            "name",
            "element",
            "resname",
            "resid",
            "insertion",
            "chain",
            "segid",
        )
        molprop = node[1]
        value = node[2]
        if molprop in matchingprops:
            return getattr(mol, molprop) == value
        if molprop == "index":
            return np.arange(1, mol.numAtoms + 1) == value
        if molprop == "residue":
            # Unique sequential residue numbering
            residues = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid)) + 1
            return residues == value
        if molprop == "altloc":
            return mol.altloc == value
        if molprop == "segname":
            return mol.segid == value
        if molprop == "mass":
            return mol.masses == value
        if molprop == "charge":
            return mol.charge == value
        if molprop == "occupancy":
            return mol.beta == value
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
        return getattr(mol, node[1])

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
            return np.sqrt(node[2])
        raise RuntimeError(f"Invalid function {fn}")

    raise RuntimeError(f"Invalid operation {operation}")


mol = Molecule("3ptb")
mol.serial[10] = -88
mol.charge[1000:] = -1

analysis = analyze(mol)

tests = [
    "serial 1",
    "serial -88",
    "index 1",
    "resname ILE and (index 2)",
    "chain A",
    "charge >= 0",
    "abs(charge) >= 0",
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
]

for test in tests:
    ast = parser.parse(test)
    print(ast)
    res = traverse_ast(mol, analysis, ast)
    print(res.sum())
