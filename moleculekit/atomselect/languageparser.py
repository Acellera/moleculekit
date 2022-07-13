import moleculekit.ply.lex as lex
import moleculekit.ply.yacc as yacc
import unittest

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
    "beta",
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
    "to",
]
# functions
reserved += ["abs", "sqr", "sqrt"]
# List of token names.   This is always required
tokens = [
    "PLUS",
    "MINUS",
    "TIMES",
    "MODULO",
    "DIVIDE",
    "LPAREN",
    "RPAREN",
    "STRING",
    "INTEGER",
    "FLOAT",
    "QUOTEDINT",
    "QUOTEDFLOAT",
    "QUOTEDSTRING",
    "QUOTEDSTRINGSINGLE",
    "EQUAL",
    "LESSER",
    "GREATER",
    "LESSEREQUAL",
    "GREATEREQUAL",
    "XCOOR",
    "YCOOR",
    "ZCOOR",
    "DOUBLEEQ",
] + [rr.upper() for rr in reserved]

# Regular expression rules for simple tokens
t_PLUS = r"\+"
t_MINUS = r"-"
t_TIMES = r"\*"
t_MODULO = r"\%"
t_DIVIDE = r"/"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_EQUAL = r"\="
t_LESSER = r"\<"
t_GREATER = r"\>"
t_LESSEREQUAL = r"\<\="
t_GREATEREQUAL = r"\>\="
t_DOUBLEEQ = r"\=\="

# Put the coors here for priority overriding


def t_XCOOR(t):
    r"x"
    return t


def t_YCOOR(t):
    r"y"
    return t


def t_ZCOOR(t):
    r"z"
    return t


def t_QUOTEDINT(t):
    r'"(-?\d+)"|\'(-?\d+)\' '
    t.value = int(t.value[1:-1])
    return t


def t_QUOTEDFLOAT(t):
    r'"(\d*\.\d+)"|"(\d+\.\d*)"|\'(\d*\.\d+)\'|\'(\d+\.\d*)\' '
    t.value = float(t.value[1:-1])
    return t


def t_FLOAT(t):
    r"(\d*\.\d+)|(\d+\.\d*)"
    t.value = float(t.value)
    return t


def t_INTEGER(t):
    r"\d+"
    t.value = int(t.value)
    return t


def t_QUOTEDSTRING(t):
    r'"(?:[^"\\]|\\.)*"'
    t.value = t.value[1:-1]
    return t


def t_QUOTEDSTRINGSINGLE(t):
    r"'(?:[^'\\]|\\.)*'"
    t.value = t.value[1:-1]
    return t


def t_STRING(t):
    r"[a-zA-Z_0-9']+"
    t.type = "STRING"
    if t.value.lower() in reserved:
        t.type = t.value.upper()
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
    ("nonassoc", "AND", "OR"),
    ("nonassoc", "COMP"),
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
    ("right", "UMINUS"),
    ("right", "UNOT"),
)


# --------- Grammar rules ---------


def p_expression_logop(p):
    """
    expression : expression logop expression
    """
    p[0] = ("logop", p[2], p[1], p[3])


def p_logop(p):
    """
    logop : AND
          | OR
    """
    p[0] = p[1]


def p_expression_unary_not(p):
    """
    expression : NOT expression %prec UNOT
    """
    p[0] = ("logop", "not", p[2])


def p_expression_sameas(p):
    """
    expression : SAME FRAGMENT AS expression
               | SAME molprop_int AS expression
               | SAME molprop_str AS expression
    """
    p[0] = ("sameas", p[2], p[4])


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
         | SQRT LPAREN number RPAREN
    """
    p[0] = ("func", p[1], p[3])


def p_prop_funcs(p):
    """
    func : ABS LPAREN numprop RPAREN
         | SQR LPAREN numprop RPAREN
         | SQRT LPAREN numprop RPAREN
    """
    p[0] = ("func", p[1], p[3])


def p_expression_comp(p):
    """
    expression : expression compop expression  %prec COMP
    """
    p[0] = ("comp", p[2], p[1], p[3])


def p_compop(p):
    """
    compop : EQUAL
           | LESSER
           | GREATER
           | LESSEREQUAL
           | GREATEREQUAL
    """
    p[0] = p[1]


def p_number_mathop(p):
    """
    number : number mathop number
    """
    p[0] = ("mathop", p[2], p[1], p[3])


def p_mathop(p):
    """
    mathop : PLUS
           | MINUS
           | TIMES
           | DIVIDE
    """
    p[0] = p[1]


def p_expression_molprop(p):
    """
    expression : molprop_str_eq
               | molprop_int_eq
    """
    p[0] = p[1]


def p_molprop_string_eq(p):
    """
    molprop_str_eq : molprop_str string
                   | molprop_str number
    """
    val2 = p[2]
    if not isinstance(val2, list):
        val2 = str(val2)
    else:
        val2 = [str(x) for x in val2]
    p[0] = ("molprop_str_eq", p[1], val2)


def p_molprop_string(p):
    """
    molprop_str : NAME
                | ELEMENT
                | RESNAME
                | ALTLOC
                | SEGNAME
                | SEGID
                | INSERTION
                | CHAIN
    """
    p[0] = p[1]


def p_molprop_int_modulo(p):
    """
    molprop_int_eq : molprop_int MODULO integer DOUBLEEQ integer
    """
    p[0] = ("molprop_int_modulo", p[1], p[3], p[5])


def p_molprop_int_eq(p):
    """
    molprop_int_eq : molprop_int integer
    """
    p[0] = ("molprop_int_eq", p[1], p[2])


def p_molprop_int(p):
    """
    molprop_int : INDEX
                | SERIAL
                | RESID
                | RESIDUE
    """
    p[0] = p[1]


def p_expression_numprop(p):
    """
    expression : numprop
    """
    p[0] = p[1]


def p_numprop_mathop(p):
    """
    numprop : numprop mathop number
            | numprop mathop numprop
    """
    p[0] = ("mathop", p[2], p[1], p[3])


def p_numprop_number(p):
    """
    numprop : CHARGE
            | MASS
            | OCCUPANCY
            | BETA
            | XCOOR
            | YCOOR
            | ZCOOR
    """
    p[0] = ("numprop", p[1])


def p_string_list(p):
    """
    string : string string
    """
    val1 = p[1]
    val2 = p[2]
    if not isinstance(val1, list):
        val1 = [val1]
    if not isinstance(val2, list):
        val2 = [val2]
    p[0] = val1 + val2


def p_string(p):
    """
    string : STRING
           | QUOTEDSTRING
           | QUOTEDSTRINGSINGLE
    """
    p[0] = p[1]


def p_integer_range(p):
    """
    integer : integer TO integer
    """
    p[0] = list(range(p[1], p[3] + 1))


def p_integer_list(p):
    """
    integer : integer integer
    """
    val1 = p[1]
    val2 = p[2]
    if not isinstance(val1, list):
        val1 = [val1]
    if not isinstance(val2, list):
        val2 = [val2]
    p[0] = val1 + val2


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
            | QUOTEDINT
    """
    p[0] = p[1]


def p_float(p):
    """
    float : FLOAT
          | QUOTEDFLOAT
    """
    p[0] = p[1]


def p_error(p):
    raise RuntimeError(f"Syntax error at '{p.value!r}'")


# Build the parser
parser = yacc.yacc()


class _TestLanguareParser(unittest.TestCase):
    def test_parser(self):
        # Parse an expression
        selections = [
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
        ]

        for sel in selections:
            try:
                _ = parser.parse(sel)
            except Exception as e:
                raise RuntimeError(f"Failed to parse selection {sel} with error {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
