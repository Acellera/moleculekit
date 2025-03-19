import moleculekit.ply.lex as lex
import moleculekit.ply.yacc as yacc

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
    "backbonetype",
    "proteinback",
    "nucleicback",
    "normal",
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
    "NOTEQ",
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
t_NOTEQ = r"\!\="

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


# Don't consider it an integer if it continues with letter.
# (?![A-Za-z]) is a negative look-ahead, i.e. doesn't consume the characters
def t_INTEGER(t):
    r"\d+(?![A-Za-z])"
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


def p_expression_backbonetype(p):
    """
    expression : BACKBONETYPE PROTEINBACK
               | BACKBONETYPE NUCLEICBACK
               | BACKBONETYPE NORMAL
    """
    p[0] = ("backbonetype", p[2])


def p_expression_sameas(p):
    """
    expression : SAME FRAGMENT AS expression
               | SAME molprop_int AS expression
               | SAME molprop_str AS expression
               | SAME numprop_as_str AS expression
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


def p_molecule(p):
    """
    expression : PROTEIN
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
           | DOUBLEEQ
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
                   | molprop_str list
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


def p_molprop_int_comp(p):
    """
    expression : molprop_int compop expression  %prec COMP
    """
    p[0] = ("molprop_int_comp", p[2], p[1], p[3])


def p_molprop_int_modulo(p):
    """
    molprop_int_eq : molprop_int MODULO integer DOUBLEEQ integer
                   | molprop_int MODULO integer NOTEQ integer
    """
    p[0] = ("molprop_int_modulo", p[1], p[3], p[5], p[4])


def p_molprop_int_eq(p):
    """
    molprop_int_eq : molprop_int integer
                   | molprop_int list
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


def p_numprop_eq_list(p):
    """
    numprop : numprop list
    """
    p[0] = ("numprop_list_eq", p[1][1], p[2])


def p_numprop_eq_number(p):
    """
    numprop : numprop number
    """
    p[0] = ("comp", "=", p[1], p[2])


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


def p_numprop_as_str(p):
    """
    numprop_as_str : CHARGE
                   | MASS
                   | OCCUPANCY
                   | BETA
                   | XCOOR
                   | YCOOR
                   | ZCOOR
    """
    p[0] = p[1]


def p_literal_list(p):
    """
    list : list string
         | list integer
         | string string
         | string integer
         | integer integer
         | integer string
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
    if p is None:
        raise RuntimeError("Syntax error")
    raise RuntimeError(f"Syntax error at '{p.value!r}'")


# Build the parser
parser = yacc.yacc()
