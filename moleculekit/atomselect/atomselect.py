import moleculekit.ply.lex as lex
import moleculekit.ply.yacc as yacc

reserved = [
    "protein",
    "nucleic",
    "lipid",
    "ion",
    "resname",
    "resid",
    "residue",
    "index",
    "serial",
    "and",
    "or",
    "within",
    "exwithin",
    "of",
    "same",
    "as",
]
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
] + [rr.upper() for rr in reserved]

# Regular expression rules for simple tokens
t_PLUS = r"\+"
t_MINUS = r"-"
t_TIMES = r"\*"
t_DIVIDE = r"/"
t_LPAREN = r"\("
t_RPAREN = r"\)"


def t_ID(t):
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    t.type = "ID"
    if t.value.lower() in reserved:
        t.type = t.value.upper()
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
    ("right", "UPLUS", "UMINUS"),
)


# Write functions for each grammar rule which is
# specified in the docstring.
def p_expression_binop(p):
    """
    expression : expression PLUS expression
               | expression MINUS expression
               | expression TIMES expression
               | expression DIVIDE expression
    """
    p[0] = ("binop", p[2], p[1], p[3])


def p_expression_logop(p):
    """
    expression : expression AND expression
               | expression OR expression
    """
    p[0] = ("logop", p[2], p[1], p[3])


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
    p[0] = ("exwithin", p[2], p[4])


def p_expression_id(p):
    """
    expression : ID
    """
    p[0] = ("name", p[1])


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
             | LIPID
    """
    p[0] = ("molecule", p[1])


def p_expression_molprop(p):
    """
    expression : molprop
    """
    p[0] = p[1]


def p_molprop_resname(p):
    """
    molprop : RESNAME ID
    """
    p[0] = ("resname", p[2])


def p_molprop_resid(p):
    """
    molprop : RESID number
            | RESIDUE number
    """
    p[0] = (p[1], p[2])


def p_molprop_indexserial(p):
    """
    molprop : INDEX number
            | SERIAL number
    """
    p[0] = (p[1], p[2])


def p_number_unary(p):
    """
    number : PLUS number %prec UPLUS
           | MINUS number %prec UMINUS
    """
    p[0] = ("unary", p[1], p[2])


def p_expression_grouped(p):
    """
    expression : LPAREN expression RPAREN
    """
    p[0] = ("grouped", p[2])


def p_number(p):
    """
    number : INTEGER
           | FLOAT
    """
    p[0] = ("number", p[1])


def p_error(p):
    print(f"Syntax error at {p.value!r}")


# Build the parser
parser = yacc.yacc()

# Parse an expression
ast = parser.parse("index -15")
print(ast)
ast = parser.parse("protein and within 8.3 of resname ACE")
print(ast)
ast = parser.parse("protein and (within -8.3 of resname ACE or exwithin 4 of index 2)")
print(ast)
