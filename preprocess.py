"""Module 2 + 3: Preprocessing and Tokenizer/Normalizer.

Removes comments, whitespace, then turns code into a list of normalized tokens:
  variable/function names -> ID, numbers -> NUM, strings -> STR.
Keywords and operators are kept, so the *structure* of the code is preserved.
"""
import re

KEYWORDS = {
    # Python
    "def", "class", "return", "if", "elif", "else", "for", "while", "in", "import",
    "from", "as", "try", "except", "finally", "with", "lambda", "pass", "break",
    "continue", "and", "or", "not", "is", "None", "True", "False", "print", "range",
    # C / C++ / Java
    "int", "float", "double", "char", "void", "long", "short", "bool", "boolean",
    "switch", "case", "default", "do", "struct", "new", "public", "private",
    "static", "const", "include", "using", "namespace", "std", "cout", "cin",
    "printf", "scanf", "main", "this", "null", "true", "false", "String",
}

COMMENT = r"//[^\n]*|/\*.*?\*/|#[^\n]*"
STRING = (r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|'
          r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
TOKEN_RE = re.compile(
    rf"(?P<comment>{COMMENT})|(?P<string>{STRING})|(?P<num>\b\d+(?:\.\d+)?\b)"
    r"|(?P<id>[A-Za-z_]\w*)|(?P<op>\S)",
    re.DOTALL,
)


def tokenize(code):
    """Return (tokens, lines): normalized tokens and the source line of each."""
    tokens, lines = [], []
    line, last = 1, 0
    for m in TOKEN_RE.finditer(code):
        line += code.count("\n", last, m.start())
        last = m.start()
        kind = m.lastgroup
        text = m.group()
        if kind == "comment":
            continue
        if kind == "string":
            tok = "STR"
        elif kind == "num":
            tok = "NUM"
        elif kind == "id":
            tok = text if text in KEYWORDS else "ID"
        else:
            tok = text
        tokens.append(tok)
        lines.append(line)
    return tokens, lines
