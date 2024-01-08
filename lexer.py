from __future__ import annotations

import re
import sys

RESERVED = "RESERVED"
FUNC = "FUNC"
INT = "INT"
STR = "STR"
ID = "ID"
SPECIAL = "SPECIAL"
MATH = "MATH"
BOOL = "BOOL"
DEFUNC = "DEFUNC"
ARG = "ARG"

token_expressions = [
    (r"[ \n\t]+", None),
    (r"#[^\n]*", None),
    (r"\(", RESERVED),
    (r"\)", RESERVED),
    (r"prints", FUNC),
    (r"printi", FUNC),
    (r"read", FUNC),
    (r"set", SPECIAL),
    (r"if", SPECIAL),
    (r"=", BOOL),
    (r">=", BOOL),
    (r"[-]?[0-9]+", INT),
    (r"\+", MATH),
    (r"-", MATH),
    (r"\*", MATH),
    (r"/", MATH),
    (r"mod", MATH),
    (r"defun", DEFUNC),
    (r"[A-Za-z][A-Za-z_]*:[is]", ARG),
    (r"\"(.*?)\"", STR),
    (r"[A-Za-z][A-Za-z_]*", ID),
]


class TokenInfo:
    def __init__(self, tag: str, string: str, pos: int):
        self.tag = tag
        self.string = string
        self.pos = pos

    def __str__(self):
        return f"TokenInfo(string={self.string},tag={self.tag},pos={self.pos})"


def lex(characters: str, token_exprs: list[tuple[str, str]]):
    pos = 0
    tokens = []
    while pos < len(characters):
        match = None
        for token_expr in token_exprs:
            pattern, tag = token_expr
            regex = re.compile(pattern)
            match = regex.match(characters, pos)
            if match:
                text = match.group(0)
                if tag:
                    tokens.append(TokenInfo(tag, text, pos))
                break
        if not match:
            sys.stderr.write("Illegal character: %s\n" % characters[pos])
            sys.exit(1)
        else:
            pos = match.end(0)
    return tokens
