from __future__ import annotations

import re
import sys

RESERVED = "RESERVED"
FUNC = "FUNC"
INT = "INT"
STR = "STR"
ID = "ID"
SPECIAL = "SPECIAL"

token_expressions = [
    (r"[ \n\t]+", None),
    (r"#[^\n]*", None),
    (r"\(", RESERVED),
    (r"\)", RESERVED),
    (r"prints", FUNC),
    (r"printi", FUNC),
    (r"read", FUNC),
    (r"set", SPECIAL),
    (r"[-]?[0-9]+", INT),
    (r"\"(.*?)\"", STR),
    (r"[A-Za-z][A-Za-z0-9_]*", ID),
]


class TokenInfo:
    def __init__(self, tag=None, string=None, pos=None):
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
