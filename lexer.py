from __future__ import annotations

import re
import sys

RESERVED = "RESERVED"
INT = "INT"
STR = "STR"
ID = "ID"

token_expressions = [
    (r"[ \n\t]+",               None),
    (r"#[^\n]*",                None),
    (r"\:=",                    RESERVED),
    (r"\(",                     RESERVED),
    (r"\)",                     RESERVED),
    (r"\+",                     RESERVED),
    (r"-",                      RESERVED),
    (r"\*",                     RESERVED),
    (r"/",                      RESERVED),
    (r"<=",                     RESERVED),
    (r"<",                      RESERVED),
    (r">=",                     RESERVED),
    (r">",                      RESERVED),
    (r"=",                      RESERVED),
    (r"and",                    RESERVED),
    (r"or",                     RESERVED),
    (r"not",                    RESERVED),
    (r"if",                     RESERVED),
    (r"mod",                    RESERVED),
    (r"defun",                  RESERVED),
    (r"print",                  RESERVED),
    (r"format",                 RESERVED),
    (r"[0-9]+",                 INT),
    (r"([\"\"`])(.*?)\1",       STR),
    (r"[A-Za-z][A-Za-z0-9_]*",  ID),
]


class TokenInfo:
    def __init__(self, tag=None, string=None, pos=None):
        self.tag = tag
        self.string = string
        self.pos = pos


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
