from __future__ import annotations

import logging
import sys

import lexer


class ASTNode:
    def __init__(self, token: str | None, parent: ASTNode | None):
        self.token = token
        self.parent = parent
        self.children = []


class Statement:
    pass


class InvokeStatement:
    def __init__(self, func: str | None, args: list[Statement] | None):
        self.func = func
        self.args = args


class DefunStatement:
    def __init__(self, name: str | None, args: list[str] | None, body: Statement | None):
        self.name = name
        self.args = args
        self.body = body


def build_ast(tokens: list[lexer.TokenInfo]) -> ASTNode:
    root = ASTNode(None)
    node = root
    for token in tokens:
        if token.string == "(":
            child = ASTNode(token, node)
            node.children.append(child)
            node = child
        elif token.string == ")":
            assert node.parent, "Wrong parenthesis count"
            node = node.parent
        else:
            node.children.append(ASTNode(token, node))
    assert node == root, "Wrong parenthesis count"
    return root


def extract_tokens(src) -> list[lexer.TokenInfo]:
    return lexer.lex(src, lexer.token_expressions)


def translate(src):
    logging.debug("Extracting tokens from text")
    tokens = extract_tokens(src)
    logging.debug(f"Extracted {len(tokens)} tokens")

    ast = build_ast(tokens)
    return tokens


def write_code(dst, code):
    pass


def main(src: str, dst: str):
    logging.debug(f"Start reading from {src}")
    with open(src, encoding="utf-8") as f:
        src = f.read()
    preview = (src[:10] + "..." + src[-10:]).replace("\n", "")
    logging.debug(f"Read {preview} from file")

    logging.debug("Start translating text into code")
    code = translate(src)
    logging.debug("Successfully translated into code")

    logging.debug(f"Start writing to {dst}")
    write_code(dst, code)
    logging.debug(f"Successfully wrote {len(code)} code instructions to {dst}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Wrong arguments: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
