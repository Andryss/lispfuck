import sys
import logging
import tokenize


class ASTNode:
    def __init__(self, token=None, parent=None):
        self.token = token
        self.parent = parent
        self.children = []


def build_ast(tokens: list[tokenize.TokenInfo]) -> ASTNode:
    root = ASTNode(None)
    node = root
    for token in tokens:
        if token.string == '(':
            child = ASTNode(token, node)
            node.children.append(child)
            node = child
        elif token.string == ')':
            node = node.parent
        else:
            node.children.append(ASTNode(token, node))
    return root


def extract_tokens(src) -> list[tokenize.TokenInfo]:
    t = []
    with tokenize.open(src) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for token in tokens:
            if token.type in [tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER]:
                continue
            t.append(token)
    return t


def translate(src):
    tokens = extract_tokens(src)
    ast = build_ast(tokens)
    return tokens


def write_code(dst, code):
    pass


def main(src: str, dst: str):
    logging.debug(f'Start reading from {src}')
    code = translate(src)
    logging.debug(f'Successfully translated into code')

    logging.debug(f'Start writing to {dst}')
    write_code(dst, code)
    logging.debug(f'Successfully wrote {len(code)} code instructions to {dst}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Wrong arguments: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
