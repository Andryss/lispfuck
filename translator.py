from __future__ import annotations

import logging
import sys

import lexer


class ASTNode:
    def __init__(self, token: lexer.TokenInfo | None = None, parent: ASTNode | None = None,
                 children: list[ASTNode] | None = None):
        self.token = token
        self.parent = parent
        self.children = [] if children is None else children


INT_TYPE = "INT"
STR_TYPE = "STRING"
UNK_TYPE = "UNKNOWN"


class FuncInfo:
    def __init__(self, name: str, args: list[str] = (), ret: str = UNK_TYPE):
        self.name = name
        self.args = args
        self.ret = ret
        
        
predefined_funcs: dict[str, FuncInfo] = {
    "prints": FuncInfo("prints", [STR_TYPE], INT_TYPE),
    "printi": FuncInfo("printi", [INT_TYPE], INT_TYPE),
    "read": FuncInfo("read", [], STR_TYPE)
}


class Statement:
    def __init__(self, name: str | None = None, ret_type: str = UNK_TYPE, args: list[Statement] | None = None):
        self.name = name
        self.ret_type = ret_type
        self.args = [] if args is None else args
        
        
def const_statement(node: ASTNode) -> Statement:
    token = node.token
    assert token.tag in (lexer.INT, lexer.STR), f"Unknown const type {token.tag}"
    const_type: str
    if token.tag == lexer.INT:
        const_type = INT_TYPE
    else:
        const_type = STR_TYPE
    return Statement(node.token.string, const_type)


def invoke_statement(node: ASTNode) -> Statement:
    children = node.children
    assert len(children[0].children) == 0, "Name of statement must be string"
    name = children[0].token.string
    assert name in predefined_funcs, f"Unknown func {name}"
    func = predefined_funcs[name]
    children_nodes = children[1:]
    assert len(func.args) == len(children_nodes), f"Wrong amount of arguments for func {name}"
    children_statements = []
    for i in range(len(children_nodes)):
        statement = ast_to_statement(children_nodes[i])
        assert func.args[i] == statement.ret_type, f"Wrong ret_type of {name} {i} argument, " \
                                                   f"expected: {func.args[i]}, got: {statement.ret_type}"
        children_statements.append(statement)
    return Statement(name, func.ret, children_statements)
        
        
def ast_to_statement(node: ASTNode) -> Statement:
    children = node.children
    if len(children) == 0:
        return const_statement(node)
    else:
        return invoke_statement(node)


def ast_root_to_statements(root: ASTNode) -> list[Statement]:
    statements = []
    for node in root.children:
        statements.append(ast_to_statement(node))
    return statements
            
        
def build_ast(tokens: list[lexer.TokenInfo]) -> ASTNode:
    root = ASTNode()
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

    logging.debug("Building ast from tokens")
    ast = build_ast(tokens)
    logging.debug("Built ast")

    logging.debug("Translating ast to statements")
    statements = ast_root_to_statements(ast)
    logging.debug(f"Translated ast to {len(statements)} statements")

    print(statements)
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
