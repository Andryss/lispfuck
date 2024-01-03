from __future__ import annotations

import logging
import sys
from enum import Enum

import lexer
from isa import Address, AddressType, Opcode, Term


def extract_tokens(src) -> list[lexer.TokenInfo]:
    return lexer.lex(src, lexer.token_expressions)


class ASTNode:
    def __init__(self, token: lexer.TokenInfo | None = None, parent: ASTNode | None = None,
                 children: list[ASTNode] | None = None):
        self.token = token
        self.parent = parent
        self.children = [] if children is None else children


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
            if token.tag == lexer.STR:
                token.string = token.string[1:-1]
            node.children.append(ASTNode(token, node))
    assert node == root, "Wrong parenthesis count"
    return root


class Type(Enum):
    INT_TYPE = "INT"
    STR_TYPE = "STRING"
    UNK_TYPE = "UNKNOWN"


class FuncInfo:
    def __init__(self, name: str, args: list[Type] = (), ret: Type = Type.UNK_TYPE, code: list[Term] | None = None):
        self.name = name
        self.args = args
        self.ret = ret
        self.code = [] if code is None else code


predefined_funcs: dict[str, FuncInfo] = {
    "prints": FuncInfo("prints", [Type.STR_TYPE], Type.INT_TYPE, [
        Term(Opcode.PUSH),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_INDIRECT_SPR, 0)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
        Term(Opcode.BRANCH_ZERO, Address(AddressType.RELATIVE_IPR, 4)),
        Term(Opcode.STORE, Address(AddressType.ABSOLUTE, 5555)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -5)),
        Term(Opcode.POP),
        Term(Opcode.SUBTRACT, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.POPN),
        Term(Opcode.RETURN),
    ]),
    "printi": FuncInfo("printi", [Type.INT_TYPE], Type.INT_TYPE),
    "read": FuncInfo("read", [], Type.STR_TYPE, [
        Term(Opcode.PUSH),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.EXACT, 0)),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.ABSOLUTE, 6666)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, ord("\n"))),
        Term(Opcode.BRANCH_ZERO, Address(AddressType.RELATIVE_IPR, 8)),
        Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 1)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 1)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, 127)),
        Term(Opcode.BRANCH_ZERO, Address(AddressType.RELATIVE_IPR, 2)),
        Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -8)),
        Term(Opcode.LOAD, Address(AddressType.EXACT, ord("\0"))),
        Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 1)),
        Term(Opcode.POP),
        Term(Opcode.POP),
        Term(Opcode.POP),
        Term(Opcode.RETURN),
    ])
}


class GlobalContext:
    def __init__(self):
        self.function_table: dict[str, FuncInfo] = {}
        self.const_table = {}
        self.const_pointer = 0

    def require_func(self, func: str):
        assert func in predefined_funcs, f"Unknown func {func}"
        self.function_table[func] = predefined_funcs[func]

    def allocate_str_const(self, val: str):
        assert val[0] != '"', f"Value must be trimmed, got {val}"
        assert val[-1] != '"', f"Value must be trimmed, got {val}"
        self.const_table[val] = self.const_pointer
        self.const_pointer += len(val) + 1

    def allocate_int_const(self, val: int):
        assert val < (63 << 1), f"Value must be less than {63 << 1}, got {val}"
        assert val > (- (63 << 1) - 1), f"Value must be greater than {- (63 << 1) - 1}, got {val}"
        self.const_table[val] = self.const_pointer
        self.const_pointer += 1

    def get_const_addr(self, val: str | int):
        assert val in self.const_table, f"unknown const {val}"
        return self.const_table[val]


global_context = GlobalContext()


class Statement:
    def __init__(self, ret_type: Type = Type.UNK_TYPE):
        self.ret_type = ret_type


class InvokeStatement(Statement):
    def __init__(self, ret_type: Type = Type.UNK_TYPE, name: str | None = None, args: list[Statement] | None = None):
        super().__init__(ret_type)
        self.name = name
        self.args = [] if args is None else args


class ConstStatement(Statement):
    def __init__(self, ret_type: Type = Type.UNK_TYPE, val: str | int | None = None):
        super().__init__(ret_type)
        self.val = val


class ValueStatement(Statement):
    def __init__(self, val: int | None = None):
        super().__init__(Type.INT_TYPE)
        self.val = val


def const_statement(node: ASTNode) -> Statement:
    token = node.token
    assert token.tag in (lexer.INT, lexer.STR), f"Unknown const type {token.tag}"
    const_type: Type
    if token.tag == lexer.INT:
        const_type = Type.INT_TYPE
        int_val = int(node.token.string)
        if (31 << 1) > int_val > (- (31 << 1) - 1):
            return ValueStatement(int_val)
        global_context.allocate_int_const(int_val)
        return ConstStatement(const_type, int_val)
    const_type = Type.STR_TYPE
    str_val = node.token.string
    global_context.allocate_str_const(str_val)
    return ConstStatement(const_type, str_val)


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
    return InvokeStatement(func.ret, name, children_statements)


def ast_to_statement(node: ASTNode) -> Statement:
    children = node.children
    if len(children) == 0:
        return const_statement(node)
    return invoke_statement(node)


def extract_statements(root: ASTNode) -> list[Statement]:
    statements = []
    for node in root.children:
        statements.append(ast_to_statement(node))
    return statements


def translate_invoke_statement_argument(arg: Statement) -> list[Term]:
    arg_code = []
    if isinstance(arg, ValueStatement):
        arg_code.append(Term(Opcode.LOAD, Address(AddressType.EXACT, arg.val)))
    elif isinstance(arg, ConstStatement):
        addr = global_context.get_const_addr(arg.val)
        arg_code.append(Term(Opcode.LOAD, Address(AddressType.ABSOLUTE, addr)))
    elif isinstance(arg, InvokeStatement):
        arg_code.append(*translate_invoke_statement(arg))
    else:
        raise NotImplementedError(f"unknown type of InvokeStatement argument, got {arg}")
    return arg_code


def translate_invoke_statement(statement: InvokeStatement) -> list[Term]:
    global_context.require_func(statement.name)
    args = statement.args
    code = []
    for arg in args[-1:0:-1]:
        code.extend(translate_invoke_statement_argument(arg))
        code.append(Term(Opcode.PUSH))
    code.extend(translate_invoke_statement_argument(args[0]))
    code.append(Term(Opcode.CALL, statement.name))
    return code


def translate_statement(statement: Statement) -> list[Term]:
    if isinstance(statement, InvokeStatement):
        return translate_invoke_statement(statement)
    raise NotImplementedError(f"unknown type of Statement to translate, got {statement}")


class Code:
    def __init__(self, context: GlobalContext, start_code: list[Term]):
        self.data_memory: list[tuple[int, int, int | str]] = []

        self.instr_memory: list[tuple[int, int, str, list[Term]]] = []
        self.instr_pointer = 0
        self.func_table = {}

        for symbol, addr in sorted(context.const_table.items(), key=lambda i: i[1]):
            size: int
            if isinstance(symbol, int):
                size = 1
            elif isinstance(symbol, str):
                size = len(symbol) + 1
            else:
                raise NotImplementedError(f"unknown symbol, got {symbol}")
            self.data_memory.append((addr, size, symbol))

        self.instr_memory.append((0, 1, "#", [Term(Opcode.BRANCH_ANY, "start")]))
        self.instr_pointer += 1
        for func_info in context.function_table.values():
            func_code = func_info.code
            self.func_table[func_info.name] = self.instr_pointer
            self.instr_memory.append((self.instr_pointer, len(func_code), func_info.name, func_code))
            self.instr_pointer += len(func_code)

        self.func_table["start"] = self.instr_pointer
        self.instr_memory.append((self.instr_pointer, len(start_code), "start", start_code))

        for block in self.instr_memory:
            for instr in block[3]:
                if isinstance(instr.arg, str):
                    assert instr.arg in self.func_table, f"unknown func, got {instr.arg}"
                    instr.arg = Address(AddressType.ABSOLUTE, self.func_table[instr.arg])

    def __len__(self):
        return sum(map(lambda blk: blk[1], self.instr_memory))


def translate_into_code(statements: list[Statement]) -> Code:
    start_code = []
    for s in statements:
        start_code.extend(translate_statement(s))
    start_code.append(Term(Opcode.HALT))
    return Code(global_context, start_code)


def translate(src):
    logging.debug("Extracting tokens from text")
    tokens = extract_tokens(src)
    logging.debug(f"Extracted {len(tokens)} tokens")

    logging.debug("Building ast from tokens")
    ast = build_ast(tokens)
    logging.debug("Built ast")

    logging.debug("Translating ast to statements")
    statements = extract_statements(ast)
    logging.debug(f"Translated ast to {len(statements)} statements")

    logging.debug("Translating statements into code")
    code = translate_into_code(statements)
    logging.debug(f"Translated to {len(code)} instructions")

    return code


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
