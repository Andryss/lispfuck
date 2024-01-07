from __future__ import annotations

import logging
import sys
from enum import Enum

import lexer
from isa import Address, AddressType, Opcode, Term


def extract_tokens(src) -> list[lexer.TokenInfo]:
    return lexer.lex(src, lexer.token_expressions)


class ASTNode:
    def __init__(
        self, token: lexer.TokenInfo | None = None, parent: ASTNode | None = None, children: list[ASTNode] | None = None
    ):
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
    "prints": FuncInfo(
        "prints",
        [Type.STR_TYPE],
        Type.INT_TYPE,
        [
            Term(Opcode.PUSH),
            Term(Opcode.PUSH),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_INDIRECT_SPR, 0)),
            Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
            Term(Opcode.BRANCH_ZERO, Address(AddressType.RELATIVE_IPR, 4)),
            Term(Opcode.STORE, Address(AddressType.ABSOLUTE, 5556)),
            Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -5)),
            Term(Opcode.POP),
            Term(Opcode.SUBTRACT, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.POPN),
            Term(Opcode.RETURN),
        ],
    ),
    "printi": FuncInfo("printi", [Type.INT_TYPE], Type.INT_TYPE),
    "read": FuncInfo(
        "read",
        [],
        Type.STR_TYPE,
        [
            Term(Opcode.PUSH),
            Term(Opcode.PUSH),
            Term(Opcode.LOAD, Address(AddressType.EXACT, 0)),
            Term(Opcode.PUSH),
            Term(Opcode.LOAD, Address(AddressType.ABSOLUTE, 5555)),
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
        ],
    ),
}


class GlobalContext:
    def __init__(self):
        self.function_table: dict[str, FuncInfo] = {}
        self.const_table: set[str | int] = set()
        self.anon_var_table: dict[str, int] = {}
        self.anon_var_counter = 0

    def require_func(self, func: str):
        assert func in predefined_funcs, f"Unknown func {func}"
        self.function_table[func] = predefined_funcs[func]

    def require_int_const(self, const: int):
        assert const < (1 << 63), f"Value must be less than {1 << 63}, got {const}"
        assert const > (-(1 << 63) - 1), f"Value must be greater than {- (1 << 63) - 1}, got {const}"
        self.const_table.add(const)

    def require_str_const(self, const: str):
        assert const[0] != '"', f"Value must be trimmed, got {const}"
        assert const[-1] != '"', f"Value must be trimmed, got {const}"
        self.const_table.add(const)

    def require_anon_variable(self, size: int) -> str:
        assert size > 0, f"negative size buffer?, got {size}"
        name = f"anon${self.anon_var_counter}"
        self.anon_var_counter += 1
        self.anon_var_table[name] = size
        return name


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
        assert val < (1 << 19), f"Value must be less than {1 << 19}, got {val}"
        assert val > (-(1 << 19) - 1), f"Value must be greater than {- (1 << 19) - 1}, got {val}"
        super().__init__(Type.INT_TYPE)
        self.val = val


def const_statement(node: ASTNode) -> Statement:
    token = node.token
    assert token.tag in (lexer.INT, lexer.STR), f"Unknown const type {token.tag}"
    if token.tag == lexer.INT:
        const_type = Type.INT_TYPE
        int_val = int(node.token.string)
        if (19 << 1) > int_val > (-(19 << 1) - 1):
            return ValueStatement(int_val)
        global_context.require_int_const(int_val)
        return ConstStatement(const_type, int_val)
    if token.tag == lexer.STR:
        const_type = Type.STR_TYPE
        str_val = node.token.string
        global_context.require_str_const(str_val)
        return ConstStatement(const_type, str_val)
    raise NotImplementedError(f"unknown lex tag of ConstStatement, got {token.tag}")


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
        assert func.args[i] == statement.ret_type, (
            f"Wrong ret_type of {name} {i} argument, " f"expected: {func.args[i]}, got: {statement.ret_type}"
        )
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
        arg_code.append(Term(Opcode.LOAD, arg.val))
    elif isinstance(arg, InvokeStatement):
        arg_code.extend(translate_invoke_statement(arg))
    else:
        raise NotImplementedError(f"unknown type of InvokeStatement argument, got {arg}")
    return arg_code


def translate_read_statement(read: InvokeStatement) -> list[Term]:
    var = global_context.require_anon_variable(128)
    return [Term(Opcode.LOAD, var), Term(Opcode.CALL, read.name)]


def translate_invoke_statement(statement: InvokeStatement) -> list[Term]:
    global_context.require_func(statement.name)
    if statement.name == "read":
        return translate_read_statement(statement)
    args = statement.args
    code = []
    for arg in args[-1:0:-1]:
        code.extend(translate_invoke_statement_argument(arg))
        code.append(Term(Opcode.PUSH))
    if len(args) > 0:
        code.extend(translate_invoke_statement_argument(args[0]))
    code.append(Term(Opcode.CALL, statement.name))
    return code


def translate_statement(statement: Statement) -> list[Term]:
    if isinstance(statement, InvokeStatement):
        return translate_invoke_statement(statement)
    raise NotImplementedError(f"unknown type of Statement to translate, got {statement}")


class Code:
    def __init__(self, context: GlobalContext, start_code: list[Term]):
        self.data_memory: list[tuple[int, int, str, list]] = []
        self.data_pointer = 0

        self.instr_memory: list[tuple[int, int, str, list[Term]]] = []
        self.instr_pointer = 0

        self.symbols: set[str] = set()
        self.const_table: dict[str | int, int] = {}
        self.func_table: dict[str, int] = {}

        self.init_global_context(context)
        self.init_start_code(start_code)
        self.init_symbols()

    def init_global_context(self, context: GlobalContext):
        for symbol in context.const_table:
            data: list
            if isinstance(symbol, int):
                data = [symbol]
            elif isinstance(symbol, str):
                data = [*symbol, "\0"]
            else:
                raise NotImplementedError(f"unknown symbol type, got {symbol}")
            self.const_table[symbol] = self.data_pointer
            self.symbols.add(str(symbol))
            self.data_memory.append((self.data_pointer, len(data), str(symbol), data))
            self.data_pointer += len(data)

        for symbol in context.anon_var_table:
            self.const_table[symbol] = self.data_pointer
            self.symbols.add(symbol)

        self.instr_memory.append((0, 1, "#", [Term(Opcode.BRANCH_ANY, "start")]))
        self.func_table["#"] = 0
        self.instr_pointer += 1
        for func_info in context.function_table.values():
            func_code = func_info.code
            self.func_table[func_info.name] = self.instr_pointer
            self.symbols.add(func_info.name)
            self.instr_memory.append((self.instr_pointer, len(func_code), func_info.name, func_code))
            self.instr_pointer += len(func_code)

    def init_start_code(self, start_code: list[Term]):
        self.func_table["start"] = self.instr_pointer
        self.symbols.add("start")
        self.instr_memory.append((self.instr_pointer, len(start_code), "start", start_code))
        self.instr_pointer += len(start_code)

    def init_symbols(self):
        for block in self.instr_memory:
            for instr in block[3]:
                if isinstance(instr.arg, str):
                    assert instr.arg in self.symbols, f"unknown symbol, got {instr.arg}"
                    if instr.arg in self.const_table:
                        instr.arg = Address(AddressType.EXACT, self.const_table[instr.arg])
                    else:
                        instr.arg = Address(AddressType.ABSOLUTE, self.func_table[instr.arg])
                elif isinstance(instr.arg, int):
                    assert str(instr.arg) in self.symbols, f"unknown symbol, got {instr.arg}"
                    instr.arg = Address(AddressType.ABSOLUTE, self.const_table[instr.arg])

    def __len__(self):
        return sum(map(lambda blk: blk[1], self.instr_memory))


def translate_into_code(statements: list[Statement]) -> Code:
    start_code = []
    for s in statements:
        start_code.extend(translate_statement(s))
    start_code.append(Term(Opcode.HALT))
    return Code(global_context, start_code)


def translate(src: str) -> Code:
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


def int_to_bytes(i: int, size: int) -> bytearray:
    assert i >= 0, f"must be non negative, got {i}"
    assert i < (1 << (8 * size)), f"value would lose precision, got {i} with size {size}"
    binary = []
    while size > 0:
        binary.append(i & 0xFF)
        i >>= 8
        size -= 1
    binary = binary[::-1]
    return bytearray(binary)


def term_to_binary(term: Term) -> bytearray:
    binary = term.op.value[1] << 24
    if term.arg:
        assert isinstance(term.arg, Address), f"found unresolved symbol in term, got {term}"
        binary += term.arg.kind.value[1] << 20
        val_unsigned = term.arg.val if term.arg.val >= 0 else (1 << 20) + term.arg.val
        binary += val_unsigned << 0
    return int_to_bytes(binary, 4)


def terms_to_binary(terms: list[Term]) -> bytearray:
    binary = bytearray()
    for term in terms:
        binary.extend(term_to_binary(term))
    return binary


def write_data_memory(code: Code) -> bytearray:
    binary = bytearray()
    for block in code.data_memory:
        for word in block[3]:
            if isinstance(word, int):
                binary.extend(int_to_bytes(word, 8))
            elif isinstance(word, str):
                assert len(word) == 1, f"unexpected char size, got {word}"
                binary.extend(int_to_bytes(ord(word), 8))
            else:
                raise NotImplementedError(f"unknown data type, got {word}")
    return binary


def write_instr_memory(code: Code) -> bytearray:
    binary = bytearray()
    for block in code.instr_memory:
        binary.extend(terms_to_binary(block[3]))
    return binary


def code_to_binary(code: Code) -> bytearray:
    magic = 0xC0DE
    base_offset = 2 + 2 + 2
    data_binary = write_data_memory(code)
    data_length = len(data_binary)
    data_offset = base_offset
    assert data_length < (1 << 16), f"data sector limit exceeded, got {data_length}"
    instr_binary = write_instr_memory(code)
    instr_length = len(instr_binary)
    instr_offset = base_offset + 2 + data_length
    assert instr_length < (1 << 16), f"instruction sector limit exceeded, got {instr_length}"
    assert instr_offset < (1 << 16), f"instruction offset limit exceeded, got {instr_offset}"

    binary = bytearray()
    binary.extend(int_to_bytes(data_offset, 2))
    binary.extend(int_to_bytes(instr_offset, 2))
    binary.extend(int_to_bytes(magic, 2))
    binary.extend(int_to_bytes(data_length, 2))
    binary.extend(data_binary)
    binary.extend(int_to_bytes(instr_length, 2))
    binary.extend(instr_binary)
    return binary


def text_data_memory(code: Code) -> list[str]:
    lines = ["<address> - <length> - <data>\n"]
    for block in code.data_memory:
        lines.append(f"{block[0]}\t- {block[1]}\t- {block[2]}\n")
    return lines


def text_instr_memory(code: Code) -> list[str]:
    lines = ["<address> - <hexcode> - <mnemonica>\n"]
    for block in code.instr_memory:
        lines.append(f"{block[2]}:\n")
        base_addr = block[0]
        for i, term in enumerate(block[3]):
            lines.append(f"{base_addr + i}\t- 0x{term_to_binary(term).hex()} - {term}\n")
    return lines


def code_to_text(code: Code) -> list[str]:
    lines = ["##### Data section #####\n"]
    lines.extend(text_data_memory(code))
    lines.append("\n##### Instruction section #####\n")
    lines.extend(text_instr_memory(code))
    return lines


def write_code(dst, code: Code):
    logging.debug("Serialize code into bytes")
    binary = code_to_binary(code)
    logging.debug(f"Serialized into {len(binary)} bytes")

    logging.debug(f"Start writing into {dst}")
    with open(dst, "wb") as f:
        f.write(binary)
    logging.debug("Successfully wrote binary data")

    logging.debug("Serialize code with debugging info")
    text = code_to_text(code)
    logging.debug(f"Serialized into {len(text)} lines")

    logging.debug(f"Start writing into {dst}.debug")
    with open(dst + ".debug", "w") as f:
        f.writelines(text)
    logging.debug("Successfully wrote debug data")


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
    assert len(sys.argv) == 3, "Usage: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
