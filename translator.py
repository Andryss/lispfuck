from __future__ import annotations

import argparse
import collections
import copy
import logging
import typing
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


class FuncInfo:
    def __init__(self, name: str, argc: int = 0, code: list[Term] | None = None):
        self.name = name
        self.argc = argc
        self.code = [] if code is None else code


predefined_funcs: dict[str, FuncInfo] = {
    "prints": FuncInfo(
        "prints",
        1,
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
    "printi": FuncInfo(
        "printi",
        1,
        [
            Term(Opcode.PUSH),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 2)),
            Term(Opcode.ADD, Address(AddressType.EXACT, 20)),
            Term(Opcode.STORE, Address(AddressType.RELATIVE_SPR, 2)),
            Term(Opcode.LOAD, Address(AddressType.EXACT, 0)),
            Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 2)),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
            Term(Opcode.BRANCH_GREATER_EQUALS, Address(AddressType.RELATIVE_IPR, 5)),
            Term(Opcode.LOAD, Address(AddressType.EXACT, ord("-"))),
            Term(Opcode.STORE, Address(AddressType.ABSOLUTE, 5556)),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.INVERSE),
            Term(Opcode.PUSH),
            Term(Opcode.MODULO, Address(AddressType.EXACT, 10)),
            Term(Opcode.ADD, Address(AddressType.EXACT, ord("0"))),
            Term(Opcode.DECREMENT, Address(AddressType.RELATIVE_SPR, 3)),
            Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 3)),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.DIVIDE, Address(AddressType.EXACT, 10)),
            Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
            Term(Opcode.BRANCH_ZERO, Address(AddressType.RELATIVE_IPR, 3)),
            Term(Opcode.STORE, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -9)),
            Term(Opcode.POP),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 2)),
            Term(Opcode.CALL, "prints"),
            Term(Opcode.PUSH),
            Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 1)),
            Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
            Term(Opcode.BRANCH_GREATER_EQUALS, Address(AddressType.RELATIVE_IPR, 2)),
            Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.POP),
            Term(Opcode.POPN),
            Term(Opcode.RETURN),
        ],
    ),
    "read": FuncInfo(
        "read",
        0,
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
            Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -9)),
            Term(Opcode.LOAD, Address(AddressType.EXACT, ord("\0"))),
            Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 1)),
            Term(Opcode.POP),
            Term(Opcode.POP),
            Term(Opcode.POP),
            Term(Opcode.RETURN),
        ],
    ),
}


class FuncContext:
    def __init__(self):
        self.args_table: dict[str, int] = {}

    def has_in_acr(self) -> bool:
        return -1 in self.args_table.values()

    def get_in_acr(self) -> str:
        return list(self.args_table.keys())[list(self.args_table.values()).index(-1)]

    def on_push(self):
        self.args_table = {k: v + 1 for k, v in self.args_table.items()}

    def on_pop(self):
        self.args_table = {k: v - 1 for k, v in self.args_table.items()}


class ProgramContext:
    def __init__(self):
        self.defined_funcs: dict[str, FuncInfo] = copy.deepcopy(predefined_funcs)
        self.function_table: list[str] = []

        self.str_const_table: list[str] = []
        self.int_const_table: list[int] = []
        self.var_table: list[str] = []
        self.anon_var_table: dict[str, tuple[int, int]] = collections.OrderedDict()
        self.anon_var_pointer = 0
        self.anon_var_counter = 0

        self.func_context: FuncContext | None = None

    def require_func(self, func: str):
        assert func in self.defined_funcs, f"Unknown func {func}"
        if func not in self.function_table:
            self.function_table.append(func)

    def define_func(self, name: str, argc: int):
        assert name not in self.defined_funcs, f"Function {name} already defined"
        assert argc >= 0, f"Function must have positive or zero arguments, got {argc}"
        self.defined_funcs[name] = FuncInfo(name, argc)

    def implement_func(self, name: str, code: list[Term]):
        assert name in self.defined_funcs, f"Unknown function to implement, got {name}"
        func_info = self.defined_funcs[name]
        assert len(func_info.code) == 0, f"Reimplementation of function {name}"
        assert len(code) > 0, f"Implementation must have at least 1 statement, got {code}"
        func_info.code = code

    def func_info(self, func: str) -> FuncInfo:
        assert func in self.defined_funcs, f"Unknown func {func}"
        return self.defined_funcs[func]

    def require_int_const(self, const: int):
        assert const < (1 << 63), f"Value must be less than {1 << 63}, got {const}"
        assert const > (-(1 << 63) - 1), f"Value must be greater than {- (1 << 63) - 1}, got {const}"
        if const not in self.int_const_table:
            self.int_const_table.append(const)

    def require_str_const(self, const: str):
        assert const[0] != '"', f"Value must be trimmed, got {const}"
        assert const[-1] != '"', f"Value must be trimmed, got {const}"
        if const not in self.str_const_table:
            self.str_const_table.append(const)

    def require_anon_variable(self, size: int) -> str:
        assert size > 0, f"negative size buffer?, got {size}"
        name = f"anon${self.anon_var_counter}"
        self.anon_var_counter += 1
        self.anon_var_table[name] = (self.anon_var_pointer, size)
        self.anon_var_pointer += size
        return name

    def require_variable(self, name: str):
        if name not in self.var_table:
            self.var_table.append(name)

    def set_func_context(self, fc: FuncContext | None):
        self.func_context = fc

    def get_func_context(self) -> FuncContext | None:
        return self.func_context

    def func_context_has_in_acr(self) -> bool:
        return self.func_context and self.func_context.has_in_acr()

    def func_context_on_push(self):
        if self.func_context:
            self.func_context.on_push()

    def func_context_on_pop(self):
        if self.func_context:
            self.func_context.on_pop()


class Tag(Enum):
    INVOKE = "INVOKE"
    INT_CONST = "INT_CONST"
    STR_CONST = "STR_CONST"
    VALUE = "VALUE"
    REFERENCE = "REFERENCE"


class Statement:
    def __init__(self, tag: Tag, name: str | None = None, args: list[Statement] | None = None, val: int | None = None):
        self.tag = tag
        self.name = name
        self.args = [] if args is None else args
        self.val = val


def const_statement(node: ASTNode, context: ProgramContext) -> Statement:
    token = node.token
    assert token.tag in (lexer.INT, lexer.STR), f"Unknown const type {token.tag}"
    if token.tag == lexer.INT:
        int_val = int(node.token.string)
        if (1 << 19) > int_val > (-(1 << 19) - 1):
            return Statement(Tag.VALUE, val=int_val)
        context.require_int_const(int_val)
        return Statement(Tag.INT_CONST, val=int_val)
    if token.tag == lexer.STR:
        str_val = node.token.string
        context.require_str_const(str_val)
        return Statement(Tag.STR_CONST, name=str_val)
    raise NotImplementedError(f"unknown lex tag of constant statement, got {token.tag}")


def invoke_statement(node: ASTNode, context: ProgramContext) -> Statement:
    children = node.children
    assert len(children[0].children) == 0, "Name of statement must be string"
    name = children[0].token.string
    func = context.func_info(name)
    children_nodes = children[1:]
    assert func.argc == len(children_nodes), f"Wrong amount of arguments for func {name}"
    children_statements = []
    for i in range(len(children_nodes)):
        children_statements.append(ast_to_statement(children_nodes[i], context))
    return Statement(Tag.INVOKE, name=name, args=children_statements)


def special_statement(node: ASTNode, context: ProgramContext) -> Statement:
    children = node.children
    name = children[0].token.string
    if name == "set":
        assert len(children) == 3, "Set statement must contains of variable name and value to set"
        key, val = children[1], children[2]
        assert key.token.tag == lexer.ID, f"variable name must be ID, got {key.token}"
        args = [Statement(Tag.REFERENCE, name=key.token.string), ast_to_statement(val, context)]
        context.require_variable(key.token.string)
        return Statement(Tag.INVOKE, name=name, args=args)
    if name == "if":
        assert len(children) == 4, "if statement must contains condition and 2 options"
        cond, opt1, opt2 = children[1], children[2], children[3]
        args = [ast_to_statement(cond, context), ast_to_statement(opt1, context), ast_to_statement(opt2, context)]
        return Statement(Tag.INVOKE, name=name, args=args)
    raise NotImplementedError(f"unknown special statement, got {name}")


def reference_statement(node: ASTNode) -> Statement:
    symbol = node.token.string
    return Statement(Tag.REFERENCE, name=symbol)


def math_statement(node: ASTNode, context: ProgramContext) -> Statement:
    children = node.children
    name = children[0].token.string
    assert name in ("mod", "+", "-", "/", "*"), f"unknown math statement, got {name}"
    if name in ("mod", "-", "/"):
        assert len(children) == 3, f"math statement can operate only with 2 args, got {len(children)}"
    args = []
    for child in children[1:]:
        args.append(ast_to_statement(child, context))
    return Statement(Tag.INVOKE, name=name, args=args)


def bool_statement(node: ASTNode, context: ProgramContext):
    children = node.children
    name = children[0].token.string
    assert name in ("=", ">="), f"unknown boolean statement, got {name}"
    assert len(children) == 3, "= statement must contains of 2 values to compare"
    args = [ast_to_statement(children[1], context), ast_to_statement(children[2], context)]
    return Statement(Tag.INVOKE, name=name, args=args)


def defun_statement(node: ASTNode, context: ProgramContext) -> Statement:
    children = node.children
    name = children[0].token.string
    assert len(children) == 4, "defunc statement must contains of name, arguments and body"

    func_name = children[1].token
    assert func_name.tag == lexer.ID, f"func name must be ID, got {func_name.tag}"
    func_name = func_name.string

    args = children[2].children
    assert children[2].token.string == "(", f"arguments must be in parenthesis, got {children[2].token}"

    name_st = Statement(Tag.REFERENCE, name=func_name)
    args_st = Statement(Tag.INVOKE)
    for arg in args:
        assert arg.token.tag == lexer.ID, f"args must be ID, got {arg}"
        arg_name = arg.token.string
        args_st.args.append(Statement(Tag.REFERENCE, name=arg_name))

    context.define_func(func_name, len(args_st.args))
    body_st = ast_to_statement(children[3], context)
    return Statement(Tag.INVOKE, name=name, args=[name_st, args_st, body_st])


def ast_to_statement(node: ASTNode, context: ProgramContext) -> Statement:
    children = node.children
    if len(children) > 0:
        tag = children[0].token.tag
        if tag == lexer.FUNC:
            return invoke_statement(node, context)
        if tag == lexer.SPECIAL:
            return special_statement(node, context)
        if tag == lexer.MATH:
            return math_statement(node, context)
        if tag == lexer.BOOL:
            return bool_statement(node, context)
        if tag == lexer.DEFUNC:
            return defun_statement(node, context)
        if tag == lexer.ID:
            return invoke_statement(node, context)
        raise NotImplementedError(f"unknown node with children token tag, got {tag}")
    tag = node.token.tag
    if tag in (lexer.STR, lexer.INT):
        return const_statement(node, context)
    if tag == lexer.ID:
        return reference_statement(node)
    raise NotImplementedError(f"unknown node without children token tag, got {tag}")


def extract_statements(root: ASTNode, context: ProgramContext) -> list[Statement]:
    statements = []
    for node in root.children:
        statements.append(ast_to_statement(node, context))
    return statements


def translate_invoke_statement_argument(arg: Statement, context: ProgramContext) -> list[Term]:
    arg_code = []
    fc = context.get_func_context()
    if arg.tag == Tag.VALUE:
        if fc and fc.has_in_acr():
            arg_code.append(Term(Opcode.PUSH))
            fc.on_push()
        arg_code.append(Term(Opcode.LOAD, Address(AddressType.EXACT, arg.val)))
    elif arg.tag in (Tag.INT_CONST, Tag.STR_CONST):
        if fc and fc.has_in_acr():
            arg_code.append(Term(Opcode.PUSH))
            fc.on_push()
        arg = arg.val if arg.tag == Tag.INT_CONST else arg.name
        arg_code.append(Term(Opcode.LOAD, arg))
    elif arg.tag == Tag.INVOKE:
        arg_code.extend(translate_invoke_statement(arg, context))
    elif arg.tag == Tag.REFERENCE:
        if fc and fc.has_in_acr() and fc.get_in_acr() == arg.name:
            pass
        elif fc and arg.name in fc.args_table:
            if fc.has_in_acr():
                arg_code.append(Term(Opcode.PUSH))
                fc.on_push()
            arg_code.append(Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, fc.args_table[arg.name]), arg.name))
        else:
            arg_code.append(Term(Opcode.LOAD, arg.name))
    else:
        raise NotImplementedError(f"unknown tag of invoke statement argument, got {arg.tag}")
    return arg_code


def translate_read_statement(read: Statement, context: ProgramContext) -> list[Term]:
    code = []
    if context.func_context_has_in_acr():
        code.append(Term(Opcode.PUSH))
        context.func_context_on_push()
    read.anon_var_name = context.require_anon_variable(128)
    code.extend([Term(Opcode.LOAD, read.anon_var_name), Term(Opcode.CALL, read.name)])
    return code


def translate_printi_statement(printi: Statement, context: ProgramContext) -> list[Term]:
    code = []
    if context.func_context_has_in_acr():
        code.append(Term(Opcode.PUSH))
        context.func_context_on_push()
    context.require_func("prints")
    printi.anon_var_name = context.require_anon_variable(21)
    code.extend([Term(Opcode.LOAD, printi.anon_var_name), Term(Opcode.PUSH)])
    context.func_context_on_push()
    code.extend(translate_invoke_statement_argument(printi.args[0], context))
    code.extend([Term(Opcode.CALL, printi.name), Term(Opcode.POPN)])
    context.func_context_on_pop()
    return code


def translate_set_statement(set_st: Statement, context: ProgramContext) -> list[Term]:
    variable, value = set_st.args[0], set_st.args[1]
    assert variable.tag == Tag.REFERENCE, f"unexpected variable statement type, got {variable}"
    context.require_variable(variable.name)
    code = translate_invoke_statement_argument(value, context)
    code.append(Term(Opcode.STORE, variable.name))
    return code


math_opcode = {"mod": Opcode.MODULO, "+": Opcode.ADD, "-": Opcode.SUBTRACT, "*": Opcode.MULTIPLY, "/": Opcode.DIVIDE}


def translate_math_statement(statement: Statement, context: ProgramContext) -> list[Term]:
    code = []
    opcode = math_opcode[statement.name]
    last_arg = statement.args[-1]
    code.extend(translate_invoke_statement_argument(last_arg, context))
    code.append(Term(Opcode.PUSH))
    context.func_context_on_push()
    for arg in statement.args[-2::-1]:
        code.extend(translate_invoke_statement_argument(arg, context))
        code.append(Term(opcode, Address(AddressType.RELATIVE_SPR, 0)))
        code.append(Term(Opcode.STORE, Address(AddressType.RELATIVE_SPR, 0)))
    code.append(Term(Opcode.POP))
    context.func_context_on_pop()
    return code


bool_opcode = {"=": Opcode.BRANCH_ZERO, ">=": Opcode.BRANCH_GREATER_EQUALS}


def translate_bool_statement(statement: Statement, context: ProgramContext) -> list[Term]:
    code = []
    opcode = bool_opcode[statement.name]
    code.extend(translate_invoke_statement_argument(statement.args[1], context))
    code.append(Term(Opcode.PUSH))
    context.func_context_on_push()
    code.extend(translate_invoke_statement_argument(statement.args[0], context))
    code.extend(
        [
            Term(Opcode.COMPARE, Address(AddressType.RELATIVE_SPR, 0)),
            Term(Opcode.POPN),
            Term(opcode, Address(AddressType.RELATIVE_IPR, 3)),
            Term(Opcode.LOAD, Address(AddressType.EXACT, 0)),
            Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, 2)),
            Term(Opcode.LOAD, Address(AddressType.EXACT, 1)),
        ]
    )
    context.func_context_on_pop()
    return code


def translate_if_statement(statement: Statement, context: ProgramContext) -> list[Term]:
    code = []
    cond_code = translate_invoke_statement_argument(statement.args[0], context)
    opt1_code = translate_invoke_statement_argument(statement.args[1], context)
    opt2_code = translate_invoke_statement_argument(statement.args[2], context)
    opt1_code.append(Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, len(opt2_code) + 1)))
    cond_code.extend(
        [
            Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
            Term(Opcode.BRANCH_ZERO, Address(AddressType.RELATIVE_IPR, len(opt1_code) + 1)),
        ]
    )
    code.extend(cond_code)
    code.extend(opt1_code)
    code.extend(opt2_code)
    return code


def translate_defun_statement(defun: Statement, context: ProgramContext) -> list[Term]:
    func_code = []
    func_name = defun.args[0].name
    arguments: list[Statement] = defun.args[1].args
    func_context = FuncContext()
    if len(arguments) > 0:
        first_argument = arguments[0]
        func_context.args_table[first_argument.name] = -1
    for i, argument in enumerate(arguments[1:]):
        func_context.args_table[argument.name] = 1 + i
    context.set_func_context(func_context)
    func_code.extend(translate_invoke_statement_argument(defun.args[2], context))
    context.set_func_context(None)
    if 0 in func_context.args_table.values():
        func_code.append(Term(Opcode.POPN))
    func_code.append(Term(Opcode.RETURN))
    context.implement_func(func_name, func_code)
    return []


def translate_invoke_statement_common(statement: Statement, context: ProgramContext) -> list[Term]:
    args = statement.args
    code = []
    for arg in args[-1:0:-1]:
        code.extend(translate_invoke_statement_argument(arg, context))
        code.append(Term(Opcode.PUSH))
        context.func_context_on_push()
    if len(args) > 0:
        code.extend(translate_invoke_statement_argument(args[0], context))
    code.append(Term(Opcode.CALL, statement.name))
    for _ in range(len(args) - 1):
        code.append(Term(Opcode.POPN))
        context.func_context_on_pop()
    return code


def translate_invoke_statement(statement: Statement, context: ProgramContext) -> list[Term]:
    if statement.name == "set":
        return translate_set_statement(statement, context)
    if statement.name in ("mod", "+", "-", "/", "*"):
        return translate_math_statement(statement, context)
    if statement.name in ("=", ">="):
        return translate_bool_statement(statement, context)
    if statement.name == "if":
        return translate_if_statement(statement, context)
    if statement.name == "defun":
        return translate_defun_statement(statement, context)
    context.require_func(statement.name)
    if statement.name == "read":
        return translate_read_statement(statement, context)
    if statement.name == "printi":
        return translate_printi_statement(statement, context)
    return translate_invoke_statement_common(statement, context)


def translate_statement(statement: Statement, context: ProgramContext) -> list[Term]:
    if statement.tag == Tag.INVOKE:
        return translate_invoke_statement(statement, context)
    raise NotImplementedError(f"unknown tag of statement to translate, got {statement.tag}")


class Code:
    def __init__(self, context: ProgramContext, start_code: list[Term]):
        self.data_memory: list[tuple[int, int, str, list]] = []
        self.data_pointer = 0

        self.instr_memory: list[tuple[int, int, str, list[Term]]] = []
        self.instr_pointer = 0

        self.symbols: set[str] = set()
        self.const_table: dict[str | int, int] = {}
        self.var_table: dict[str, int] = {}
        self.func_table: dict[str, int] = {}

        self.init_global_context(context)
        self.init_start_code(start_code)
        self.init_symbols()

    def init_global_context(self, context: ProgramContext):
        for str_const in context.str_const_table:
            data = [*str_const, "\0"]
            self.const_table[str_const] = self.data_pointer
            self.symbols.add(str_const)
            self.data_memory.append((self.data_pointer, len(data), repr(str_const), data))
            self.data_pointer += len(data)

        for int_const in context.int_const_table:
            self.const_table[int_const] = self.data_pointer
            self.symbols.add(str(int_const))
            self.data_memory.append((self.data_pointer, 1, str(int_const), [int_const]))
            self.data_pointer += 1

        for symbol in context.var_table:
            self.var_table[symbol] = self.data_pointer
            self.symbols.add(symbol)
            self.data_pointer += 1

        for symbol in context.anon_var_table:
            offset, _ = context.anon_var_table[symbol]
            self.const_table[symbol] = self.data_pointer + offset
            self.symbols.add(symbol)

        self.instr_memory.append((0, 1, "#", [Term(Opcode.BRANCH_ANY, "start")]))
        self.func_table["#"] = 0
        self.instr_pointer += 1
        for func_name in context.function_table:
            func_info = context.defined_funcs[func_name]
            self.func_table[func_info.name] = self.instr_pointer
            self.symbols.add(func_info.name)
            self.instr_memory.append((self.instr_pointer, len(func_info.code), func_info.name, func_info.code))
            self.instr_pointer += len(func_info.code)

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
                    instr.desc = instr.arg
                    if instr.arg in self.const_table:
                        instr.arg = Address(AddressType.EXACT, self.const_table[instr.arg])
                    elif instr.arg in self.var_table:
                        instr.arg = Address(AddressType.ABSOLUTE, self.var_table[instr.arg])
                    else:
                        instr.arg = Address(AddressType.ABSOLUTE, self.func_table[instr.arg])
                elif isinstance(instr.arg, int):
                    assert str(instr.arg) in self.symbols, f"unknown symbol, got {instr.arg}"
                    instr.desc = str(instr.arg)
                    instr.arg = Address(AddressType.ABSOLUTE, self.const_table[instr.arg])

    def __len__(self):
        return sum(map(lambda blk: blk[1], self.instr_memory))


def translate_into_code(statements: list[Statement], context: ProgramContext) -> Code:
    start_code = []
    for s in statements:
        start_code.extend(translate_statement(s, context))
    start_code.append(Term(Opcode.HALT))
    return Code(context, start_code)


def translate(src: str) -> Code:
    tokens = extract_tokens(src)

    ast = build_ast(tokens)

    program_context = ProgramContext()

    statements = extract_statements(ast, program_context)

    return translate_into_code(statements, program_context)


def read_source(src: typing.TextIO) -> str:
    return src.read()


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


mask_64 = (1 << 64) - 1


def write_data_memory(code: Code) -> bytearray:
    binary = bytearray()
    for block in code.data_memory:
        for word in block[3]:
            if isinstance(word, int):
                binary.extend(int_to_bytes(word if word >= 0 else word & mask_64, 8))
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
    lines = ["<address>\t<length>\t<data>"]
    for block in code.data_memory:
        addr = "0x{:03x}".format(block[0])
        lines.append(f"{addr}\t{block[1]}\t{block[2]}")
    return lines


def text_instr_memory(code: Code) -> list[str]:
    lines = ["<address>\t<hexcode>\t<mnemonica>"]
    for block in code.instr_memory:
        lines.append(f"{block[2]}:")
        base_addr = block[0]
        for i, term in enumerate(block[3]):
            addr = "0x{:08x}".format(base_addr + i)
            lines.append(f"{addr}\t0x{term_to_binary(term).hex()}\t{term}")
    return lines


def code_to_text(code: Code) -> str:
    lines = ["##### Data section #####"]
    lines.extend(text_data_memory(code))
    lines.append("\n##### Instruction section #####")
    lines.extend(text_instr_memory(code))
    return "\n".join(lines).expandtabs(15)


def write_code(dst: typing.BinaryIO, dbg: typing.TextIO, code: Code) -> (bytearray, str):
    binary = code_to_binary(code)
    dst.write(binary)
    text = code_to_text(code)
    dbg.write(text)
    return binary, text


def main(src: typing.TextIO, dst: typing.BinaryIO, dbg: typing.TextIO, verbose: bool = False):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source = read_source(src)
    code = translate(source)
    binary, debug = write_code(dst, dbg, code)

    loc, code_byte, code_instr, debug_lines = len(source.split("\n")), len(binary), len(code), len(debug.split("\n"))
    logging.info(f"LoC: {loc} code byte: {code_byte} code instr: {code_instr} debug lines: {debug_lines}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translates lispfuck code into en executable file.")
    parser.add_argument(
        "src", type=argparse.FileType(encoding="utf-8"), metavar="source_file", help="file with lispfuck code"
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="dst",
        type=argparse.FileType("wb"),
        default="output",
        metavar="output_file",
        help="file for storing a binary executable (default: output)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        dest="dbg",
        type=argparse.FileType("w", encoding="utf-8"),
        default="debug.txt",
        metavar="debug_file",
        help="file for storing a binary executable explanation info (default: debug.txt)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        help="print verbose information during conversion",
    )
    namespace = parser.parse_args()
    main(namespace.src, namespace.dst, namespace.dbg, namespace.verbose)
