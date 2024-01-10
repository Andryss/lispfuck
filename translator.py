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


predefined_funcs: dict[str, FuncInfo]


class FuncVars:
    def __init__(self):
        self.args: dict[str, Type] = {}

    def add_var(self, name: str, t: Type):
        self.args[name] = t

    def has_var(self, name: str) -> bool:
        return name in self.args

    def var_type(self, name: str) -> Type:
        return self.args[name]


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


class GlobalContext:
    def __init__(self):
        self.function_table: dict[str, FuncInfo] = {}
        self.const_table: set[str | int] = set()
        self.var_table: dict[str, Type] = {}
        self.anon_var_table: dict[str, tuple[int, int]] = {}
        self.anon_var_pointer = 0
        self.anon_var_counter = 0

        self.func_vars: FuncVars | None = None
        self.func_context: FuncContext | None = None

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

    def require_anon_variable(self, size: int, offset: int = 0) -> str:
        assert size > 0, f"negative size buffer?, got {size}"
        name = f"anon${self.anon_var_counter}"
        self.anon_var_counter += 1
        self.anon_var_table[name] = (self.anon_var_pointer + offset, size)
        self.anon_var_pointer += size
        return name

    def require_variable(self, name: str, t: Type):
        assert t != Type.UNK_TYPE, f"unexpected variable type, got {t}"
        self.var_table[name] = t

    def variable_type(self, name: str) -> Type:
        if self.func_vars and self.func_vars.has_var(name):
            return self.func_vars.var_type(name)
        if name in self.var_table:
            return self.var_table[name]
        return Type.UNK_TYPE

    def set_func_vars(self, fv: FuncVars | None):
        self.func_vars = fv

    def set_func_context(self, fc: FuncContext | None):
        self.func_context = fc

    def get_func_context(self) -> FuncContext | None:
        return self.func_context


global_context: GlobalContext


def init_base_funcs():
    global predefined_funcs, global_context
    global_context = GlobalContext()
    predefined_funcs = {
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
        "printi": FuncInfo(
            "printi",
            [Type.INT_TYPE],
            Type.INT_TYPE,
            [
                Term(Opcode.PUSH),
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


class ReferenceStatement(Statement):
    def __init__(self, ret_type: Type = Type.UNK_TYPE, symbol: str | None = None):
        super().__init__(ret_type)
        self.symbol = symbol


def const_statement(node: ASTNode) -> Statement:
    token = node.token
    assert token.tag in (lexer.INT, lexer.STR), f"Unknown const type {token.tag}"
    if token.tag == lexer.INT:
        const_type = Type.INT_TYPE
        int_val = int(node.token.string)
        if (1 << 19) > int_val > (-(1 << 19) - 1):
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


def special_statement(node: ASTNode) -> Statement:
    children = node.children
    name = children[0].token.string
    if name == "set":
        assert len(children) == 3, "Set statement must contains of variable name and value to set"
        key, val = children[1], children[2]
        assert key.token.tag == lexer.ID, f"variable name must be ID, got {key.token}"
        args = [ReferenceStatement(symbol=key.token.string), ast_to_statement(val)]
        global_context.require_variable(key.token.string, args[1].ret_type)
        return InvokeStatement(args[1].ret_type, name, args)
    if name == "if":
        assert len(children) == 4, "if statement must contains condition and 2 options"
        cond, opt1, opt2 = children[1], children[2], children[3]
        cond_arg = ast_to_statement(cond)
        assert cond_arg.ret_type == Type.INT_TYPE, f"condition must return int, got {cond_arg.ret_type}"
        args = [cond_arg, ast_to_statement(opt1), ast_to_statement(opt2)]
        assert args[1].ret_type == args[2].ret_type, (
            f"options must have same ret type, " f"got {args[1].ret_type} and {args[2].ret_type}"
        )
        return InvokeStatement(args[1].ret_type, name, args)
    raise NotImplementedError(f"unknown special statement, got {name}")


def reference_statement(node: ASTNode) -> Statement:
    symbol = node.token.string
    return ReferenceStatement(global_context.variable_type(symbol), symbol)


def math_statement(node: ASTNode) -> Statement:
    children = node.children
    name = children[0].token.string
    assert name in ("mod", "+", "-", "/", "*"), f"unknown math statement, got {name}"
    if name in ("mod", "-", "/"):
        assert len(children) == 3, f"math statement can operate only with 2 args, got {len(children)}"
    args = []
    for child in children[1:]:
        statement = ast_to_statement(child)
        assert statement.ret_type == Type.INT_TYPE, f"math statements can operate with INT, got {statement.ret_type}"
        args.append(statement)
    return InvokeStatement(Type.INT_TYPE, name, args)


def bool_statement(node: ASTNode):
    children = node.children
    name = children[0].token.string
    if name == "=":
        assert len(children) == 3, "= statement must contains of 2 values to compare"
        args = [ast_to_statement(children[1]), ast_to_statement(children[2])]
        assert args[0].ret_type == args[1].ret_type == Type.INT_TYPE, "= can compare only ints"
        return InvokeStatement(Type.INT_TYPE, name, args)
    raise NotImplementedError(f"unknown boolean statement, got {name}")


def parse_arg(arg: str) -> (str, Type):
    idx = arg.find(":")
    name, ret_type = arg[:idx], arg[idx + 1 :]
    assert ret_type in ("i", "s"), f"ret type must be int or str, got {ret_type}"
    ret_type = Type.INT_TYPE if ret_type == "i" else Type.STR_TYPE
    return name, ret_type


def defun_statement(node: ASTNode) -> Statement:
    children = node.children
    name = children[0].token.string
    assert len(children) == 4, "defunc statement must contains of name, arguments and body"

    func_name = children[1].token
    assert func_name.tag == lexer.ARG, f"func name must be ARG, got {func_name.tag}"
    func_name, func_ret_type = parse_arg(func_name.string)

    args = children[2].children
    assert children[2].token.string == "(", f"arguments must be in parenthesis, got {children[2].token}"

    name_st = ReferenceStatement(func_ret_type, func_name)
    args_st = InvokeStatement()
    func_vars = FuncVars()
    arg_types = []
    for arg in args:
        assert arg.token.tag == lexer.ARG, f"args must be ARG, got {arg.token}"
        arg_name, arg_type = parse_arg(arg.token.string)
        args_st.args.append(ReferenceStatement(arg_type, arg_name))
        func_vars.add_var(arg_name, arg_type)
        arg_types.append(arg_type)

    predefined_funcs[func_name] = FuncInfo(func_name, arg_types, func_ret_type)
    global_context.set_func_vars(func_vars)
    body_st = ast_to_statement(children[3])
    global_context.set_func_vars(None)
    return InvokeStatement(name=name, args=[name_st, args_st, body_st])


def ast_to_statement(node: ASTNode) -> Statement:
    children = node.children
    if len(children) > 0:
        tag = children[0].token.tag
        if tag == lexer.FUNC:
            return invoke_statement(node)
        if tag == lexer.SPECIAL:
            return special_statement(node)
        if tag == lexer.MATH:
            return math_statement(node)
        if tag == lexer.BOOL:
            return bool_statement(node)
        if tag == lexer.DEFUNC:
            return defun_statement(node)
        if tag == lexer.ID:
            return invoke_statement(node)
        raise NotImplementedError(f"ast to statement translation, got {tag}")
    tag = node.token.tag
    if tag in (lexer.STR, lexer.INT):
        return const_statement(node)
    if tag == lexer.ID:
        return reference_statement(node)
    raise NotImplementedError(f"ast to statement translation, got {tag}")


def extract_statements(root: ASTNode) -> list[Statement]:
    init_base_funcs()
    statements = []
    for node in root.children:
        statements.append(ast_to_statement(node))
    return statements


def translate_invoke_statement_argument(arg: Statement) -> list[Term]:
    arg_code = []
    fc = global_context.get_func_context()
    if isinstance(arg, ValueStatement):
        if fc and fc.has_in_acr():
            arg_code.append(Term(Opcode.PUSH))
            fc.on_push()
        arg_code.append(Term(Opcode.LOAD, Address(AddressType.EXACT, arg.val)))
    elif isinstance(arg, ConstStatement):
        if fc and fc.has_in_acr():
            arg_code.append(Term(Opcode.PUSH))
            fc.on_push()
        arg_code.append(Term(Opcode.LOAD, arg.val))
    elif isinstance(arg, InvokeStatement):
        arg_code.extend(translate_invoke_statement(arg))
    elif isinstance(arg, ReferenceStatement):
        if fc and fc.has_in_acr() and fc.get_in_acr() == arg.symbol:
            pass
        elif fc and arg.symbol in fc.args_table:
            if fc.has_in_acr():
                arg_code.append(Term(Opcode.PUSH))
                fc.on_push()
            arg_code.append(Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, fc.args_table[arg.symbol]), arg.symbol))
        else:
            arg_code.append(Term(Opcode.LOAD, arg.symbol))
    else:
        raise NotImplementedError(f"unknown type of InvokeStatement argument, got {arg}")
    return arg_code


def translate_read_statement(read: InvokeStatement) -> list[Term]:
    code = []
    if global_context.get_func_context() and global_context.get_func_context().has_in_acr():
        code.append(Term(Opcode.PUSH))
        global_context.get_func_context().on_push()
    read.anon_var_name = global_context.require_anon_variable(128)
    code.extend([Term(Opcode.LOAD, read.anon_var_name), Term(Opcode.CALL, read.name)])
    return code


def translate_printi_statement(printi: InvokeStatement) -> list[Term]:
    code = []
    if global_context.get_func_context() and global_context.get_func_context().has_in_acr():
        code.append(Term(Opcode.PUSH))
        global_context.get_func_context().on_push()
    global_context.require_func("prints")
    printi.anon_var_name = global_context.require_anon_variable(21, offset=20)
    code.extend([Term(Opcode.LOAD, printi.anon_var_name), Term(Opcode.PUSH)])
    if global_context.get_func_context():
        global_context.get_func_context().on_push()
    code.extend(translate_invoke_statement_argument(printi.args[0]))
    code.extend([Term(Opcode.CALL, printi.name), Term(Opcode.POPN)])
    if global_context.get_func_context():
        global_context.get_func_context().on_pop()
    return code


def translate_set_statement(set_st: InvokeStatement) -> list[Term]:
    variable, value = set_st.args[0], set_st.args[1]
    assert isinstance(variable, ReferenceStatement), f"unexpected variable statement type, got {variable}"
    global_context.require_variable(variable.symbol, value.ret_type)
    code = translate_invoke_statement_argument(value)
    code.append(Term(Opcode.STORE, variable.symbol))
    return code


math_opcode = {"mod": Opcode.MODULO, "+": Opcode.ADD, "-": Opcode.SUBTRACT, "*": Opcode.MULTIPLY, "/": Opcode.DIVIDE}


def translate_math_statement(statement: InvokeStatement) -> list[Term]:
    code = []
    opcode = math_opcode[statement.name]
    last_arg = statement.args[-1]
    code.extend(translate_invoke_statement_argument(last_arg))
    code.append(Term(Opcode.PUSH))
    if global_context.get_func_context():
        global_context.get_func_context().on_push()
    for arg in statement.args[-2::-1]:
        code.extend(translate_invoke_statement_argument(arg))
        code.append(Term(opcode, Address(AddressType.RELATIVE_SPR, 0)))
        code.append(Term(Opcode.STORE, Address(AddressType.RELATIVE_SPR, 0)))
    code.append(Term(Opcode.POP))
    if global_context.get_func_context():
        global_context.get_func_context().on_pop()
    return code


bool_opcode = {"=": Opcode.BRANCH_ZERO, ">=": Opcode.BRANCH_GREATER_EQUALS}


def translate_bool_statement(statement: InvokeStatement) -> list[Term]:
    code = []
    opcode = bool_opcode[statement.name]
    code.extend(translate_invoke_statement_argument(statement.args[1]))
    code.append(Term(Opcode.PUSH))
    if global_context.get_func_context():
        global_context.get_func_context().on_push()
    code.extend(translate_invoke_statement_argument(statement.args[0]))
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
    if global_context.get_func_context():
        global_context.get_func_context().on_pop()
    return code


def translate_if_statement(statement: InvokeStatement) -> list[Term]:
    code = []
    cond_code = translate_invoke_statement_argument(statement.args[0])
    opt1_code = translate_invoke_statement_argument(statement.args[1])
    opt2_code = translate_invoke_statement_argument(statement.args[2])
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


def translate_defun_statement(defun: InvokeStatement) -> list[Term]:
    predefined_code = []
    func_name = defun.args[0].symbol
    arguments = defun.args[1].args
    func_context = FuncContext()
    if len(arguments) > 0:
        first_argument = arguments[0]
        func_context.args_table[first_argument.symbol] = -1
    for i, argument in enumerate(arguments[1:]):
        func_context.args_table[argument.symbol] = 1 + i
    global_context.set_func_context(func_context)
    predefined_code.extend(translate_invoke_statement_argument(defun.args[2]))
    if 0 in func_context.args_table.values():
        predefined_code.append(Term(Opcode.POPN))
    global_context.set_func_context(None)
    predefined_code.append(Term(Opcode.RETURN))
    predefined_funcs[func_name].code = predefined_code
    return []


def translate_invoke_statement_common(statement: InvokeStatement) -> list[Term]:
    args = statement.args
    code = []
    for arg in args[-1:0:-1]:
        code.extend(translate_invoke_statement_argument(arg))
        code.append(Term(Opcode.PUSH))
        if global_context.get_func_context():
            global_context.get_func_context().on_push()
    if len(args) > 0:
        code.extend(translate_invoke_statement_argument(args[0]))
    code.append(Term(Opcode.CALL, statement.name))
    for _ in range(len(args) - 1):
        code.append(Term(Opcode.POPN))
        if global_context.get_func_context():
            global_context.get_func_context().on_pop()
    return code


def translate_invoke_statement(statement: InvokeStatement) -> list[Term]:
    if statement.name == "set":
        return translate_set_statement(statement)
    if statement.name in ("mod", "+", "-", "/", "*"):
        return translate_math_statement(statement)
    if statement.name in ("=", ">="):
        return translate_bool_statement(statement)
    if statement.name == "if":
        return translate_if_statement(statement)
    if statement.name == "defun":
        return translate_defun_statement(statement)
    global_context.require_func(statement.name)
    if statement.name == "read":
        return translate_read_statement(statement)
    if statement.name == "printi":
        return translate_printi_statement(statement)
    return translate_invoke_statement_common(statement)


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
        self.var_table: dict[str, int] = {}
        self.func_table: dict[str, int] = {}

        self.init_global_context(context)
        self.init_start_code(start_code)
        self.init_symbols()

    def init_global_context(self, context: GlobalContext):
        for symbol in sorted(context.const_table, key=str):
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

        for symbol in sorted(context.var_table):
            self.var_table[symbol] = self.data_pointer
            self.symbols.add(symbol)
            self.data_pointer += 1

        for symbol in sorted(context.anon_var_table):
            offset, _ = context.anon_var_table[symbol]
            self.const_table[symbol] = self.data_pointer + offset
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


def translate_into_code(statements: list[Statement]) -> Code:
    start_code = []
    for s in statements:
        start_code.extend(translate_statement(s))
    start_code.append(Term(Opcode.HALT))
    return Code(global_context, start_code)


def translate(src: str) -> Code:
    tokens = extract_tokens(src)

    ast = build_ast(tokens)

    statements = extract_statements(ast)

    return translate_into_code(statements)


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
    lines = ["<address> - <length> - <data>\n"]
    for block in code.data_memory:
        addr = "0x{:03x}".format(block[0])
        lines.append(f"{addr} - {block[1]}\t- {block[2]}\n")
    return lines


def text_instr_memory(code: Code) -> list[str]:
    lines = ["<address> - <hexcode> - <mnemonica>\n"]
    for block in code.instr_memory:
        lines.append(f"{block[2]}:\n")
        base_addr = block[0]
        for i, term in enumerate(block[3]):
            addr = "0x{:03x}".format(base_addr + i)
            lines.append(f"{addr} - 0x{term_to_binary(term).hex()} - {term}\n")
    return lines


def code_to_text(code: Code) -> list[str]:
    lines = ["##### Data section #####\n"]
    lines.extend(text_data_memory(code))
    lines.append("\n##### Instruction section #####\n")
    lines.extend(text_instr_memory(code))
    return lines


def write_code(dst: str, code: Code):
    binary = code_to_binary(code)

    with open(dst, "wb") as f:
        f.write(binary)

    text = code_to_text(code)

    with open(dst + ".debug", "w") as f:
        f.writelines(text)

    return binary


def main(src: str, dst: str):
    with open(src, encoding="utf-8") as f:
        src = f.read()

    code = translate(src)

    binary = write_code(dst, code)

    print("LoC:", len(src.split("\n")), "code byte:", len(binary), "code instr:", len(code))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Usage: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
