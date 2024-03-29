from __future__ import annotations

from isa import Address, AddressType, Opcode, Term

INPUT_PORT = 5555
OUTPUT_PORT = 5556


class FuncInfo:
    def __init__(self, name: str, argc: int = 0, code: list[Term] | None = None):
        self.name = name
        self.argc = argc
        self.code = [] if code is None else code


"""Print string in stdout.
Accept one string argument.
Returns amount of printed symbols.
"""
PRINT_LIMIT = 128
PRINT_FUNC = FuncInfo(
    "print",
    1,
    [
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.EXACT, 0)),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_INDIRECT_SPR, 1)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
        Term(Opcode.BRANCH_EQUAL, Address(AddressType.RELATIVE_IPR, 8)),
        Term(Opcode.STORE, Address(AddressType.ABSOLUTE, OUTPUT_PORT)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 1)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, PRINT_LIMIT)),
        Term(Opcode.BRANCH_EQUAL, Address(AddressType.RELATIVE_IPR, 2)),
        Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -9)),
        Term(Opcode.POP),
        Term(Opcode.POPN),
        Term(Opcode.RETURN),
    ],
)

"""Print integer in stdout.
Accept one integer argument.
Returns amount of printed symbols.
"""
PRINT_INTEGER_FUNC = FuncInfo(
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
        Term(Opcode.BRANCH_GREATER_EQUAL, Address(AddressType.RELATIVE_IPR, 5)),
        Term(Opcode.LOAD, Address(AddressType.EXACT, ord("-"))),
        Term(Opcode.STORE, Address(AddressType.ABSOLUTE, OUTPUT_PORT)),
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
        Term(Opcode.BRANCH_EQUAL, Address(AddressType.RELATIVE_IPR, 3)),
        Term(Opcode.STORE, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -9)),
        Term(Opcode.POP),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 2)),
        Term(Opcode.CALL, PRINT_FUNC.name),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 1)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, 0)),
        Term(Opcode.BRANCH_GREATER_EQUAL, Address(AddressType.RELATIVE_IPR, 2)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.POP),
        Term(Opcode.POPN),
        Term(Opcode.RETURN),
    ],
)

"""Print character in stdout.
Accept char code.
Returns amount of printed symbols (1).
"""
PRINT_CHAR_FUNC = FuncInfo(
    "printc",
    1,
    [
        Term(Opcode.STORE, Address(AddressType.ABSOLUTE, OUTPUT_PORT)),
        Term(Opcode.LOAD, Address(AddressType.EXACT, 1)),
        Term(Opcode.RETURN),
    ],
)

"""Print newline character (\\n) in stdout.
Accept no arguments.
Returns amount of printed symbols (1).
"""
PRINT_NEWLINE_FUNC = FuncInfo(
    "printline",
    0,
    [
        Term(Opcode.LOAD, Address(AddressType.EXACT, ord("\n"))),
        Term(Opcode.STORE, Address(AddressType.ABSOLUTE, OUTPUT_PORT)),
        Term(Opcode.LOAD, Address(AddressType.EXACT, 1)),
        Term(Opcode.RETURN),
    ],
)

"""Read string from stdout (until newline character).
Accept no arguments.
Returns string.
"""
READ_LIMIT = 128
READLINE_FUNC = FuncInfo(
    "readline",
    0,
    [
        Term(Opcode.PUSH),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.EXACT, 0)),
        Term(Opcode.PUSH),
        Term(Opcode.LOAD, Address(AddressType.ABSOLUTE, INPUT_PORT)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, ord("\n"))),
        Term(Opcode.BRANCH_EQUAL, Address(AddressType.RELATIVE_IPR, 8)),
        Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 1)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 1)),
        Term(Opcode.INCREMENT, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.LOAD, Address(AddressType.RELATIVE_SPR, 0)),
        Term(Opcode.COMPARE, Address(AddressType.EXACT, READ_LIMIT)),
        Term(Opcode.BRANCH_EQUAL, Address(AddressType.RELATIVE_IPR, 2)),
        Term(Opcode.BRANCH_ANY, Address(AddressType.RELATIVE_IPR, -9)),
        Term(Opcode.LOAD, Address(AddressType.EXACT, ord("\0"))),
        Term(Opcode.STORE, Address(AddressType.RELATIVE_INDIRECT_SPR, 1)),
        Term(Opcode.POP),
        Term(Opcode.POP),
        Term(Opcode.POP),
        Term(Opcode.RETURN),
    ],
)

"""Read char from stdout.
Accept no arguments.
Returns int char code.
"""
READCHAR_FUNC = FuncInfo(
    "readchar",
    0,
    [
        Term(Opcode.LOAD, Address(AddressType.ABSOLUTE, INPUT_PORT)),
        Term(Opcode.RETURN),
    ],
)


ALL_FUNCS = [PRINT_FUNC, PRINT_INTEGER_FUNC, PRINT_CHAR_FUNC, PRINT_NEWLINE_FUNC, READLINE_FUNC, READCHAR_FUNC]
