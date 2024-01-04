from __future__ import annotations

from enum import Enum


class Opcode(Enum):
    NOOP = ("noop", 0)
    HALT = ("halt", 1)

    LOAD = ("ld", 2)
    STORE = ("st", 3)

    CALL = ("call", 4)
    RETURN = ("ret", 5)

    PUSH = ("push", 6)
    POP = ("pop", 7)
    POPN = ("popn", 8)

    COMPARE = ("cmp", 9)
    BRANCH_ZERO = ("brz", 10)
    BRANCH_ANY = ("br", 11)

    INCREMENT = ("inc", 12)
    MODULO = ("mod", 13)
    MULTIPLY = ("mul", 14)
    DIVIDE = ("div", 15)
    SUBTRACT = ("sub", 16)


no_arg_ops: list[Opcode] = [
    Opcode.HALT, Opcode.RETURN, Opcode.PUSH, Opcode.POP, Opcode.POPN
]

addr_ops: list[Opcode] = [
    Opcode.STORE, Opcode.CALL, Opcode.BRANCH_ZERO, Opcode.BRANCH_ANY,
    Opcode.INCREMENT
]

value_ops: list[Opcode] = [
    Opcode.LOAD, Opcode.COMPARE, Opcode.MODULO, Opcode.MULTIPLY,
    Opcode.DIVIDE, Opcode.SUBTRACT
]


class AddressType(Enum):
    EXACT = ("#", 1)
    ABSOLUTE = ("*", 2)
    RELATIVE_IPR = ("*ipr", 3)
    RELATIVE_SPR = ("*spr", 4)
    RELATIVE_INDIRECT_SPR = ("**spr", 5)


offset_addresses: list[AddressType] = [
    AddressType.RELATIVE_IPR, AddressType.RELATIVE_SPR,
    AddressType.RELATIVE_INDIRECT_SPR
]


class Address:
    def __init__(self, kind: AddressType, val: int):
        self.kind = kind
        self.val = val

    def __str__(self):
        val_str: str = hex(self.val)
        if self.kind in offset_addresses:
            if self.val > 0:
                val_str = "+" + val_str
            elif self.val == 0:
                val_str = ""
        return f"{self.kind.value[0]}{val_str}"


class Term:
    def __init__(self, op: Opcode, arg: str | Address | None = None):
        self.op = op
        self.arg = arg

    def __str__(self):
        if self.arg is None:
            return f"{self.op.value[0]}"
        return f"{self.op.value[0]} {self.arg}"
