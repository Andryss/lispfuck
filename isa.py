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
    BRANCH_EQUAL = ("bre", 10)
    BRANCH_GREATER_EQUAL = ("brge", 11)
    BRANCH_ANY = ("br", 12)

    INCREMENT = ("inc", 13)
    DECREMENT = ("dec", 14)

    MODULO = ("mod", 15)
    ADD = ("add", 16)
    SUBTRACT = ("sub", 17)
    MULTIPLY = ("mul", 18)
    DIVIDE = ("div", 19)
    INVERSE = ("inv", 20)


no_arg_ops: list[Opcode] = [Opcode.HALT, Opcode.RETURN, Opcode.PUSH, Opcode.POP, Opcode.POPN, Opcode.INVERSE]

addr_ops: list[Opcode] = [
    Opcode.STORE,
    Opcode.CALL,
    Opcode.BRANCH_EQUAL,
    Opcode.BRANCH_GREATER_EQUAL,
    Opcode.BRANCH_ANY,
    Opcode.INCREMENT,
    Opcode.DECREMENT,
]

value_ops: list[Opcode] = [
    Opcode.LOAD,
    Opcode.COMPARE,
    Opcode.MODULO,
    Opcode.ADD,
    Opcode.SUBTRACT,
    Opcode.MULTIPLY,
    Opcode.DIVIDE,
]

branch_ops: list[Opcode] = [Opcode.BRANCH_EQUAL, Opcode.BRANCH_GREATER_EQUAL, Opcode.BRANCH_ANY]

opcode_by_code: dict[int, Opcode] = {op.value[1]: op for op in Opcode}


class AddressType(Enum):
    EXACT = ("#", 1)
    ABSOLUTE = ("*", 2)
    RELATIVE_IPR = ("*ipr", 3)
    RELATIVE_SPR = ("*spr", 4)
    RELATIVE_INDIRECT_SPR = ("**spr", 5)


offset_addresses: list[AddressType] = [
    AddressType.RELATIVE_IPR,
    AddressType.RELATIVE_SPR,
    AddressType.RELATIVE_INDIRECT_SPR,
]

address_by_code: dict[int, AddressType] = {adr.value[1]: adr for adr in AddressType}


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
    def __init__(self, op: Opcode, arg: str | int | Address | None = None, desc: str | None = None):
        self.op = op
        self.arg = arg
        self.desc = desc

    def __str__(self):
        s = f"{self.op.value[0]}"
        if self.arg:
            s += f" {self.arg}"
        if self.desc:
            s += f"\t({self.desc})"
        return s.expandtabs(20)
