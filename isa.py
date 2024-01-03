from __future__ import annotations

from enum import Enum


class Opcode(Enum):
    HALT = "halt"

    LOAD = "ld"
    STORE = "st"

    CALL = "call"
    RETURN = "ret"

    PUSH = "push"
    POP = "pop"
    POPN = "popn"

    COMPARE = "cmp"
    BRANCH_ZERO = "brz"
    BRANCH_ANY = "br"

    INCREMENT = "inc"
    MODULO = "mod"
    MULTIPLY = "mul"
    DIVIDE = "div"
    SUBTRACT = "sub"


class AddressType(Enum):
    EXACT = "#"
    ABSOLUTE = "*"
    RELATIVE_IPR = "*ipr"
    RELATIVE_SPR = "*spr"
    RELATIVE_INDIRECT_SPR = "**spr"


class Address:
    def __init__(self, kind: AddressType, val: int):
        self.kind = kind
        self.val = val

    def __str__(self):
        return f"{self.kind.value}{self.val}"


class Term:
    def __init__(self, op: Opcode, arg: str | Address | None = None):
        self.op = op
        self.arg = arg

    def __str__(self):
        if self.arg is None:
            return f"{self.op.value}"
        return f"{self.op.value} {self.arg}"
